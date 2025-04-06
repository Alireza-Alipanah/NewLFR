import torch
import torch.nn.functional as F
from tqdm import tqdm
import wandb

from utils import (
    calculate_group_accuracies,
    log_to_wandb,
    convert_to_wandb_format,
    get_criterion
)


class Config:
    def __init__(self, params):
        for key, value in params.items():
            setattr(self, key, value)
        self.implicit_regularization_every_counter = 0


class DefaultRunner(Config):
    def __init__(self, **kwargs):
        super().__init__(params=kwargs)

    def evaluate(
            self,
            dataloader,
            apply_criterion=False # TODO make this part cleaner
        ):
        self.model.eval()
        y = torch.rand((0))
        outputs = torch.rand((0))
        groups = torch.rand((0))
        with torch.no_grad():
            for img, batch_y, g, _ in dataloader:
                img = img.cuda()
                batch_outputs = self.model(img).squeeze().cpu()
                outputs = torch.concat((outputs, batch_outputs))
                y = torch.concat((y, batch_y))
                groups = torch.concat((groups, g))
        if apply_criterion:
            running_loss = self.criterion(outputs, y.long())
        outputs = outputs
        pred = self.classification_func(outputs).numpy()
        y = y.cpu().int().numpy()
        groups = groups.cpu().numpy()
        group_data = calculate_group_accuracies(pred, y, groups, self.group_proportions)
        if apply_criterion:
            return group_data, running_loss
        return group_data


    def test_initial_model(self):
        group_data = self.evaluate(self.test_dataloader)
        prefix = 'test_start'
        self.logger.add_test_data(prefix, group_data)
        log_str = 'initial test metrics:\n\t' + '\n\t'.join([f'{key}: {value}' for key, value in group_data.items()])
        print(log_str)


    def evaluate_epoch(self, train_running_loss, epoch):
        train_group_data = self.evaluate(self.train_dataloader)
        validation_group_data, validation_running_loss = self.evaluate(self.val_dataloader, apply_criterion=True)
        test_data = self.evaluate(self.test_dataloader)
        train_running_loss /= len(self.train_dataloader)
        validation_running_loss /= len(self.val_dataloader)
        train_data = {'avg_loss': train_running_loss, **train_group_data}
        validation_data = {'avg_loss': validation_running_loss, **validation_group_data}
        log_to_wandb(
            {
                **convert_to_wandb_format('train', train_data),
                **convert_to_wandb_format('validation', validation_data),
                **convert_to_wandb_format('test', test_data),
            },
            epoch=epoch
        )
        self.logger.update(train_data, validation_data, test_data, self.model, epoch)
        self.logger.write_logs()
        self.logger.save_model(self.model, 'last_epoch')


    def updae_gradient(self, inputs, targets):
        outputs = self.model(inputs).squeeze()
        loss = self.criterion(outputs, targets)
        loss_value = loss.item()
        loss.backward()
        return loss_value
    

    def get_gradients_vector(self):
        """
            returns a flattened tensor of model gradient values
        """
        grad_list = []
        for param in self.model.parameters():
            if param.grad is not None:
                flat_grad = param.grad.flatten()
                grad_list.append(flat_grad)
        return torch.cat(grad_list)
    

    def calculate_implicit_regularization(self):
        """
        cacluated according to the paper:
            ON THE ORIGIN OF IMPLICIT REGULARIZATION IN STOCHASTIC GRADIENT DESCENT
        """
        self.model.eval()
        criterion = get_criterion(reduction='sum')
        # calculate accumulated gradient on all of data
        for img, batch_y, _, _ in self.val_dataloader:
            img = img.cuda()
            batch_outputs = self.model(img).squeeze().cpu()
            outputs = torch.concat((outputs, batch_outputs))
            loss = criterion(outputs, batch_y.long())
            loss.backward()
        full_batch_grad = self.get_gradients_vector()

        running_loss = 0
        implicit_regularization_values = []

        # calculate gradient on each batch
        for img, batch_y, _, _ in self.val_dataloader:
            img = img.cuda()
            batch_outputs = self.model(img).squeeze().cpu()
            outputs = torch.concat((outputs, batch_outputs))
            loss = criterion(outputs, batch_y.long())
            loss.backward()
            running_loss += loss.item()
            batch_grad = self.get_gradients_vector()
            implicit_regularization_values.append(
                torch.sum((full_batch_grad - batch_grad).pow(2))
            )

        return torch.mean(implicit_regularization_values), running_loss / len(self.val_dataloader)
    

    def calculate_and_log_step_metrics(self, train_loss_value):
        step_log_data = {'train/step_loss': train_loss_value,}
        if self.implicit_regularization:
            self.implicit_regularization_every_counter += 1
            if self.implicit_regularization_every_counter == self.implicit_regularization_every:
                self.implicit_regularization_every_counter = 0
                validation_step_implicit_regularization, validation_step_loss = self.calculate_implicit_regularization()
                step_log_data['train/step_validation_implicit_regularization'] = validation_step_implicit_regularization
                step_log_data['train/step_validation_loss'] = validation_step_loss
        wandb.log(step_log_data)

    
    def train(self):
        for epoch in tqdm(range(self.args.epochs)):
            train_running_loss = 0
            for img, y, _, _ in self.train_dataloader:
                self.model.train() 
                y = y.long()
                img, y = img.cuda(), y.cuda()
                self.optimizer.zero_grad()
                loss_value = self.updae_gradient(img, y)
                self.optimizer.step()
                train_running_loss += loss_value
                self.calculate_and_log_step_metrics(train_loss_value=loss_value)
            self.evaluate_epoch(train_running_loss, epoch)

    
    def finish(self):
        self.logger.save_model(self.model, 'last_epoch')
        self.logger.write_logs()
        self.logger.log_to_wandb()



class OptimalDirectionRunner(DefaultRunner):    
    def calculate_optimal_direction(self, vectors):
        """
        Finds a unit vector v that:
        1. Has a positive inner product with all given vectors in `vectors`.
        2. Maximizes the sum of inner products with all vectors while satisfying the first condition.

        Parameters:
            vectors (torch.Tensor): A batch of n vectors, each of dimension d, representing per-sample gradients.
            oprimal_direction_lr (float, default=0.1): Learning rate for updating the direction vector.
            optimal_direction_tol (float, default=1e-6): Tolerance for convergence (stopping criterion based on change in v).
            optimal_direction_max_iter (int, default=10): Maximum number of iterations to adjust v.

        Returns:
            torch.Tensor: The optimized unit vector v that aligns positively with all input vectors 
                        while maximizing the total inner product.
        """
        s = vectors.sum(dim=0)
        v = s / s.norm(p=2)

        for _ in range(self.optimal_direction_max_iter):
            ips = torch.matmul(vectors, v)
            viol = ips <= 0
            if not viol.any():
                break
            correction = vectors[viol].sum(dim=0)
            v_new = v + self.oprimal_direction_lr * correction
            v_new = v_new / v_new.norm(p=2)
            if torch.norm(v - v_new, p=2) < self.optimal_direction_tol:
                break
            v = v_new

        return v
    

    def updae_gradient(self, inputs, targets):
        outputs = self.model(inputs)
        loss = self.criterion(outputs, targets)
        loss_value = loss.item()
        
        # Compute per-sample gradients
        grads = torch.autograd.grad(loss, self.model.parameters(), retain_graph=True, create_graph=True)
        
        # Stack gradients (each row corresponds to a sample)
        grad_matrix = torch.stack([g.flatten() for g in grads], dim=1).T  # Shape: (batch_size, num_params)
        
        # Compute the optimal update direction
        optimal_v = self.calculate_optimal_direction(grad_matrix)

        # Apply the optimal gradient update
        with torch.no_grad():
            idx = 0
            for param in self.model.parameters():
                param.grad = -optimal_v[idx : idx + param.numel()].reshape(param.shape)  # Negate direction
                idx += param.numel()

        return loss_value
    