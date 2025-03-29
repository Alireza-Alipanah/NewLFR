import torch
import tqdm
import wandb

from utils import (
    calculate_group_accuracies,
    log_to_wandb,
    convert_to_wandb_format
)


class DefaultRunner:
    def __init__(
            self,
            args,
            train_dataloader,
            val_dataloader,
            test_dataloader,
            model,
            optimizer,
            criterion,
            classification_func,
            group_proportions,
            logger,
            **kwargs, # NOT USED, ONLY ADDED FOR EASE OF USE
        ):
        self.args = args
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.classification_func = classification_func
        self.group_proportions = group_proportions
        self.logger = logger

    
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

    
    def train(self):
        for epoch in tqdm(range(self.args.epochs)):
            self.model.train()
            train_running_loss = 0
            for img, y, _, _ in self.train_dataloader:
                y = y.long() if self.args.use_softmax else y.float()
                img, y = img.cuda(), y.cuda()
                self.optimizer.zero_grad()
                loss_value = self.updae_gradient(img, y)
                self.optimizer.step()
                train_running_loss += loss_value
                wandb.log({'train/step_loss': loss_value})
            self.evaluate_epoch(train_running_loss, epoch)

    
    def finish(self):
        self.logger.save_model(self.model, 'last_epoch')
        self.logger.write_logs()
        self.logger.log_to_wandb()



class OptimalDirectionRunner(DefaultRunner):
    def __init__(
            self,
            args,
            train_dataloader,
            val_dataloader,
            test_dataloader,
            model,
            optimizer,
            criterion,
            classification_func,
            group_proportions,
            logger,
            oprimal_direction_lr,
            optimal_direction_tol,
            optimal_direction_max_iter,
            **kwargs
        ):
        super().__init__(
            args=args,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            test_dataloader=test_dataloader,
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            classification_func=classification_func,
            group_proportions=group_proportions,
            logger=logger,
            **kwargs
        )
        self.oprimal_direction_lr = oprimal_direction_lr
        self.optimal_direction_tol = optimal_direction_tol
        self.optimal_direction_max_iter = optimal_direction_max_iter

    
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
    