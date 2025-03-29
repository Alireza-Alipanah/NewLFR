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
            logger
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
        if apply_criterion is not None:
            running_loss = self.criterion(outputs, y.long())
        outputs = outputs
        pred = self.classification_func(outputs).numpy()
        y = y.cpu().int().numpy()
        groups = groups.cpu().numpy()
        group_data = calculate_group_accuracies(pred, y, groups, self.group_proportions)
        if apply_criterion is not None:
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
        loss.backward()
        return loss

    
    def train_model(self):
        for epoch in tqdm(range(self.args.epochs)):
            self.model.train()
            train_running_loss = 0
            for img, y, _, _ in self.train_dataloader:
                y = y.long() if self.args.use_softmax else y.float()
                img, y = img.cuda(), y.cuda()
                self.optimizer.zero_grad()
                loss = self.updae_gradient(img, y)
                self.optimizer.step()
                train_running_loss += loss.item()
                wandb.log({'train/step_loss': loss.item()})
            self.evaluate_epoch(train_running_loss, epoch)
    