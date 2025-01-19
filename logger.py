import os
import pprint

from utils import convert_to_wandb_format, log_to_wandb

import wandb
import torch


class Logger:
    
    def __init__(self, args, tracked_metrics, path, save_name):
        self.args = args
        self.tracked_metrics = tracked_metrics
        self.validation_data = {}
        self.test_data = {}
        self.train_data = {}
        self.path = path
        self.save_name = save_name

    def save_model(self, model, epoch_name):
        model_state = model.state_dict()
        os.makedirs(self.path, exist_ok=True)
        torch.save(model_state, os.path.join(self.path, f'{self.save_name}_{epoch_name}.pth'))

    def update(self, train_data, validation_data, test_data, model, ecpoh):
        for metric, multiplier in self.tracked_metrics:
            current = self.validation_data.get(metric, {metric: -multiplier * float('inf')})
            if current[metric] * multiplier < validation_data[metric] * multiplier:
                self.test_data[metric] = {'epoch': epoch, **test_data}
                self.validation_data[metric] = {'epoch': epoch, **validation_data}
                self.train_data[metric] = {'epoch': epoch, **train_data}
                self.save_model(model, metric)
        self.validation_data['last_epoch'] = validation_data
        self.test_data['last_epoch'] = test_data
        self.train_data['last_epoch'] = train_data

    def add_test_data(self, metric, test_data): # this function is only used to add initial and last epoch test data. TODO: make this cleaner
        self.test_data[metric] = test_data

    def write_logs(self):
        os.makedirs(self.path, exist_ok=True)
        data = {
            'test': self.test_data,
            'validation': self.validation_data,
            'train': self.train_data,
            'args': str(self.args),
        }
        with open(f'{self.path}/logs', 'w') as f:
            f.write(pprint.pformat(data, width=1))

    def log_to_wandb(self):
        for metric in self.test_data.keys():
            log_to_wandb(convert_to_wandb_format(f'test_{metric}', self.test_data[metric]))
        for metric in self.validation_data.keys():
            log_to_wandb(convert_to_wandb_format(f'validation_{metric}', self.validation_data[metric]))
