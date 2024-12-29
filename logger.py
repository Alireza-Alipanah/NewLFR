import os
import pprint


class Logger:
    
    def __init__(self, tracked_metrics, path):
        self.tracked_metrics = tracked_metrics
        self.validation_data = {}
        self.test_data = {}
        self.train_data = {}
        self.path = path

    def update(self, train_data, validation_data, test_data):
        for metric, multiplier in self.tracked_metrics:
            print('--------------------', metric, multiplier)
            current = self.validation_data.get(metric, {metric: -multiplier * float('inf')})
            if current[metric] * multiplier < validation_data[metric] * multiplier:
                self.test_data[metric] = test_data
                self.validation_data[metric] = validation_data
                self.train_datat[metric] = train_data
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
        }
        with open(f'{self.path}/logs', 'w') as f:
            f.write(pprint.pformat(data, width=1))
