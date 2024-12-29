import os
import pprint


class Logger:
    
    def __init__(tracked_metrics, path):
        self.tracked_metrics = tracked_metrics
        self.validation_data = {}
        self.test_data = {}
        self.train_data = {}
        self.path = path

    def update(train_data, validation_data, test_data):
        if self.tracked_metrics is not None:
            for metric, multiplier in self.tracked_metrics:
                current = self.validation_data.get(metric, {metric: multiplier * float('inf')})
                if current * multiplier < validation_data[metric] * multiplier:
                    self.test_data[metric] = test_data
                    self.validation_data[metric] = validation_data
                    self.train_datat[metric] = train_data
        self.validation_data['last_epoch'] = validation_data
        self.test_data['last_epoch'] = test_data
        self.train_data['last_epoch'] = train_data

    def write_logs():
        os.makedirs(self.path, exist_ok=True)
        data = {
            'test': self.test_data,
            'validation': self.validation_data,
            'train': self.train_data,
        }
        with open(f'{self.path}/logs', 'w') as f:
            f.write(pprint.pformat(data, width=1))
