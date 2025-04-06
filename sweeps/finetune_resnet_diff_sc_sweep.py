import argparse
import wandb


def get_sweep_conf():
    sweep_configuration = {
        'method': 'grid',
        'name': f'fine tune resnet with different batch sizes in 8 to 64 and sc 99 sweep',
        'program': 'finetune_resnet.py',
        'description': '',
        'parameters': {
            'seed': {
                'value': 0
            },
            'save_path': {
                'value': PATH_TO_SAVE
            },
            'save_name': {
                'value': 'model_state_dict'
            },
            'base_directory': {
                'value': BASE_DIRECTORY_OF_IMAGES
            },
            'epochs': {
                'values': 100
            },
            'wd': {
                'value' : 1e-3
            },
            'lr': {
                'value' : 1e-3
            },
            'batch_size': {
                'values' : [8, 16, 24, 32, 40, 48, 56, 64]
            },
            'spurious_correlation_value': {
                'value': 99
            },
            'tracked_metrics': {
                'value': str(
                    [('mean_accuracy', 1), ('our_mean_accuracy', 1), ('worst_group_accuracy', 1)]
                )
            }
        }
    }
    return sweep_configuration


def main():
    config = get_sweep_conf()
    wandb.sweep(
        config,
        project=f"Resnet50 on Waterbirds",
    )


if __name__ == '__main__':
    main()