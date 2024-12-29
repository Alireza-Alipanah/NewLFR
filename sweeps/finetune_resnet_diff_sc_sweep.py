import argparse
import wandb


def get_sweep_conf():
    sweep_configuration = {
        'method': 'grid',
        'name': f'fine tune resnet',
        'program': 'finetune_resnet.py',
        'description': '',
        'parameters': {
            'use_softmax': {
                'value': True
            },
            'seed': {
                'value': 0
            },
            'save_path': {
                'value': PATH_TO_SAVE
            },
            'save_name': {
                'value': 'model_state_dict'
            },
            'experiment': {
                'value': 'CifarMnistConvexCombination'
            },
            'cc_alpha': {
                'values': [0, 0.2, 0.4, 0.8]
            },
            'cc_mapping_basedir': {
                'value' : '/home/user01/data/convex_combination_data_mappings'
            },
            'cc_correlation': {
                'values' : [0.65, 0.75, 0.85, 0.95, 0.99, 1.0]
            }
        }
    }
    return sweep_configuration


def main():
    config = get_sweep_conf()
    wandb.sweep(
        config,
        project=f"ResNet50 Fine-tune",
    )


if __name__ == '__main__':
    main()