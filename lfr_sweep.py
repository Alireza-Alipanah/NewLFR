import argparse
import wandb


def get_sweep_conf():
    sweep_configuration = {
        'method': 'grid',
        'name': f'LFR on Waterbirds with indiscriminate sample choosing',
        'program': 'lfr_train.py',
        'description': '',
        'parameters': {
            'embeddings_dir': {
                'value': '/kaggle/input/faithful-universe-57-embeddings/faithful-universe-57.csv'
            },
            'use_softmax': {
                'value': True
            },
            'model_path': {
                'value': '/kaggle/input/resnet50-waterbirds-faithful-universe-57/NewLFR/saved_models/resnet50_waterbirds_faithful-universe-57.pth',
            },
            'number_to_pick': {
                'values': [5] + list(range(10, 169, 10))
            },
            'C': {
                'values': [1.0, 0.7, 0.3, 0.1, 0.07, 0.03, 0.01]
            },
        }
    }
    return sweep_configuration


def main():
    config = get_sweep_conf()
    wandb.sweep(
        config,
        project=f"LFR on Waterbirds",
    )


if __name__ == '__main__':
    main()