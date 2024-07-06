import argparse
import wandb


def get_sweep_conf():
    sweep_configuration = {
        'method': 'grid',
        'name': f'DFR on Waterbirds',
        'program': 'lfr_train.py',
        'description': '',
        'parameters': {
            'embeddings_dir': {
                'value': '/kaggle/input/wild-violet-60-embeddings/wild-violet-60.csv'
            },
            'use_softmax': {
                'value': True
            },
            'model_path': {
                'value': '/kaggle/input/resnet50-waterbirds-wild-violet-60/NewLFR/saved_models/resnet50_waterbirds_wild-violet-60.pth',
            },
            'C': {
                'values': [1.0, 0.7, 0.3, 0.1, 0.07, 0.03, 0.01]
            },
            'method': {
                'value': 'DFR'
            },
            'use_standard_scaler': {
                'value': True
            },
            'subsample': {
                'value': True
            },
            'eiil': {
                'value': True
            },
            # 'seed': {
            #     'values': [0, 1, 2, 3, 4]
            # }
        }
    }
    return sweep_configuration


def main():
    config = get_sweep_conf()
    wandb.sweep(
        config,
        project=f"DFR on Waterbirds",
    )


if __name__ == '__main__':
    main()