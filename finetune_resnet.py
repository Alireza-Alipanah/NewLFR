import argparse
import os
from functools import partial

import torch
import wandb

from utils import (
    set_seed,
    get_model,
    get_optimizer,
    get_criterion,
    get_classification_func,
    get_train_group_proportions,
)
from logger import Logger
from runner import (
    DefaultRunner,
    OptimalDirectionRunner
)
from experiment_configs import EXPERIMENT_CONFIGS

if not torch.cuda.is_available():
    raise Exception('cuda not available')


def get_args():
    parser = argparse.ArgumentParser('Pretraining Resnet50')
    parser.add_argument('--project_name', default='Resnet18 on DSprites', help='project name to use for wandb', type=str)
    parser.add_argument('--base_dir', default='/kaggle/working/waterbird_complete95_forest2water2', help='base directory of images', type=str)
    parser.add_argument('--optimizer', default='SGD', help='optimizer to use', type=str)
    parser.add_argument('--momentum_decay', default=0.9, help='momentum decay of optimizer', type=float)
    parser.add_argument('--wd', default=1e-3, help='weight decay of optimizer', type=float)
    parser.add_argument('--lr', default=1e-3, help='learning rate of optimizer', type=float)
    parser.add_argument('--batch_size', default=128, help='data loader batch size', type=int)
    parser.add_argument('--epochs', default=100, help='number of epochs', type=int)
    parser.add_argument('--model', default='ResNet50', help='model to train', type=str)
    parser.add_argument('--weights', default='ResNet50_Weights.IMAGENET1K_V1', help='specify the weights to use for the pretrained model', type=str)
    parser.add_argument('--continue_from_chechpoint', default=False, help='whether to continue from previous model', type=bool)
    parser.add_argument('--save_path', default=os.path.join('/', 'kaggle', 'working', 'NewLFR', 'saved_models'), help='path to save the model', type=str)
    parser.add_argument('--save_name', default='resnet50_waterbirds', help='name of the saved model file without .pth extension', type=str)
    parser.add_argument('--seed', default=None, help='seed', type=int)
    parser.add_argument('--spurious_correlation_value', default=None, help='spurious correlation value to build the train dataloader with, use None for default', type=int)
    parser.add_argument('--spurious_correlation_value_validation', default=None, help='spurious correlation value to build the validation dataloader with' + \
                        ', use None for default, does not apply to convex combination experiments', type=int)
    parser.add_argument('--same_spurious_correlation_value_for_train_and_validation', default=False, help='Whether to use the same spurious correlation value when' + \
                        ' create train and validation datasets. Overrides spurious_correlation_value_validation argument. Does not apply to convex combination experiments.', type=bool)
    parser.add_argument('--tracked_metrics', default=[('mean_accuracy', 1), ('worst_group_accuracy', 1)], help='list of metrics to track. ' + \
    ' each should be of the form (metric name, 1|-1) with 1 indiciation higher is better', type=lambda x: eval(x))


    # Expermients DataLoaders
    parser.add_argument('--experiment', default='NormalWaterbirds', help='datset to load for experiments', type=str,
        choices=[
            'NormalWaterbirds',
            'CifarMnistConvexCombination',
            'DSprite',
            'Dominos_Cmnist_all_exps'
        ]
    )

    # convex combination experiments values
    parser.add_argument('--cc_alpha', default=0, help='alpha value for belding', type=float)
    parser.add_argument('--cc_mapping_basedir', default='', help='path to the base directory of csv that specifies mapping of CIFAR and MNIST images', type=str)
    parser.add_argument('--cc_correlation', default=0.65, help='correlation_value', type=str)

    # runners arguemtns
    parser.add_argument('--runner', default='DefaultRunner', help='Runner to use for training the model. supported: DefaultRunner, OptimalDirectionRunner', type=str,
                        choices=[
                            'DefaultRunner',
                            'OptimalDirectionRunner',
                        ]
                        )
    parser.add_argument('--runner_oprimal_direction_lr', default=0.1, help='learning rate used in calculating optimal direction. only for OptimalDirectionRunner.', type=float)
    parser.add_argument('--runner_optimal_direction_tol', default=1e-6, help='tolerance used in calculating optimal direction. only for OptimalDirectionRunner.', type=float)
    parser.add_argument('--runner_optimal_direction_max_iter', default=10, help='max iterations used in calculating optimal direction. only for OptimalDirectionRunner.', type=int)

    # implicit regularization calculation
    parser.add_argument('--implicit_regularization', default=False, help='whether to calculate implicit regularization that happens with batching', type=bool)
    parser.add_argument('--implicit_regularization_every', default=1, help='specifizes calculting implicit regularization every i steps', type=int)

    # dominos_cmnist_all_exps
    parser.add_argument('--dataset_total_count', default=40000, help='total count of data points in the dataset', type=int)
    parser.add_argument('--color_correlation', default=0.95, help='correlation of color attribute', type=float)
    parser.add_argument('--shape_correlation', default=0.75, help='correlation of shape attribute', type=float)

    args = parser.parse_args()
    return args


def adjust_for_experiment(args):
    """
    Apply all experiment-specific changes: select the data loader, 
    build save_path, and append run name to save_name.
    """
    run_id = wandb.run.name or wandb.run.id
    if args.same_spurious_correlation_value_for_train_and_validation:
        args.spurious_correlation_value_validation = args.spurious_correlation_value
    cfg = EXPERIMENT_CONFIGS.get(args.experiment)
    if cfg is None:
        raise ValueError(f"Unknown experiment: {args.experiment}")
    loader_fn = cfg["loader"]
    loader_params = cfg["loader_kwargs"](args)
    load_data_func = partial(loader_fn, **loader_params)
    subdirs = cfg["path_parts"](args)
    args.save_path = os.path.join(args.save_path, *subdirs)
    print(f"Saving logs at {args.save_path}")
    args.save_name = f"{args.save_name}_{run_id}"
    return load_data_func


def get_runner_arguments(args):
    load_data_func = adjust_for_experiment(args)
    train_dataloader, val_dataloader, test_dataloader = load_data_func()
    model = get_model(args, train_dataloader.dataset.n_classes)
    model = model.cuda()
    optimizer = get_optimizer(args, model.parameters())
    criterion = get_criterion()
    classification_func = get_classification_func(args)
    group_proportions = get_train_group_proportions(train_dataloader.dataset.group_array)
    logger = Logger(args=args, tracked_metrics=args.tracked_metrics, path=args.save_path, save_name=args.save_name)
    runner_arguments = {
        **vars(args),
        'args': args,
        'train_dataloader': train_dataloader,
        'val_dataloader': val_dataloader,
        'test_dataloader': test_dataloader,
        'model': model,
        'optimizer': optimizer,
        'criterion': criterion,
        'classification_func': classification_func,
        'group_proportions': group_proportions,
        'logger': logger,
    }
    return runner_arguments



def get_runner_class(args):
    if args.runner == 'DefaultRunner':
        runner_class = DefaultRunner
    elif args.runner == 'OptimalDirectionRunner':
        runner_class = OptimalDirectionRunner
    else:
        raise NotImplementedError()
    return runner_class
    


if __name__ == '__main__':
    args = get_args()
    set_seed(args)
    wandb.init(
        project=args.project_name,
        config=vars(args),
    )

    runner_arguments = get_runner_arguments(args)
    runner_class = get_runner_class(args)
    runner = runner_class(**runner_arguments)
    runner.test_initial_model()
    runner.train()
    runner.finish()

    wandb.finish()
