import argparse
import os
from functools import partial

import torch
import wandb

from utils import (
    load_data,
    load_convex_combination_data,
    load_dsprite_data,
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
    parser.add_argument('--spurious_correlation_value', default=None, help='spurious correlation value to build the dataset with, use None for default', type=int)
    parser.add_argument('--tracked_metrics', default=[('mean_accuracy', 1), ('worst_group_accuracy', 1)], help='list of metrics to track. ' + \
    ' each should be of the form (metric name, 1|-1) with 1 indiciation higher is better', type=lambda x: eval(x))


    # Expermients DataLoaders
    parser.add_argument('--experiment', default='NormalWaterbirds', help='datset to load for experiments', type=str,
        choices=[
            'NormalWaterbirds',
            'CifarMnistConvexCombination',
            'DSprite'
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

    args = parser.parse_args()
    return args


def adjust_for_experiment(args):
    # The goal of this function is to apply all experiment specific changes in one functions to facilitate future adjustments
    run_name = wandb.run.name if wandb.run.name else wandb.run.id
    if args.experiment == 'NormalWaterbirds':
        load_data_func = partial(load_data, basedir=args.base_dir, batch_size=args.batch_size, spurious_correlation_value=args.spurious_correlation_value)
        args.save_path = os.path.join(args.save_path, run_name)
        print(f'saving logs at {args.save_path}')
    elif args.experiment == 'CifarMnistConvexCombination':
        mapping_csv_path = os.path.join(args.cc_mapping_basedir, f'{args.cc_correlation}_mapping.csv')
        load_data_func = partial(load_convex_combination_data, mapping_csv_path=mapping_csv_path,
         alpha=args.cc_alpha, correlation=args.cc_correlation, batch_size=args.batch_size)
        args.save_path = os.path.join(args.save_path, f'alpha_{args.cc_alpha}', f'seed_{args.seed}', f'correlation_{int(float(args.cc_correlation) * 100)}')
    elif args.experiment == 'DSprite':
        load_data_func = partial(load_dsprite_data, args=args, basedir=args.base_dir, batch_size=args.batch_size,
                                  spurious_correlation_value=args.spurious_correlation_value / 100)
        args.save_path = os.path.join(args.save_path, 'DSprite', f'seed_{args.seed}', f'sc_{args.spurious_correlation_value}', f'batch_size_{args.batch_size}')
    args.save_name = f'{args.save_name}_{run_name}'
    return load_data_func


def get_runner_arguments(args):
    load_data_func = adjust_for_experiment(args)
    train_dataloader, val_dataloader, test_dataloader = load_data_func()
    model = get_model(args, train_dataloader.dataset.n_classes)
    model = model.cuda()
    optimizer = get_optimizer(args, model.parameters())
    criterion = get_criterion(args)
    classification_func = get_classification_func(args)
    group_proportions = get_train_group_proportions(train_dataloader.dataset.group_array)
    logger = Logger(args=args, tracked_metrics=args.tracked_metrics, path=args.save_path, save_name=args.save_name)
    runner_arguments = {
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
        'oprimal_direction_lr': args.runner_oprimal_direction_lr,
        'optimal_direction_tol': args.runner_optimal_direction_tol,
        'optimal_direction_max_iter': args.runner_optimal_direction_max_iter
    }
    return runner_arguments



def get_runner(args, runner_arguments):
    if args.runner == 'DefaultRunner':
        runner_class = DefaultRunner
    elif args.runner == 'OptimalDirectionRunner':
        runner_class = OptimalDirectionRunner
    else:
        raise NotImplementedError()
    runner = runner_class(**runner_arguments)
    return runner
    


if __name__ == '__main__':
    args = get_args()
    set_seed(args)
    wandb.init(
        project=args.project_name,
        config=vars(args),
    )

    runner_arguments = get_runner_arguments(args)
    runner = get_runner(args, runner_arguments)
    runner.test_initial_model()
    runner.train()
    runner.finish()

    wandb.finish()
