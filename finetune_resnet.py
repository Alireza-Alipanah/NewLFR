import argparse
import os
from functools import partial

import torch
import wandb

from utils import (
    load_data,
    load_convex_combination_data,
    train_model,
    test_model,
    set_seed,
    get_resnet50_weights,
    get_model,
    get_optimizer,
    get_criterion,
    get_classification_func,
    get_train_group_proportions,
)
from logger import Logger

if not torch.cuda.is_available():
    raise Exception('cuda not available')


def get_args():
    parser = argparse.ArgumentParser('Pretraining Resnet50')
    parser.add_argument('--base_dir', default='/kaggle/working/waterbird_complete95_forest2water2', help='base directory of images', type=str)
    parser.add_argument('--optimizer', default='SGD', help='optimizer to use', type=str)
    parser.add_argument('--momentum_decay', default=0.9, help='momentum decay of optimizer', type=float)
    parser.add_argument('--wd', default=1e-3, help='weight decay of optimizer', type=float)
    parser.add_argument('--lr', default=1e-3, help='learning rate of optimizer', type=float)
    parser.add_argument('--batch_size', default=128, help='data loader batch size', type=int)
    parser.add_argument('--epochs', default=100, help='number of epochs', type=int)
    parser.add_argument('--model', default='ResNet50', help='model to train', type=str)
    parser.add_argument('--weights', default='ResNet50_Weights.IMAGENET1K_V1', help='specify the weights to use for the pretrained model', type=str)
    parser.add_argument('--save_path', default=os.path.join('/', 'kaggle', 'working', 'NewLFR', 'saved_models'), help='path to save the model', type=str)
    parser.add_argument('--save_name', default='resnet50_waterbirds', help='name of the saved model file without .pth extension', type=str)
    parser.add_argument('--use_softmax', default=False, help='wether to use softmax or sigmoid', type=bool)
    parser.add_argument('--seed', default=None, help='seed', type=int)
    parser.add_argument('--spurious_correlation_value', default=None, help='spurious correlation value to build the dataset with, use None for default', type=int)
    parser.add_argument('--tracked_metrics', default=[('mean_accuracy', 1)], help='list of metrics to track. each should be of the form (metric name, 1|-1) with 1 indiciation higher is better', type=lambda x: eval(x))

    # Expermients DataLoaders
    parser.add_argument('--experiment', default='NormalWaterbirds', help='datset to load for experiments', type=str,
        choices=[
            'NormalWaterbirds',
            'CifarMnistConvexCombination'
        ]
        )

    # convex combination experiments values
    parser.add_argument('--cc_alpha', default=0, help='alpha value for belding', type=float)
    parser.add_argument('--cc_mapping_basedir', default='', help='path to the base directory of csv that specifies mapping of CIFAR and MNIST images', type=str)
    parser.add_argument('--cc_correlation', default=0.65, help='correlation_value', type=str)
    args = parser.parse_args()
    return args


def adjust_for_experiment(args):
    # The goal of this function is to apply all experiment specific changes in one functions to facilitate future adjustments
    if args.experiment == 'NormalWaterbirds':
        load_data_func = partial(load_data, basedir=args.base_dir, batch_size=args.batch_size, spurious_correlation_value=args.spurious_correlation_value)
        args.save_path = os.path.join(args.save_path, wandb.run.name)
        print(f'saving logs at {args.save_path}')
    elif args.experiment == 'CifarMnistConvexCombination':
        mapping_csv_path = os.path.join(args.cc_mapping_basedir, f'{args.cc_correlation}_mapping.csv')
        load_data_func = partial(load_convex_combination_data, mapping_csv_path=mapping_csv_path,
         alpha=args.cc_alpha, correlation=args.cc_correlation, batch_size=args.batch_size)
        args.save_path = os.path.join(args.save_path, f'alpha_{args.cc_alpha}', f'seed_{args.seed}', f'correlation_{int(float(args.cc_correlation) * 100)}')
    return load_data_func


def save_model(args, model, log_str):
    model_state = model.state_dict()
    os.makedirs(args.save_path, exist_ok=True)
    torch.save(model_state, os.path.join(args.save_path, f'{args.save_name}.pth'))
    with open(os.path.join(args.save_path, f'{args.save_name}_last_epoch_test_logs'), 'w') as f:
        f.write(log_str + '\n\nwandb logs saved at:' + wandb.run.dir)


if __name__ == '__main__':
    args = get_args()
    set_seed(args)
    wandb.init(
        project='Resnet50 on Waterbirds',
        config=vars(args),
    )
    load_data_func = adjust_for_experiment(args)
    train_dataloader, val_dataloader, test_dataloader = load_data_func()
    model = get_model(args, train_dataloader.dataset.n_classes)
    model = model.cuda()
    optimizer = get_optimizer(args, model.parameters())
    criterion = get_criterion(args)
    classification_func = get_classification_func(args)
    group_proportions = get_train_group_proportions(train_dataloader.dataset.group_array)
    logger = Logger(tracked_metrics=args.tracked_metrics, path=args.save_path)
    test_model(
        test_dataloader=test_dataloader,
        model=model,
        classification_func=classification_func,
        is_start=True,
        group_proportions=group_proportions,
        logger=logger,
    )
    train_model(
        args=args,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader,
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        classification_func=classification_func,
        group_proportions=group_proportions,
        logger=logger,
    )
    log_str = test_model(
        test_dataloader=test_dataloader,
        model=model,
        classification_func=classification_func,
        group_proportions=group_proportions,
        logger=logger,
    )
    save_model(args, model, log_str)
    logger.write_logs()
    wandb.finish()
