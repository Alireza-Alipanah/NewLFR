import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
from torchvision.models import resnet50, ResNet50_Weights
import wandb

from utils import load_data, train_model, test_model, set_seed


if not torch.cuda.is_available():
    raise Exception('cuda not available')

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
parser.add_argument('--seed', default=None, help='seed', type=1)
args = parser.parse_args()


def get_resnet50_weights(args):
    if args.weights == 'ResNet50_Weights.IMAGENET1K_V1':
        weights = ResNet50_Weights.IMAGENET1K_V1
    elif args.weights == 'ResNet50_Weights.IMAGENET1K_V2':
        weights = ResNet50_Weights.IMAGENET1K_V2
    elif args.weights == 'None':
        weights = None
    else:
        raise NotImplementedError()
    return weights


def get_model(args, n_classes):
    if n_classes > 2 and (not args.use_softmax):
        raise Exception('Cannot use sigomid when n_classes > 2')
    if args.model == 'ResNet50':
        weights = get_resnet50_weights(args)
        model = resnet50(weights=weights)
        in_features = model.fc.in_features
        out_features = n_classes if args.use_softmax else 1
        model.fc = nn.Linear(in_features=in_features, out_features=out_features)
    else:
        raise NotImplementedError()
    return model


def get_data(args):
    train_dataloader, val_dataloader, test_dataloader = load_data(basedir=args.base_dir, batch_size=args.batch_size)
    return train_dataloader, val_dataloader, test_dataloader


def get_optimizer(args, params):
    if args.optimizer == 'SGD':
        optimizer = SGD(
            params=params,
            momentum=args.momentum_decay,
            weight_decay=args.wd,
            lr=args.lr,
        )
    else:
        raise NotImplementedError()
    return optimizer


def get_criterion(args):
    if args.use_softmax:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()
    return criterion


def get_classification_func(args, sigmoid_threshold=0.5):
    if args.use_softmax:
        return lambda x: torch.argmax(F.softmax(x, dim=1), dim=1)
    else:
        return lambda x: F.sigmoid(x.squeeze()) >= sigmoid_threshold


if __name__ == '__main__':
    if args.seed is not None:
        set_seed(args.seed)
    train_dataloader, val_dataloader, test_dataloader = get_data(args)
    model = get_model(args, train_dataloader.dataset.n_classes)
    model = model.cuda()
    optimizer = get_optimizer(args, model.parameters())
    criterion = get_criterion(args)
    classification_func = get_classification_func(args)
    wandb.init(
        project='Resnet50 on Waterbirds',
        config=vars(args),
    )
    test_model(
        test_dataloader=test_dataloader,
        model=model,
        classification_func=classification_func,
        is_start=True
    )
    train_model(
        args=args,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        classification_func=classification_func
    )
    test_model(
        test_dataloader=test_dataloader,
        model=model,
        classification_func=classification_func
    )
    model_state = model.state_dict()
    os.makedirs(args.save_path, exist_ok=True)
    torch.save(model_state, os.path.join(args.save_path, f'{args.save_name}_{wandb.run.name}.pth'))
    wandb.finish()
