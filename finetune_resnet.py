import argparse

import torch
from torchvision.models import resnet50
import torch.nn as nn
from torch.optim import SGD
import wandb
import tqdm

from utils import load_data


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
parser.add_argument('--pretrained', default=True, help='specify wether to use pretrained model', type=bool)
parser.add_argument('--save_path', default='/kaggle/working/resnet50_waterbirds.pth', help='path to save the model', type=str)
args = parser.parse_args()


def get_model(args, n_classes):
    if args.model == 'ResNet50':
        model = resnet50(pretrained=args.pretrained)
        model.fc = nn.Linear(in_features=2048, out_features=n_classes)
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


def train_model(
    args,
    train_dataloader,
    val_dataloader,
    test_dataloader,
    model,
    optimizer,
    criterion
):
    model = model.cuda()
    for epoch in tqdm.tqdm(range(args.epochs)):
        model.train()
        running_loss = 0
        for img, y, _, _ in train_dataloader:
            img, y = img.cuda(), y.float().cuda()
            optimizer.zero_grad()
            outputs = model(img).squeeze()
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            wandb.log({'train/step_loss': loss.item()})
        running_loss /= len(train_dataloader)
        wandb.log({'epoch': epoch, 'train/avg_loss': running_loss})



if __name__ == '__main__':
    train_dataloader, val_dataloader, test_dataloader = get_data(args)
    model = get_model(args, 1) # TODO make it better ?
    optimizer = get_optimizer(args, model.parameters())
    criterion = nn.BCEWithLogitsLoss()
    wandb.init(
        project='Resnet50 on Waterbirds',
        config=vars(args),
    )
    train_model(
        args=args,
        train_dataloader=test_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader,
        model=model,
        optimizer=optimizer,
        criterion=criterion
    )
    model_state = model.state_dict()
    toch.save(model_state, args.save_path)
