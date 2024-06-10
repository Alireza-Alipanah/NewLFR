import argparse
import os

import torch
from torchvision.models import resnet50
import torch.nn as nn
from torch.optim import SGD
import wandb
import tqdm
from sklearn.metrics import accuracy_score

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
parser.add_argument('--save_path', default=os.path.join('/', 'kaggle', 'working', 'NewLFR', 'saved_models'), help='path to save the model', type=str)
parser.add_argument('--save_name', default='resnet50_waterbirds.pth', help='name of the saved model file', type=str)
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
    y = torch.rand((0))
    for _, batch_y, _, _ in train_dataloader:
        y = torch.concat((y, batch_y))
    model = model.cuda()
    for epoch in tqdm.tqdm(range(args.epochs)):
        model.train()
        running_loss = 0
        outputs = torch.rand((0))
        for img, batch_y, _, _ in train_dataloader:
            img, batch_y = img.cuda(), y.float().cuda()
            optimizer.zero_grad()
            batch_outputs = model(img).squeeze()
            loss = criterion(batch_outputs, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            wandb.log({'train/step_loss': loss.item()})
            outputs = torch.concat((outputs, batch_outputs))
        running_loss /= len(train_dataloader)
        acc = accuracy_score(y, outputs)
        wandb.log({
            'epoch': epoch,
            'train/avg_loss': running_loss,
            'train/epoch_accuracy': acc
            })



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
    os.makedirs(args.save_path, exist_ok=True)
    torch.save(model_state, os.path.join(args.save_path, args.save_name))
