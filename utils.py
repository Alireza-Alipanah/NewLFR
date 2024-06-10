from wb_data import WaterBirdsDataset, get_transform_cub, get_loader, log_data

from sklearn.metrics import accuracy_score
import numpy as np
import tqdm
import wandb
import torch


def load_data(basedir, batch_size, target_resolution=(224, 224)):
    train_transform = get_transform_cub(target_resolution=target_resolution, train=True, augment_data=True)
    val_transform = get_transform_cub(target_resolution=target_resolution, train=False, augment_data=True) # TODO augment or not ?
    test_transform = get_transform_cub(target_resolution=target_resolution, train=False, augment_data=False)

    train_dataset = WaterBirdsDataset(
        basedir=basedir,
        split='train',
        transform=train_transform
        )
    val_dataset = WaterBirdsDataset(
        basedir=basedir,
        split='val',
        transform=val_transform
        )
    test_dataset = WaterBirdsDataset(
        basedir=basedir,
        split='test',
        transform=test_transform
        )

    train_dataloader = get_loader(
        data=train_dataset,
        train=True,
        reweight_groups=None,
        reweight_classes=None,
        reweight_places=None,
        batch_size=batch_size,
        )
    val_dataloader = get_loader(
        data=val_dataset,
        train=False,
        reweight_groups=None,
        reweight_classes=None,
        reweight_places=None,
        batch_size=batch_size,
        )
    test_dataloader = get_loader(
        data=test_dataset,
        train=False,
        reweight_groups=None,
        reweight_classes=None,
        reweight_places=None,
        batch_size=batch_size,
        )
    return train_dataloader, val_dataloader, test_dataloader


def log_to_wandb(prefix, data_dict, epoch=None):
    new_data = {f'{prefix}/{key}': value for key, value in data_dict.items()}
    if epoch is not None:
        new_data['epoch'] = epoch
    wandb.log(new_data)


def train_model(
    args,
    train_dataloader,
    val_dataloader,
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
        group_data = evaluate(train_dataloader, model)
        log_to_wandb(
            'train',
            {
                'avg_loss': running_loss,
                **group_data
            },
            epoch=epoch
        )


def calculate_group_accuracies(outputs, y, groups, theshold=0.5):
    group_data = dict()
    worst_group_acc = float('inf')
    for group in np.unique(groups):
        group_outputs = outputs[groups == group]
        group_y = y[groups == group]
        group_pred = group_y >= theshold
        group_accuracy = accuracy_score(group_pred, group_outputs)
        group_accuracy = min(group_accuracy, worst_group_acc)
        group_data[f'group_{group}_accuracy'] = group_accuracy
    group_data['worst_group_accyracy'] = worst_group_acc
    return group_data


def evaluate(
    dataloader,
    model
):
    model.eval()
    y = torch.rand((0))
    outputs = torch.rand((0))
    groups = torch.rand((0))
    with torch.no_grad():
        for img, batch_y, g, _ in dataloader:
            img = img.cuda()
            batch_outputs = model(img).squeeze().cpu()
            outputs = torch.concat((outputs, batch_outputs))
            y = torch.concat((y, batch_y))
            groups = torch.concat((groups, g))
    outputs = outputs.cpu().numpy()
    y = y.cpu().int().numpy()
    groups = groups.cpu().numpy()
    group_data = calculate_group_accuracies(outputs, y, groups)
    return group_data


def test_model(test_dataloader, model):
    group_data = evaluate(test_dataloader, model)
    log_to_wandb('test', group_data)



if __name__ == '__main__':
    # for testing only
    train_dataloader, val_dataloader, test_dataloader = load_data('/kaggle/working/waterbird_complete95_forest2water2', batch_size=1)
    print(f'len train:\t{len(train_dataloader)}')
    print(f'len validation:\t{len(val_dataloader)}')
    print(f'len test:\t{len(test_dataloader)}')
    for img, y, g, p in train_dataloader:
        print(img.shape, y.shape, g.shape, p.shape, sep='\t')
        print(y, g, p, sep='\t')
        exit()