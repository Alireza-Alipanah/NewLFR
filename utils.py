from wb_data import WaterBirdsDataset, get_transform_cub, get_loader, log_data

from sklearn.metrics import accuracy_score
import numpy as np
from tqdm import tqdm
import wandb
import torch
from torchvision.models import resnet50, ResNet50_Weights
from torch.optim import SGD
import torch.nn.functional as F
import torch.nn as nn


def set_seed(args):
    if args.seed is None:
        return
    seed = args.seed
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_data(
    basedir,
    batch_size,
    target_resolution=(224, 224),
    validation_for_train=False,
    augment_train=True,
    shuffle_train=True
):
    train_transform = get_transform_cub(target_resolution=target_resolution, train=True, augment_data=augment_train)
    val_transform = get_transform_cub(target_resolution=target_resolution, train=validation_for_train, augment_data=True) # TODO augment or not ?
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
        train=shuffle_train,
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
    criterion,
    classification_func,
    group_proportions
):
    for epoch in tqdm(range(args.epochs)):
        model.train()
        running_loss = 0
        for img, y, _, _ in train_dataloader:
            y = y.long() if args.use_softmax else y.float()
            img, y = img.cuda(), y.cuda()
            optimizer.zero_grad()
            outputs = model(img).squeeze()
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            wandb.log({'train/step_loss': loss.item()})
        running_loss /= len(train_dataloader)
        group_data = evaluate(train_dataloader, model, classification_func, group_proportions)
        log_to_wandb(
            'train',
            {
                'avg_loss': running_loss,
                **group_data
            },
            epoch=epoch
        )


def get_train_group_proportions(array):
    flattened_arr = array.flatten()
    value_counts = np.bincount(flattened_arr)
    total_count = np.sum(value_counts)
    proportions = value_counts / total_count
    unique_values = np.nonzero(value_counts)[0]
    proportions_dict = dict(zip(unique_values, proportions[unique_values]))
    return proportions_dict


def calculate_group_accuracies(pred, y, groups, proportions_dict):
    group_data = dict()
    worst_group_acc = float('inf')
    mean_acc = 0
    for group in np.unique(groups):
        group_pred = pred[groups == group]
        group_y = y[groups == group]
        group_accuracy = accuracy_score(group_y, group_pred)
        worst_group_acc = min(group_accuracy, worst_group_acc)
        group_data[f'group_{int(group)}_accuracy'] = group_accuracy
        mean_acc += group_accuracy * proportions_dict[group] # calculated according to JTT paper
    group_data['worst_group_accuracy'] = worst_group_acc
    group_data['accuracy'] = accuracy_score(y, pred)
    group_data['mean_accuracy'] = mean_acc
    return group_data


def evaluate(
    dataloader,
    model,
    classification_func,
    group_proportions
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
    outputs = outputs
    pred = classification_func(outputs).numpy()
    y = y.cpu().int().numpy()
    groups = groups.cpu().numpy()
    group_data = calculate_group_accuracies(pred, y, groups, group_proportions)
    return group_data


def test_model(test_dataloader, model, classification_func, group_proportions, is_start=False):
    group_data = evaluate(test_dataloader, model, classification_func, group_proportions)
    prefix = 'test_start' if is_start else 'test'
    log_to_wandb(prefix, group_data)
    print(
        'Test metrics:\n\t' + \
        '\n\t'.join([f'{key}: {value}' for key, value in group_data.items()])
    )


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


def new_fc(args, n_classes, model):
    in_features = model.fc.in_features
    out_features = n_classes if args.use_softmax else 1
    model.fc = nn.Linear(in_features=in_features, out_features=out_features)
    return model


def get_model(args, n_classes, modify_fc=True):
    if n_classes > 2 and (not args.use_softmax):
        raise Exception('Cannot use sigomid when n_classes > 2')
    if args.model == 'ResNet50':
        weights = get_resnet50_weights(args)
        model = resnet50(weights=weights)
        if modify_fc:
            model = new_fc(args, n_classes, model)
    else:
        raise NotImplementedError()
    return model



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


def print_weigth_match(loaded_state_dict, model):
    loaded_keys = ['loaded'] + list(loaded_state_dict.keys())
    model_keys = ['model'] + list(model.state_dict().keys())
    
    max_a = max(len(x) for x in loaded_keys)
    max_b = max(len(y) for y in model_keys)
    
    if len(loaded_keys) > len(model_keys):
        new_b = model_keys + [' '] * (len(loaded_keys)-len(model_keys))
        for i in range(len(loaded_keys)):
            print(f"{loaded_keys[i]:{max_a}}|{new_b[i]:{max_b}}")
            if i == 0:
                print('-' * len(f"{loaded_keys[i]:{max_a}}|{new_b[i]:{max_b}}"))
    else:
        new_a = loaded_keys + [' '] * (len(model_keys)-len(loaded_keys))
        for j in range(len(model_keys)):
            print(f"{new_a[j]:{max_a}}|{model_keys[j]:{max_b}}")
            if j == 0:
                print('-' * len(f"{new_a[j]:{max_a}}|{model_keys[j]:{max_b}}"))


def load_model_from_state_dict(model_path, model=None, remove_fc=True):
    if model is None:
        model = resnet50(weights=None)
    if remove_fc:
        model.fc = nn.Identity()
    state_dict = torch.load(model_path)
    print_weigth_match(state_dict, model)
    model.load_state_dict(state_dict, strict=False)
    return model
