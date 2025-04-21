from wb_data import WaterBirdsDataset, get_transform_cub, get_loader, log_data, build_dataloader_with_correlation
from data import CifarMnistConvexCombinationDataset, DSprite, building_dsprite_dataloader

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
import wandb
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
import torchvision
from torchvision.models import resnet50, ResNet50_Weights, resnet18, ResNet18_Weights
from torch.optim import SGD
import torch.nn.functional as F
import torch.nn as nn
import pandas as pd

import random


def set_seed(args):
    if args.seed is None:
        return
    seed = args.seed
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_transforms(target_resolution, augment_train, validation_for_train):
    train_transform = get_transform_cub(target_resolution=target_resolution, train=True, augment_data=augment_train)
    val_transform = get_transform_cub(target_resolution=target_resolution, train=validation_for_train, augment_data=validation_for_train)
    test_transform = get_transform_cub(target_resolution=target_resolution, train=False, augment_data=False)
    return train_transform, val_transform, test_transform


def get_loaders(
    train_dataset,
    val_dataset,
    test_dataset,
    shuffle_train,
    batch_size,
    spurious_correlation_value=None
):
    if spurious_correlation_value is None:
        train_dataloader = get_loader(
            data=train_dataset,
            train=shuffle_train,
            reweight_groups=None,
            reweight_classes=None,
            reweight_places=None,
            batch_size=batch_size,
            )
    else:
        train_dataloader = build_dataloader_with_correlation(
            train_dataset,
            spurious_correlation_value, 
            batch_size,
            shuffle=shuffle_train
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


def load_data(
    basedir,
    batch_size,
    target_resolution=(224, 224),
    validation_for_train=False,
    augment_train=True,
    shuffle_train=True,
    alternative_format=False,
    spurious_correlation_value=None,
):

    train_transform, val_transform, test_transform = get_transforms(target_resolution, augment_train, validation_for_train)

    train_dataset = WaterBirdsDataset(
        basedir=basedir,
        split='train',
        transform=train_transform,
        alternative_format=alternative_format
        )
    val_dataset = WaterBirdsDataset(
        basedir=basedir,
        split='val',
        transform=val_transform,
        alternative_format=alternative_format
        )
    test_dataset = WaterBirdsDataset(
        basedir=basedir,
        split='test',
        transform=test_transform,
        alternative_format=alternative_format
        )
    return get_loaders(
        train_dataset,
        val_dataset,
        test_dataset,
        shuffle_train,
        batch_size,
        spurious_correlation_value=spurious_correlation_value
    )


def load_convex_combination_data(
    mapping_csv_path,
    alpha,
    correlation,
    batch_size,
    target_resolution=(224, 224),
    validation_for_train=False,
    augment_train=True,
    shuffle_train=True,
):
    mapping = pd.read_csv(mapping_csv_path)

    train_transform, val_transform, test_transform = get_transforms(target_resolution, augment_train, validation_for_train)
    train_dataset = CifarMnistConvexCombinationDataset(
        mapping=mapping,
        alpha=alpha,
        split='train',
        transform=train_transform
        )
    val_dataset = CifarMnistConvexCombinationDataset(
        mapping=mapping,
        alpha=alpha,
        split='val',
        transform=val_transform
        )
    test_dataset = CifarMnistConvexCombinationDataset(
        mapping=mapping,
        alpha=alpha,
        split='test',
        transform=test_transform
        )
    return get_loaders(train_dataset, val_dataset, test_dataset, shuffle_train, batch_size)


def load_dsprite_data(
    args,
    basedir,
    batch_size,
    shuffle_train=True,
    spurious_correlation_value=None,
):
    ds = DSprite(basedir, 0.5, seed=args.seed)
    gs = []
    ys = []
    imgs = []
    for i in range(len(ds)):
        imgs.append(ds[i][0])
        ys.append(ds[i][1])
        gs.append(ds[i][2])
    imgs, ys, gs = np.array(imgs), np.array(ys), np.array(gs)
    number_of_samples = len(ys)
    
    train_dataset_indexes, test_val_dataset_indexes = train_test_split(list(range(number_of_samples)), test_size=0.75, random_state=args.seed)
    val_dataset_indexes, test_dataset_indexes = train_test_split(test_val_dataset_indexes, test_size=1/3, random_state=args.seed)
    val_dataset_indexes, last_layer_dataset_indexes = train_test_split(val_dataset_indexes, test_size=0.5, random_state=args.seed)


    train_loader = building_dsprite_dataloader(
        imgs=imgs[train_dataset_indexes],
        ys=ys[train_dataset_indexes],
        gs=gs[train_dataset_indexes],
        correlation_num=spurious_correlation_value,
        portion_num=1, #TODO portion ratio
        batch_size=batch_size,
        shuffle=shuffle_train)
    val_loader = building_dsprite_dataloader(
        imgs=imgs[val_dataset_indexes],
        ys=ys[val_dataset_indexes],
        gs=gs[val_dataset_indexes],
        correlation_num=0.5,
        portion_num=1, #TODO portion ratio
        batch_size=batch_size,
        shuffle=False)
    test_loader = building_dsprite_dataloader(
        imgs=imgs[test_dataset_indexes],
        ys=ys[test_dataset_indexes],
        gs=gs[test_dataset_indexes],
        correlation_num=0.5,
        portion_num=1, #TODO portion ratio
        batch_size=batch_size,
        shuffle=False)
    return train_loader, val_loader, test_loader


def color_grayscale_arr(arr, red=True):
  """Converts grayscale image to either red or green"""
  assert arr.ndim == 2
  dtype = arr.dtype
  h, w = arr.shape
  arr = np.reshape(arr, [1, h, w])
  if red:
    arr = np.concatenate([arr,
                          np.zeros((2, h, w), dtype=dtype)], axis=0)
  else:
    arr = np.concatenate([np.zeros((1, h, w), dtype=dtype),
                          arr,
                          np.zeros((1, h, w), dtype=dtype)], axis=0)
  return arr


def keep_only_lbls(dataset, lbls):
  lbls = {lbl: i for i, lbl in enumerate(lbls)}
  final_X, final_Y = [], []
  for x, y in dataset:
    if y in lbls:
      final_X.append(x)
      final_Y.append(lbls[y])
  X = torch.stack(final_X)
  Y = torch.tensor(final_Y).float().view(-1,1)
  return X, Y


def load_dominos_cmnist_all_exps_data(
        args,
        batch_size
):
    c1_m_red = c2_fm_green = np.ceil(args.dataset_total_count * args.color_correlation * args.shape_correlation * 0.5)
    c1_m_green = c2_fm_red = int(args.dataset_total_count * args.shape_correlation * (1 - args.color_correlation ) * 0.5)
    c1_fm_red = c2_m_green = np.ceil(args.dataset_total_count * args.color_correlation  * (1 - args.shape_correlation) * 0.5)
    c1_fm_green = c2_m_red = int(args.dataset_total_count * (1 - args.color_correlation ) * (1 - args.shape_correlation) * 0.5)
    print(c1_m_red, c1_m_green, c1_fm_red, c1_fm_green, c2_m_red, c2_m_green, c2_fm_red, c2_fm_green, sep='\n')
    mnist_transform = torchvision.transforms.Compose([
          torchvision.transforms.ToTensor(), 
          torchvision.transforms.Normalize(mean=0.485,
                                           std=0.229),
        ])
    mnist_train = torchvision.datasets.MNIST('./data/mnist/', train=True, download=True, transform=mnist_transform)
    X_mnist, _ = keep_only_lbls(mnist_train, lbls=[0, 1, 2, 3])
    fmnist_train = torchvision.datasets.FashionMNIST('./data/fashion_mnist/', train=True, download=True, transform=mnist_transform)
    X_fmnist, _ = keep_only_lbls(fmnist_train, lbls=[0, 3, 4, 6])
    red_mnist, red_fmnist, green_mnist, green_fmnist = [], [], [], []
    for idx, img in enumerate(X_mnist):
        img = color_grayscale_arr(img.squeeze().numpy(), red=True)
        img = np.pad(img, ((0, 0), (2, 2), (2, 2)), constant_values=0)
        red_mnist.append(torch.tensor(img))
    red_mnist = torch.stack(red_mnist)
    for idx, img in enumerate(X_mnist):
        img = color_grayscale_arr(img.squeeze().numpy(), red=False)
        img = np.pad(img, ((0, 0), (2, 2), (2, 2)), constant_values=0)
        green_mnist.append(torch.tensor(img))
    green_mnist = torch.stack(green_mnist)

    for idx, img in enumerate(X_fmnist):
        img = color_grayscale_arr(img.squeeze().numpy(), red=True)
        img = np.pad(img, ((0, 0), (2, 2), (2, 2)), constant_values=0)
        red_fmnist.append(torch.tensor(img))
    red_fmnist = torch.stack(red_fmnist)
    for idx, img in enumerate(X_fmnist):
        img = color_grayscale_arr(img.squeeze().numpy(), red=False)
        img = np.pad(img, ((0, 0), (2, 2), (2, 2)), constant_values=0)
        green_fmnist.append(torch.tensor(img))
    green_fmnist = torch.stack(green_fmnist)
    idx = torch.randperm(red_mnist.shape[0])
    red_mnist_0_idx = idx[:c1_m_red]
    red_mnist_4_idx = idx[c1_m_red:c1_m_red+c2_m_red]
    red_mnist_0 = red_mnist[red_mnist_0_idx]
    red_mnist_4 = red_mnist[red_mnist_4_idx]
    idx = torch.randperm(green_mnist.shape[0])
    green_mnist_1_idx = idx[:c1_m_green]
    green_mnist_5_idx = idx[c1_m_green:c1_m_green+c2_m_green]
    green_mnist_1 = green_mnist[green_mnist_1_idx]
    green_mnist_5 = green_mnist[green_mnist_5_idx]
    idx = torch.randperm(red_fmnist.shape[0])
    red_fmnist_2_idx = idx[:c1_fm_red]
    red_fmnist_6_idx = idx[c1_fm_red:c1_fm_red+c2_fm_red]
    red_fmnist_2 = red_fmnist[red_fmnist_2_idx]
    red_fmnist_6 = red_fmnist[red_fmnist_6_idx]
    idx = torch.randperm(green_fmnist.shape[0])
    green_fmnist_3_idx = idx[:c1_fm_green]
    green_fmnist_7_idx = idx[c1_fm_green:c1_fm_green+c2_fm_green]
    green_fmnist_3 = green_fmnist[green_fmnist_3_idx]
    green_fmnist_7 = green_fmnist[green_fmnist_7_idx]
    cifar_transform = torchvision.transforms.Compose([
          torchvision.transforms.ToTensor(), 
          torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                           std=[0.229, 0.224, 0.225]),
        ])
    cifar_train = torchvision.datasets.CIFAR10('./data/cifar10/', train=True, download=True, transform=cifar_transform)
    X_c_cars, _ = keep_only_lbls(cifar_train, lbls=[0, 1, 8, 9])
    X_c_animals, _ = keep_only_lbls(cifar_train, lbls=[3, 4, 5, 7])
    X_c_cars_0 = X_c_cars[:c1_m_red]
    X_c_cars_1 = X_c_cars[c1_m_red:c1_m_red+c1_m_green]
    X_c_cars_2 = X_c_cars[c1_m_red+c1_m_green:c1_m_red+c1_m_green+c1_fm_red]
    X_c_cars_3 = X_c_cars[c1_m_red+c1_m_green+c1_fm_red:]
    X_c_animals_7 = X_c_animals[:c2_fm_green]
    X_c_animals_6 = X_c_animals[c2_fm_green:c2_fm_green+c2_fm_red]
    X_c_animals_5 = X_c_animals[c2_fm_green+c2_fm_red:c2_fm_green+c2_fm_red+c2_m_green]
    X_c_animals_4 = X_c_animals[c2_fm_green+c2_fm_red+c2_m_green:]
    group_0_X = torch.cat((X_c_cars_0, red_mnist_0), dim=2)
    group_0_Y = torch.zeros(len(group_0_X))
    group_0_G = torch.tensor([0] * len(group_0_X))
    group_1_X = torch.cat((X_c_cars_1, green_mnist_1), dim=2)
    group_1_Y = torch.zeros(len(group_1_X))
    group_1_G = torch.tensor([1] * len(group_1_X))
    group_2_X = torch.cat((X_c_cars_2, red_fmnist_2), dim=2)
    group_2_Y = torch.zeros(len(group_2_X))
    group_2_G = torch.tensor([2] * len(group_2_X))
    group_3_X = torch.cat((X_c_cars_3, green_fmnist_3), dim=2)
    group_3_Y = torch.zeros(len(group_3_X))
    group_3_G = torch.tensor([3] * len(group_3_X))
    group_4_X = torch.cat((X_c_animals_4, red_mnist_4), dim=2)
    group_4_Y = torch.ones(len(group_4_X))
    group_4_G = torch.tensor([4] * len(group_4_X))
    group_5_X = torch.cat((X_c_animals_5, green_mnist_5), dim=2)
    group_5_Y = torch.ones(len(group_5_X))
    group_5_G = torch.tensor([5] * len(group_5_X))
    group_6_X = torch.cat((X_c_animals_6, red_fmnist_6), dim=2)
    group_6_Y = torch.ones(len(group_6_X))
    group_6_G = torch.tensor([6] * len(group_6_X))
    group_7_X = torch.cat((X_c_animals_7, green_fmnist_7), dim=2)
    group_7_Y = torch.ones(len(group_7_X))
    group_7_G = torch.tensor([7] * len(group_7_X))
    X = torch.cat((group_0_X, group_1_X, group_2_X, group_3_X, group_4_X, group_5_X, group_6_X, group_7_X))
    Y = torch.cat((group_0_Y, group_1_Y, group_2_Y, group_3_Y, group_4_Y, group_5_Y, group_6_Y, group_7_Y))
    G = torch.cat((group_0_G, group_1_G, group_2_G, group_3_G, group_4_G, group_5_G, group_6_G, group_7_G))
    dataset = TensorDataset(X, Y.long(), G.long())
    train_dataset, lastlayer_dataset, val_dataset, test_dataset = random_split(dataset, [0.25, 0.25, 0.25, 0.25], generator=torch.Generator().manual_seed(args.seed))
    print(len(train_dataset), len(lastlayer_dataset), len(val_dataset), len(test_dataset))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    lastlayer_loader = DataLoader(lastlayer_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


def log_to_wandb(data_dict, epoch=None):
    if epoch is not None:
        wandb.log({
            'epoch': epoch,
            **data_dict
        })
    else:
        wandb.log(data_dict)

def convert_to_wandb_format(prefix, data_dict):
    new_data = {f'{prefix}/{key}': value for key, value in data_dict.items()}
    return new_data


def get_train_group_proportions(array):
    flattened_arr = array.flatten().astype('int')
    value_counts = np.bincount(flattened_arr)
    total_count = np.sum(value_counts)
    proportions = value_counts / total_count
    unique_values = np.nonzero(value_counts)[0]
    proportions_dict = dict(zip(unique_values, proportions[unique_values]))
    return proportions_dict


def calculate_our_mean_group_accuracy(pred, y, groups, proportions_dict): # TODO: fix: this only works when there are four groups
    maj_groups = sorted(list(proportions_dict.items()), key=lambda x: x[1])[-2:]
    min_groups = sorted(list(proportions_dict.items()), key=lambda x: x[1])[:2]
    maj_accuracy = np.mean([accuracy_score(pred[groups == group], y[groups == group]) for group in np.unique(list(zip(*maj_groups))[0])])
    min_accuracy = np.mean([accuracy_score(pred[groups == group], y[groups == group]) for group in np.unique(list(zip(*min_groups))[0])])
    maj_percent = sum(list(zip(*maj_groups))[1])
    min_percent = sum(list(zip(*min_groups))[1])
    return maj_percent * maj_accuracy + min_percent * min_accuracy


def calculate_group_accuracies(pred, y, groups, proportions_dict):
    group_data = dict()
    mean_acc = 0
    group_accuracies = []
    for group in np.unique(groups):
        group_pred = pred[groups == group]
        group_y = y[groups == group]
        group_accuracy = accuracy_score(group_y, group_pred)
        group_data[f'group_{int(group)}_accuracy'] = group_accuracy
        group_accuracies.append(group_accuracy)
        mean_acc += group_accuracy * proportions_dict[group] # calculated according to JTT paper
    group_data['worst_group_accuracy'] = min(group_data.values())
    group_data['accuracy'] = accuracy_score(y, pred)
    group_data['mean_accuracy'] = mean_acc
    group_data['our_mean_accuracy'] = calculate_our_mean_group_accuracy(pred, y, groups, proportions_dict)
    group_data['group_average_accuracy'] = np.mean(group_accuracies)
    return group_data


def calculate_env_accuracies(pred, y, envs):
    env_data = dict()
    for i, env in enumerate(envs):
        env_pred = pred[env]
        env_y = y[env]
        env_accuracy = accuracy_score(env_y, env_pred)
        env_data[f'env_{i}_accuracy'] = env_accuracy
    env_data['worst_env_accuracy'] = min(env_data.values())
    return env_data


# def test_model(test_dataloader, model, classification_func, group_proportions, logger, is_start=False):
#     group_data = evaluate(test_dataloader, model, classification_func, group_proportions)
#     prefix = 'test_start' if is_start else 'test_last_epoch'
#     test_data = convert_to_wandb_format(prefix, group_data)
#     log_to_wandb(test_data)
#     logger.add_test_data(prefix, group_data)
#     log_str = 'Test metrics:\n\t' + '\n\t'.join([f'{key}: {value}' for key, value in group_data.items()])
#     print(log_str)


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


def get_resnet18_weights(args):
    if args.weights == 'ResNet18_Weights.DEFAULT':
        weights = ResNet18_Weights.DEFAULT
    elif args.weights == 'None':
        weights = None
    else:
        raise NotImplementedError()
    return weights


def new_fc(args, n_classes, model):
    in_features = model.fc.in_features
    out_features = n_classes
    model.fc = nn.Linear(in_features=in_features, out_features=out_features)
    return model


def get_model(args, n_classes, modify_fc=True):
    if args.continue_from_chechpoint and args.model == 'ResNet50':
        model = resnet50(weights=None)
        model = new_fc(args, n_classes, model)
        model = load_model_from_state_dict(args.weights, model=model, remove_fc=False)
    elif args.model == 'ResNet50':
        weights = get_resnet50_weights(args)
        model = resnet50(weights=weights)
        if modify_fc:
            model = new_fc(args, n_classes, model)
    elif args.model == 'ResNet18':
        weights = get_resnet18_weights(args)
        model = resnet18(weights=weights)
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


def get_criterion(reduction=None):
    if reduction is None:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.CrossEntropyLoss(reduction=reduction)
    return criterion


def get_classification_func(args, sigmoid_threshold=0.5):
    return lambda x: torch.argmax(F.softmax(x, dim=1), dim=1)


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


def load_model_from_state_dict(model_path, model=None, remove_fc=True, device='cpu'):
    if model is None:
        model = resnet50(weights=None)
    if remove_fc:
        model.fc = nn.Identity()
    state_dict = torch.load(model_path, map_location=torch.device(device))
    print_weigth_match(state_dict, model)
    model.load_state_dict(state_dict, strict=False)
    return model


def load_model(path, remove_fc=False, device='cpu'):
    # loads a pytorch model that was saved as a whole
    model = torch.load(path, map_location=torch.device(device), weights_only=False)
    if remove_fc:
        model.fc = nn.Identity()
    return model
