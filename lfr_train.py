import argparse
import os

import torch
import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import wandb

from lfr_data import EmbeddingsDataset, get_loader
from utils import (
    get_model,
    set_seed,
    get_criterion,
    log_to_wandb,
    calculate_group_accuracies,
    get_train_group_proportions,
    load_model_from_state_dict,
    get_classification_func
)


def get_args():
    parser = argparse.ArgumentParser('training lfr')
    parser.add_argument('--embeddings_dir', help='base directory of embeddings', type=str)
    parser.add_argument('--embeddings_dim', default=2048, help='dimension of embeddings', type=int)
    parser.add_argument('--n_classes', default=2, help='number of classes for pretraining', type=int)
    parser.add_argument('--use_softmax', default=False, help='wether softmax was used for pretraining', type=bool)
    parser.add_argument('--seed', default=None, help='seed', type=int)
    parser.add_argument('--number_to_pick', help='number of low and high loss data to pick from each', type=int)
    parser.add_argument('--C', help='C regularization', type=float)
    parser.add_argument('--model_path', help='path to the saved model', type=str)
    parser.add_argument('--model', default='ResNet50', help='model to train', type=str)
    parser.add_argument('--use_standard_scaler', default=False, help='wether to use standard scaler based on training data', type=bool)
    parser.add_argument('--method', default='NewLFR', help='method to use for training', type=str)
    parser.add_argument('--split_random_state', default=None, help='random state value used for splitting validation in half', type=int)
    parser.add_argument('--subsample', default=False, help='whether to subsample to have balanced groups', type=bool)
    parser.add_argument('--project_name', default='LFR on Waterbirds', help='wandb project name', type=str)
    args = parser.parse_args()
    return args


def get_all_data(dataset):
    # TODO better way of doing this?
    embedding, y, g, p = dataset[:]
    return embedding, y, g, p


def get_datasets(args):
    train = EmbeddingsDataset(
        args.embeddings_dir,
        args.embeddings_dim,
        split='train'
    )
    val = EmbeddingsDataset(
        args.embeddings_dir,
        args.embeddings_dim,
        split='val'
    )
    test = EmbeddingsDataset(
        args.embeddings_dir,
        args.embeddings_dim,
        split='test'
    )
    return train, val, test


def calaculate_loss_and_prediction(args, embeddings, y, model):
    cirterion = get_criterion(args)
    classiffication_fun = get_classification_func(args)
    y_torch = torch.from_numpy(y)
    y_torch = y_torch.long() if args.use_softmax else y_torch.float()
    model.eval()
    losses = []
    preds = []
    with torch.no_grad():
        for idx in range(len(embeddings)):
            embedding = torch.from_numpy(embeddings[idx]).float()
            output = model.fc(embedding)
            loss = cirterion(output, y_torch[idx])
            losses.append(loss.item())
            pred = classiffication_fun(output.unsqueeze(0)).numpy()
            preds.append(pred)
    return losses, preds


def get_newLFR_data(args, dataset, model):
    number_to_pick = args.number_to_pick
    embeddings, y, g, p = get_all_data(dataset)
    if number_to_pick * 2 > len(embeddings):
        raise Exception('Cannot pick more data points that there exists.')
    losses, _ = calaculate_loss_and_prediction(args, embeddings, y, model)
    idx = range(len(losses))
    sorted_losses = sorted(idx, key=lambda i: losses[i])
    chosen_idx = get_at_most_n_low_and_high_samples_without_replacement(number_to_pick, sorted_losses)
    return embeddings[chosen_idx], y[chosen_idx], g[chosen_idx], p[chosen_idx]


def get_at_most_n_low_and_high_samples_without_replacement(n, ls):
    lowest = ls[:n]
    highest = ls[n:][-n:]
    return highest + lowest


def get_LFR_data(args, dataset, model):
    number_to_pick = args.number_to_pick
    embeddings, y, g, p = get_all_data(dataset)
    if number_to_pick * 4 > len(embeddings):
        raise Exception('Cannot pick more data points that there exists.')
    losses, preds = calaculate_loss_and_prediction(args, embeddings, y, model)
    idx = range(len(losses))
    correctly_classified = list(filter(lambda i: preds[i] == y[i], idx))
    misclassified = list(filter(lambda i: preds[i] != y[i], idx))
    C = sorted(correctly_classified, key=lambda i: losses[i])
    M = sorted(misclassified, key=lambda i: losses[i], reverse=True)
    chosen_correctly_classified = get_at_most_n_low_and_high_samples_without_replacement(number_to_pick, correctly_classified)
    chosen_misclassified = get_at_most_n_low_and_high_samples_without_replacement(number_to_pick, misclassified)
    chosen_idx = chosen_correctly_classified + chosen_misclassified
    return embeddings[chosen_idx], y[chosen_idx], g[chosen_idx], p[chosen_idx]


def subsample_data(dataset):
    embeddings, y, groups, place = get_all_data(dataset)
    balanced_groups_idx = []
    unique_groups, group_sizes = np.unique(groups, return_counts=True)
    min_group_size = group_sizes.min()
    for group in unique_groups:
        group_idx = np.where(groups == group)[0]
        sampled_group = resample(group_idx, replace=False, n_samples=min_group_size)
        balanced_groups_idx.append(sampled_group)
    idx = np.concatenate(balanced_groups_idx)
    return embeddings[idx], y[idx], groups[idx], place[idx]


def get_training_data(args, dataset, model):
    if args.method == 'DFR':
        if args.subsample:
            train_embeddings, train_y, train_groups, train_place = \
                subsample_data(dataset)
            return train_embeddings, train_y, train_groups, train_place
        else:
            return get_all_data(dataset)
    elif args.method == 'LFR':
        return get_LFR_data(args, dataset, model)
    elif args.method == 'NewLFR':
        return get_newLFR_data(args, dataset, model)
    else:
        raise NotImplementedError()


def test_model(pred, y, groups, prefix, group_proportions):
    group_data = calculate_group_accuracies(pred, y, groups, group_proportions)
    log_to_wandb(prefix, group_data)
    print(
        f'{prefix} metrics:\n\t' + \
        '\n\t'.join([f'{key}: {value}' for key, value in group_data.items()])
    )


def run_stage(
    args,
    train,
    val,
    test,
    model,
    stage,
    group_proportions,
):
    train_embeddings, train_y, train_groups, train_place = get_training_data(args, train, model)
    val_embeddings, val_y, val_groups, _ = get_all_data(val)
    test_embeddings, test_y, test_groups, _ = get_all_data(test)
            
    if args.use_standard_scaler:
        scaler = StandardScaler()
        train_embeddings = scaler.fit_transform(train_embeddings)
        val_embeddings = scaler.transform(val_embeddings)
        test_embeddings = scaler.transform(test_embeddings)
        
    LRModel = LogisticRegression(random_state=args.seed, C=args.C, penalty='l1', solver='liblinear', tol=1e-20, max_iter=int(1e4))

    print(f'training {stage}...')
    LRModel.fit(train_embeddings, train_y)

    print(f'testing {stage}...')
    train_pred = LRModel.predict(train_embeddings)
    test_model(train_pred, train_y, train_groups, f'train_{stage}', group_proportions)
    val_pred = LRModel.predict(val_embeddings)
    test_model(val_pred, val_y, val_groups, f'validation_{stage}', group_proportions)
    test_pred = LRModel.predict(test_embeddings)
    test_model(test_pred, test_y, test_groups, f'test_{stage}', group_proportions)


def split_dataset(args, dataset, test_size=0.5):
    val1_idx, val2_idx = train_test_split(range(len(dataset)), test_size=test_size, random_state=args.split_random_state)
    val1 = torch.utils.data.Subset(dataset, val1_idx)
    val2 = torch.utils.data.Subset(dataset, val2_idx)
    return val1, val2


def run_method(args, train, val, test, model):
    group_proportions = get_train_group_proportions(train.group)
    val1, val2 = split_dataset(args, val)
    run_stage( # for model selection
        args=args,
        train=val1,
        val=val2,
        test=test,
        model=model,
        stage='stage1',
        group_proportions=group_proportions,
    )
    if args.method == 'DFR':
        run_stage( # for retraining on all available data
            args=args,
            train=val,
            val=val,
            test=test,
            model=model,
            stage='stage2',
            group_proportions=group_proportions,
        )


if __name__ == '__main__':
    args = get_args()
    args.weights = 'None'
    set_seed(args)
    # model creation
    if args.method != 'DFR':
        model = get_model(args, args.n_classes)
        load_model_from_state_dict(args.model_path, model=model, remove_fc=False)
    else:
        model = None
    # loading data
    train, val, test = get_datasets(args)
    wandb.init(
        project=args.project_name,
        config=vars(args),
    )
    run_method(args, train, val, test, model)
    wandb.finish()
