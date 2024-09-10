import argparse
import os
import itertools

import torch
import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from spuco.group_inference import EIIL
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
    load_model,
    get_classification_func,
    calculate_env_accuracies
)


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


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
    parser.add_argument('--model_complete', default=False, help='wether the model is saved alognside the state dict', type=bool)
    parser.add_argument('--model', default='ResNet50', help='model to train', type=str)
    parser.add_argument('--use_standard_scaler', default=False, help='wether to use standard scaler based on training data', type=bool)
    parser.add_argument('--method', default='NewLFR', help='method to use for training', type=str)
    parser.add_argument('--split_random_state', default=None, help='random state value used for splitting validation in half', type=int)
    parser.add_argument('--subsample', default=False, help='whether to subsample to have balanced groups', type=bool)
    parser.add_argument('--project_name', default='LFR on Waterbirds', help='wandb project name', type=str)

    # EIIL
    parser.add_argument('--eiil', default=False, help='wether to use eiil', type=bool)
    parser.add_argument('--eiil_lr', default=0.01, help='lerning rate used by eiil', type=float)
    parser.add_argument('--eiil_steps', default=20000, help='number of steps used by eiil', type=int)
    parser.add_argument('--eiil_batch_size', default=128, help='batch size used by eiil', type=int)

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
    logits = []
    with torch.no_grad():
        for idx in range(len(embeddings)):
            embedding = torch.from_numpy(embeddings[idx]).float()
            logit = model.fc(embedding)
            loss = cirterion(logit, y_torch[idx])
            losses.append(loss.item())
            pred = classiffication_fun(logit.unsqueeze(0)).numpy()
            preds.append(pred)
            logits.append(logit)
    return losses, preds, logits


def get_newLFR_data(args, dataset, model):
    number_to_pick = args.number_to_pick
    embeddings, y, g, p = get_all_data(dataset)
    if number_to_pick * 2 > len(embeddings):
        raise Exception('Cannot pick more data points that there exists.')
    losses, _, _ = calaculate_loss_and_prediction(args, embeddings, y, model)
    idx = range(len(losses))
    sorted_losses_idx = sorted(idx, key=lambda i: losses[i])
    chosen_idx = get_at_most_n_low_and_high_samples_without_replacement(number_to_pick, sorted_losses_idx)
    return embeddings[chosen_idx], y[chosen_idx], g[chosen_idx], p[chosen_idx]


def get_evals_data(args, dataset, model):
    number_to_pick = args.number_to_pick
    embeddings, y, g, p = get_all_data(dataset)
    if number_to_pick * 2 > len(embeddings):
        raise Exception('Cannot pick more data points that there exists.')
    losses, _, _ = calaculate_loss_and_prediction(args, embeddings, y, model)
    idx = list(zip(range(len(losses)), y))
    sorted_losses_idx = sorted(idx, key=lambda i: losses[i[0]])
    chosen_idx = []
    for label in np.unique(y):
        labels_idx = list(filter(lambda x: x[1] == label, sorted_losses_idx))
        labels_idx = list(zip(*labels_idx))[0]
        chosen_idx.extend(get_at_most_n_low_and_high_samples_without_replacement(number_to_pick, labels_idx))
    return embeddings[chosen_idx], y[chosen_idx], g[chosen_idx], p[chosen_idx]


def get_at_most_n_low_and_high_samples_without_replacement(n, ls):
    lowest = ls[:n]
    highest = ls[n:][-n:]
    return highest + lowest


# TODO Fix this
def get_LFR_data(args, dataset, model):
    number_to_pick = args.number_to_pick
    embeddings, y, g, p = get_all_data(dataset)
    if number_to_pick * 4 > len(embeddings):
        raise Exception('Cannot pick more data points that there exists.')
    losses, preds, _ = calaculate_loss_and_prediction(args, embeddings, y, model)
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
    elif args.method == 'EVaLS':
        return get_evals_data(args, dataset, model)
    else:
        raise NotImplementedError()


def test_model(pred, y, groups, prefix, group_proportions, envs=None):
    group_data = calculate_group_accuracies(pred, y, groups, group_proportions)
    env_data = calculate_env_accuracies(pred, y, envs) if envs is not None else {}
    data = {**group_data, **env_data}
    log_to_wandb(prefix, data)
    print(
        f'{prefix} metrics:\n\t' + \
        '\n\t'.join([f'{key}: {value}' for key, value in data.items()])
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
        
    LRModel = LogisticRegression(random_state=args.seed, C=args.C, penalty='l1', solver='liblinear')

    print(f'training {stage}...')
    LRModel.fit(train_embeddings, train_y)

    print(f'testing {stage}...')
    train_pred = LRModel.predict(train_embeddings)
    test_model(train_pred, train_y, train_groups, f'train_{stage}', group_proportions)
    val_pred = LRModel.predict(val_embeddings)
    val_envs = get_environments(args, val_embeddings, val_y, model)
    test_model(val_pred, val_y, val_groups, f'validation_{stage}', group_proportions, val_envs)
    test_pred = LRModel.predict(test_embeddings)
    test_model(test_pred, test_y, test_groups, f'test_{stage}', group_proportions)


def split_dataset(args, dataset, test_size=0.5):
    val1_idx, val2_idx = train_test_split(range(len(dataset)), test_size=test_size, random_state=args.split_random_state)
    val1 = torch.utils.data.Subset(dataset, val1_idx)
    val2 = torch.utils.data.Subset(dataset, val2_idx)
    return val1, val2


def infer_groups_using_eiil(logits, lables, lr, stpes):
    eiil = EIIL(
        logits=logits,
        class_labels=lables,
        lr=lr,
        num_steps=stpes,
        device=DEVICE,
    )
    inferred_groups = eiil.infer_groups()
    return inferred_groups


def get_environments(args, embeddings, y, model):
    if not args.eiil:
        return None
    _, _, logits = calaculate_loss_and_prediction(args, embeddings, y, model)
    logits = torch.concat(list(map(lambda x: x.unsqueeze(0), logits)))
    class_labels = y
    inferred_groups = infer_groups_using_eiil(logits, class_labels, args.eiil_lr, args.eiil_steps)
    groups = []
    unqiue_labels = np.unique(y)
    for inferred_group in inferred_groups.values():
        for label in unqiue_labels:
            group = list(filter(lambda x: y[x] == label, inferred_group))
            groups.append(group)
    return groups


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


if __name__ == '__main__':
    args = get_args()
    args.weights = 'None'
    set_seed(args)
    # model creation
    if args.method != 'DFR' or args.eiil:
        if args.model_complete:
            model = load_model(args.model_path, remove_fc=False)
        else:
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
