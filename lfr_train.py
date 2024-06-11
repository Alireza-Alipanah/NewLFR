import argparse
import os

import torch
import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from lfr_data import EmbeddingsDataset, get_loader
from utils import (
    get_model,
    set_seed,
    get_criterion,
    log_to_wandb,
    calculate_group_accuracies,
    get_train_group_proportions,
    load_model_from_state_dict
)


if not torch.cuda.is_available():
    raise Exception('cuda not available')


def get_args():
    parser = argparse.ArgumentParser('training lfr')
    parser.add_argument('--embeddings_dir', help='base directory of embeddings', type=str)
    parser.add_argument('--embeddings_dim', default=2048, help='dimension of embeddings', type=int)
    parser.add_argument('--n_classes', default=2, help='number of classes for pretraining', type=int)
    parser.add_argument('--use_softmax', default=False, help='wether to softmax was used for pretraining', type=bool)
    parser.add_argument('--seed', default=1, help='seed', type=int)
    parser.add_argument('--number_to_pick', help='number of low and high loss data to pick from each', type=int)
    parser.add_argument('--C', help='C regularization', type=float)
    parser.add_argument('--model_path', help='path to the saved model', type=str)
    parser.add_argument('--model', default='ResNet50', help='model to train', type=str)
    args = parser.parse_args()
    return args


def get_all_data(dataset):
    # TODO better way of doing this?
    embedding, y, g, p = dataset[:]
    embedding = embedding.numpy()
    y = y.numpy()
    g = g.numpy()
    p = p.numpy()
    return embedding, y, g, p


def get_datasets(args):
    train = EmbeddingsDataset(
        args.embeddings_dir,
        args.embeddings_dim,
        split='train'
    )
    val1 = EmbeddingsDataset(
        args.embeddings_dir,
        args.embeddings_dim,
        split='valid',
        split_num=0,
        random_state=args.seed
    )
    val2 = EmbeddingsDataset(
        args.embeddings_dir,
        args.embeddings_dim,
        split='val',
        split_num=1,
        random_state=args.seed
    )
    test = EmbeddingsDataset(
        args.embeddings_dir,
        args.embeddings_dim,
        split='test'
    )
    return train, val1, val2, test


def get_training_data(dataset, model, number_to_pick):
    cirterion = get_criterion(args)
    embeddings, y, g, p = get_all_data(dataset)
    if number_to_pick * 2 > len(embeddings):
        raise Exception('Cannot pick more data points that there exists.')
    model.to(device)
    model.eval()
    losses = []
    with torch.no_grad():
        for idx in len(range(embeddings)):
            embedding = torch.from_numpy(embeddings[i]).to(device)
            output = model(e)
            loss = cirterion(output.squeeze().cpu(), torch.from_numpy(y[i]).cpu())
            losses.append(loss.item())
    idx = len(range(losses))
    sorted_losses = sorted(idx, key=lambda i: losses[i], reverse=True)
    chosen_idx = sorted_losses[:number_to_pick] + sorted_losses[-number_to_pick:]
    return embeddings[chosen_idx], y[chosen_idx], g[chosen_idx], p[chosen_idx]


def test_model(pred, y, groups, prefix, group_proportions):
    group_data = calculate_group_accuracies(pred, y, groups, group_proportions)
    log_to_wandb(prefix, group_data)
    print(
        f'{prefix} metrics:\n\t' + \
        '\n\t'.join([f'{key}: {value}' for key, value in group_data.items()])
    )


if __name__ == '__main__':
    args = get_args()
    args.weights = 'None'
    set_seed(args)
    model = get_model(args, args.n_classes)
    load_model_from_state_dict(args.model_path, model=model, remove_fc=False)
    train, val1, val2, test = get_datasets(args)
    train_embeddings, train_y, train_groups, _ = get_training_data(val1, model, args.number_to_pick)
    val_embeddings, val_y, val_groups, _ = get_all_data(val2)
    test_embeddings, test_y, test_groups, _ = get_all_data(test)
    scaler = StandardScaler()
    train_embeddings = scaler.fit_transform(train_embeddings)
    val_embeddings = scaler.fit_transform(val_embeddings)
    test_embeddings = scaler.fit_transform(test_embeddings)
    group_proportions = get_train_group_proportions(train.group)
    LRModel = LogisticRegression(random_state=args.seed, C=args.C, penalty='l1', solver='saga')
    wandb.init(
        project='LFR on Waterbirds',
        config=vars(args),
    )
    test_model()
    print('training...')
    LRModel.fit(train_embeddings, train_y)
    print('testing...')
    train_pred = LRModel.predict(train_embeddings)
    test_model(train_pred, train_y, train_groups, 'train', group_proportions)
    val_pred = LRModel.predict(train_embeddings)
    test_model(val_pred, val_y, val_groups, 'validation', group_proportions)
    test_pred = LRModel.predict(train_embeddings)
    test_model(test_pred, test_y, test_groups, 'test', group_proportions)
    wandb.finish()
