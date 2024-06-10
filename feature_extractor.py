import argparse
import os

import torch
import torch.nn as nn
from torchvision.models import resnet50
import pandas as pd
import tqdm

from utils import load_data


if not torch.cuda.is_available():
    raise Exception('cuda not available')


def get_args():
    parser = argparse.ArgumentParser('Pretraining Resnet50')
    parser.add_argument('--base_dir', default='/kaggle/working/waterbird_complete95_forest2water2', help='base directory of images', type=str)
    parser.add_argument('--batch_size', default=128, help='data loader batch size', type=int)
    parser.add_argument('--model', default='ResNet50', help='model to train', type=str)
    parser.add_argument('--weights', default='ResNet50_Weights.IMAGENET1K_V1', help='specify the weights to use for the pretrained model', type=str)
    parser.add_argument('--model_path', help='path to the saved model', type=str)
    parser.add_argument('--save_path', help='embeddings save path', type=str)
    parser.add_argument('--save_name', help='name of the embeddings file', type=str)
    parser.add_argument('--embedding_size', default=2048, help='size of the embeddings', type=int)
    args = parser.parse_args()
    return args


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


def load_model(args):
    model = resnet50(weights=None)
    state_dict = torch.load(args.model_path)
    print_weigth_match(state_dict, model)
    model.load_state_dict(state_dict, strict=False)
    model.fc = nn.Identity()
    return model


def get_dataloaders(args):
    train_dataloader, val_dataloader, test_dataloader = load_data(
        basedir=args.base_dir,
        batch_size=args.batch_size,
        augment_train=False,
        shuffle_train=False
    )
    return train_dataloader, val_dataloader, test_dataloader


def load_embeddings(args, model, loader):
    model.eval()
    embeddings = torch.rand((0, args.embedding_size))
    y = torch.rand((0))
    g = torch.rand((0))
    p = torch.rand((0))
    with torch.no_grad():
        for idx, (img, batch_y, batch_g, batch_p) in tqdm(enumerate(loader)):
            img = img.cuda()
            btach_embeddings = model(img).cpu()
            embeddings = torch.concat((embeddings, btach_embeddings))
            y = torch.concat((y, batch_y))
            g = torch.concat((g, batch_g))
            p = torch.concat((p, batch_p))
    return embeddings, y, g, p


def extract_embeddings(args, model, train_loader, val_loader, test_loader):
    model.eval()
    embeddings = torch.rand((0, args.embedding_size))
    y = torch.rand((0))
    g = torch.rand((0))
    p = torch.rand((0))
    split = torch.rand((0))
    for split_i, loader in enumerate([train_loader, val_loader, test_loader]):
       split_embedings, split_y, split_g, split_p = load_embeddings(args, model, loader)
       embeddings = torch.concat((embeddings, split_embedings))
       y = torch.concat((y, split_y))
       g = torch.concat((g, split_g))
       p = torch.concat((p, split_p))
       split = torch.concat((split, torch.tensor([split_i] * len(split_y))))
    return embeddings, y, g, p, split


def create_dataframe(embeddings, y, g, p, split):
    df = pd.DataFrame(embeddings)
    df['label'] = y
    df['group'] = g
    df['place_label'] = p
    df['split'] = split
    return df


def save_dataframe(args, df):
    df.to_csv(os.path.join(args.save_path, args.save_name))


if __name__ == '__main__':
    args = get_args()
    model = load_model(args)
    model.cuda()
    train_dataloader, val_dataloader, test_dataloader = get_dataloaders(args)
    embeddings, y, g, p, split = extract_embeddings(args, model, train_loader, val_loader, test_loader)
    df = create_dataframe(embeddings, y, g, p, split)
    save_dataframe(args, df)
