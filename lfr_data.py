from sklearn.model_selection import train_test_split
from torch.Data import Dataset
import pandas as pd


class EmbeddingsDataset(Dataset):
    def __init__(self, df_path, embeddings_dim, split="train", split_num=-1, random_state=None):
        try:
            split_i = ["train", "val", "test"].index(split)
        except ValueError:
            raise(f"Unknown split {split}")
        if idx is not None and split_num != -1:
            raise Exception('idx and split_num cannot be set simultaneously')
        df = pd.read_csv(df_path)
        df = df[split == split_i]
        if split_num != -1:
            idx = df.index.tolist()
            selected_idx = train_test_split(idx, test_size=0.5, random_state=random_state)[split_num]
            df = df.loc[selected_idx, :]
        self.embeddings = torch.from_numpy(df.loc[:, : embeddings_dim - 1].values)
        self.label = torch.from_numpy(df['label'].values)
        self.group = torch.from_numpy(df['group'].values)
        self.place_label = torch.from_numpy(df['place_label'].values)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        embedding = self.embeddings[idx]
        y = self.label[idx]
        g = self.group[idx]
        p = self.place_label[idx]
        return embedding, y, g, p


class SimpleEmbeddingsDataset(Dataset):
    def __init__(self, embeddings, label, group, place_label):
        self.embeddings = embeddings
        self.label = label
        self.group = group
        self.place_label = place_label

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        embedding = self.embeddings[idx]
        y = self.label[idx]
        g = self.group[idx]
        p = self.place_label[idx]
        return embedding, y, g, p



def get_loader(data, shuffle, batch_size, **kwargs):
    loader = DataLoader(
        data,
        shuffle=shuffle,
        batch_size=batch_size
        )
    return loader