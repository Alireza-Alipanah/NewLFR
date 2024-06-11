from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import pandas as pd


class EmbeddingsDataset(Dataset):
    def __init__(self, df_path, embeddings_dim, split="train", split_num=-1, random_state=None):
        try:
            split_i = ["train", "val", "test"].index(split)
        except ValueError:
            raise(f"Unknown split {split}")
        df = pd.read_csv(df_path)
        df = df[df.split == split_i].reset_index()
        if split_num != -1:
            idx = df.index.tolist()
            selected_idx = train_test_split(idx, test_size=0.5, random_state=random_state)[split_num]
            df = df.iloc[selected_idx, :].reset_index()
        self.embeddings = df.iloc[:, : embeddings_dim - 1].values
        self.label = df['label'].values
        self.group = df['group'].values
        self.place_label = df['place_label'].values

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