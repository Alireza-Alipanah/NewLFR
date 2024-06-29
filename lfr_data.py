from torch.utils.data import Dataset, DataLoader
import pandas as pd


class EmbeddingsDataset(Dataset):
    def __init__(self, df_path, embeddings_dim, split="train", split_num=-1, random_state=None):
        try:
            split_i = ["train", "val", "test"].index(split)
        except ValueError:
            raise(f"Unknown split {split}")
        df = pd.read_csv(df_path)
        self.embeddings = df.iloc[:, :embeddings_dim].values
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