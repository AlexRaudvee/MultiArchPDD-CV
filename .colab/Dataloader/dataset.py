import torch
import random 
from typing import DefaultDict

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset):
        self.ds = base_dataset
        # build class→idx list
        self.cls2idx = DefaultDict(list)
        for i, (_, y) in enumerate(self.ds):
            self.cls2idx[y].append(i)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        return self.ds[idx]

    def class_sample(self, c, k=1):
        idxs = random.choices(self.cls2idx[c], k=k)
        imgs, labs = zip(*(self.ds[i] for i in idxs))
        return torch.stack(imgs), torch.tensor(labs)
