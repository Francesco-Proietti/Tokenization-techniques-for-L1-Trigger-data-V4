import torch
from torch.utils.data import Dataset


class TokenDataset(Dataset):
    
    def __init__(
        self,
        tokens_path,
        masks_path,
        cb_size:int = 512,
        labels_path=None
    ):
        
        self.pad_token = cb_size
        
        self.tokens = torch.load(tokens_path)
        self.masks = torch.load(masks_path)

        self.labels = None

        if labels_path is not None:
            self.labels = torch.load(labels_path)


        assert len(self.tokens) == len(self.masks)

        if self.labels is not None:
            assert len(self.tokens) == len(self.labels)
        
        self.tokens = self.tokens.clone()

        self.tokens[self.tokens == -1] = self.pad_token


    def __len__(self):

        return len(self.tokens)


    def __getitem__(self, idx):

        item = {
            "tokens": self.tokens[idx],
            "mask": self.masks[idx],
        }

        if self.labels is not None:
            item["label"] = self.labels[idx]

        return item