import numpy as np
from glob import glob

import cv2 as cv

import torch
from torch.utils.data import Dataset, DataLoader


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

class myDataset(Dataset):
    def __init__(self, device, transform=None):
        super().__init__()
        self.images = []
        self.targets = []
        self.device = device


        fnames = glob("./yes/*png")
        for fname in fnames:
            img = cv.imread(fname, cv.IMREAD_GRAYSCALE) / 256
            self.images.append(torch.tensor(img, dtype=torch.float32))
            self.targets.append(torch.tensor(1, dtype=torch.float32))

        fnames = glob("./no/*png")
        for fname in fnames:
            img = cv.imread(fname, cv.IMREAD_GRAYSCALE) / 256
            self.images.append(torch.tensor(img, dtype=torch.float32))
            self.targets.append(torch.tensor(0, dtype=torch.float32))


    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # here adding a dummy dimension for greyscale images
        #  would need to be adapted for RBG (or otherwise multiple inputs)
        #return self.images[idx], self.targets[idx]

        return torch.tensor(self.images[idx], dtype=torch.float).to(self.device)[None,:,:], self.targets[idx]

if __name__ == "__main__":
    device = get_device()

    BATCH_SIZE=16
    dataset = myDataset(device)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    train_features, train_labels = next(iter(dataloader))
    print(train_labels)

