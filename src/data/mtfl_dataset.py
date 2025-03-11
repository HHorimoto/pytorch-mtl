import torch
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Compose, Resize

from PIL import Image
import pathlib
import numpy as np
import pandas as pd

from src.utils.seeds import worker_init_fn, generator

class MTFLDataset(torch.utils.data.Dataset):
    def __init__(self, dir_path, is_train=True):

        self.dir_path = dir_path

        if is_train:
            txt_path = pathlib.Path(self.dir_path) / "training.txt"
        else:
            txt_path = pathlib.Path(self.dir_path) / "testing.txt"

        self.data_df = pd.read_csv(txt_path, sep=" ", header=None, skipinitialspace=True, skipfooter=1, engine="python", 
                                   names=["image_path", "x1","x2","x3","x4","x5", "y1","y2","y3","y4","y5", 
                                          "gender", "smile", "wearing_glasses", "head_pose"])
        
        self.transform = Compose([Resize(size=(40, 40)), 
                                  ToTensor(),])

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, index):
        path = pathlib.Path(self.dir_path) / self.data_df.iloc[index, 0].replace('\\', '/')
        image = Image.open(path).convert('RGB')

        if self.transform:
            X = self.transform(image)
        
        target_df = self.data_df.iloc[index]
        y1 = int(str(target_df["smile"] - 1))
        y2 = int(str(target_df["gender"] - 1))

        return X, y1, y2

def create_dataset(data_path, batch_size):
    train_dataset = MTFLDataset(data_path, is_train=True)
    test_dataset = MTFLDataset(data_path, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=True, worker_init_fn=worker_init_fn,)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=2, pin_memory=True, worker_init_fn=worker_init_fn,)
    
    return train_loader, test_loader