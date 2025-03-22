import torch
from torchvision import datasets

from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Compose, Normalize
from PIL import Image
import pathlib

from src.utils.seeds import worker_init_fn

class CIFAR10Dataset(torch.utils.data.Dataset):
    def __init__(self, root='./data/', is_train=True, download=True):
        if is_train:
            self.transform = Compose([
                ToTensor(),
                Normalize(0.5, 0.5), 
            ])
        else:
            self.transform = Compose([
                ToTensor(),
                Normalize(0.5, 0.5), 
            ])

        self.data = datasets.CIFAR10(root=root, train=is_train, download=download, transform=self.transform)

        self.labels_list = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                            'dog', 'frog', 'horse', 'ship', 'truck']
        self.non_animal = [0, 1, 8, 9]

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        image = self.data[index][0]
        label1 = self.data[index][1] # original label
        label2 = 0 if self.data[index][1] in self.non_animal else 1 # animal or vehicle
        return image, label1, label2

def create_dataset(root='./data/', download=True, batch_size=64):

    train_dataset = CIFAR10Dataset(root=root, is_train=True, download=download)
    test_dataset = CIFAR10Dataset(root=root, is_train=False, download=download)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=True, worker_init_fn=worker_init_fn,)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=2, pin_memory=True, worker_init_fn=worker_init_fn,)
    
    return train_loader, test_loader