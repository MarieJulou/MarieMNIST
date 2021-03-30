from torch.utils.data import DataLoader #prend un set de data le distribue en différent batch
from torchvision.datasets.mnist import MNIST
from typing import NamedTuple #Tuple = liste d'objets non modifiables

import torchvision.transforms as T


class MNISTDataset (NamedTuple):
    train = MNIST(
        root="./data", 
        train=True, 
        transform=T.Compose([T.RandomRotation(25), T.ToTensor()]), 
        download = True,
    )
    test = MNIST(
        root="./data", 
        train=False, 
        transform=T.ToTensor(), 
        download = True,
    )


class MNISTLoader (NamedTuple):
    train = DataLoader(
        MNISTDataset.train,
        batch_size=32,
        shuffle= True,
        num_workers=4,
        pin_memory=True,
    )
    test = DataLoader(
        MNISTDataset.test,
        batch_size=32,
        shuffle= False,
        num_workers=4,
        pin_memory=True,
    )