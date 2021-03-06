import torch
import torchvision
from torchvision.io import read_image
import os
from torch.utils.data import Dataset


"""
This part of the script is an easy API to load and get the datasets
"""


def get_dataset(dataset: str, c_angle=30, new_size=[32, 32], batch_size=300):
    """
    :param new_size:
    :param c_angle:
    :param dataset: chosen dataset
    :return: train loader ,test loader and input size
    """
    if dataset == 'FASHION_MNIST':
        train_set = FASHION_MNIST('./data/' + dataset + '/', download=True, train=True,
                                  transform=torchvision.transforms.ToTensor())
        test_set = FASHION_MNIST('./data/' + dataset + '/', download=True, train=False,
                                 transform=torchvision.transforms.ToTensor())
        input_size = (28, 28, 1)
    elif dataset == 'Rotate FASHION_MNIST':
        rotate_tran_fun = lambda x: rotate_tran(x, angle=c_angle)
        train_set = FASHION_MNIST('./data/' + dataset + f'_Rotate_{c_angle}/', download=True, train=True,
                                  transform=rotate_tran_fun)
        test_set = FASHION_MNIST('./data/' + dataset + f'_Rotate_{c_angle}/', download=True, train=False,
                                 transform=rotate_tran_fun)
        input_size = (28, 28, 1)
    elif dataset == 'LFW':
        train_set = LFW('./data/' + dataset + '/', split='train',
                        transform=torchvision.transforms.ToTensor(), download=True)
        test_set = LFW('./data/' + dataset + '/', split='test',
                       transform=torchvision.transforms.ToTensor(), download=True)
        input_size = (250, 250)
    elif dataset == 'LFW_resize':
        resize_tran_fun = lambda x: resize_tran(x, new_size=new_size)
        train_set = LFW('./data/' + dataset + f'_{new_size}/', split='train', transform=resize_tran_fun, download=True)
        test_set = LFW('./data/' + dataset + f'_{new_size}/', split='test', transform=resize_tran_fun, download=True)
        input_size = (new_size[0], new_size[1], 3)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True, drop_last=True)
    return train_loader, test_loader, input_size, batch_size


def rotate_tran(img, angle):
    tensor_img = torchvision.transforms.ToTensor(img)
    return torchvision.transforms.functional.rotate(img=tensor_img, angle=angle)


def resize_tran(img, new_size=[32, 32]):
    tensor_img = torchvision.transforms.ToTensor(img)
    return torchvision.transforms.functional.resize(img=tensor_img, size=new_size)


class FASHION_MNIST(torchvision.datasets.FashionMNIST):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, index):
        return super().__getitem__(index)[0]


class LFW(torchvision.datasets.LFWPeople):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, index):
        return super().__getitem__(index)[0]


class Paining(Dataset):
    def __init__(self, img_dir, transform=None, target_transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image