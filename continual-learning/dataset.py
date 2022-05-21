import torch
from torchvision import datasets, transforms
from PIL import Image
from typing import Any, Callable, Optional, Tuple
import torch.nn as nn
import torch.nn.functional as F
from math import ceil
import torchvision.transforms as transforms
import os

MEANS = {"cifar": [0.4914, 0.4822, 0.4465]}
STDS = {"cifar": [0.2023, 0.1994, 0.2010]}


class TensorCIFAR100(datasets.CIFAR100):
    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        super(TensorCIFAR100, self).__init__(
            root, train, transform, target_transform, download
        )

        self.data = torch.stack(
            [transforms.ToTensor()(Image.fromarray(i)) for i in self.data]
        )
        self.targets = torch.tensor(self.targets)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        return img, target


class TensorDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels, transform=None):  # images: n x c x h x w tensor
        self.tensors = images.detach().cpu().float()
        self.targets = labels.detach().cpu()
        self.transform = transform

    def __getitem__(self, index):
        sample = self.tensors[index]
        if self.transform != None:
            sample = self.transform(sample)

        target = self.targets[index]
        return sample, target

    def __len__(self):
        return self.tensors.shape[0]


def get_baseline_dataset(type, net):
    if net in ["convnet", "resnet10"]:
        file_path = f"data/{type}_cifar100_{net}.pt"
        x, y = torch.load(file_path)
        train_dataset = TensorDataset(x, y)
    else:
        raise NotImplementedError()
    print(f"Loaded {type} dataset from {file_path}!")

    return train_dataset


def transform_cifar(augment=False, from_tensor=False, normalize=True):
    if not augment:
        aug = []
        print("Loader without augmentation")
    else:
        aug = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
        print("Dataset with basic Cifar augmentation")

    if from_tensor:
        cast = []
    else:
        cast = [transforms.ToTensor()]

    if normalize:
        normal_fn = [transforms.Normalize(mean=MEANS["cifar"], std=STDS["cifar"])]
    else:
        normal_fn = []

    train_transform = transforms.Compose(cast + aug + normal_fn)
    test_transform = transforms.Compose(cast + normal_fn)

    return train_transform, test_transform


def load_data_path(save_dir, augment, normalize=True):
    train_transform, _ = transform_cifar(
        augment=augment, from_tensor=True, normalize=normalize
    )
    data, target = torch.load(os.path.join(f"{save_dir}", "data.pt"))

    train_dataset = TensorDataset(data, target, train_transform)
    print("Load condensed data ", save_dir, data.shape)

    print("Training data shape: ", train_dataset[0][0].shape)

    return train_dataset


def decode_zoom(img, target, factor):
    # For CIFAR-100
    resize = nn.Upsample(size=32, mode="bilinear")

    h = img.shape[-1]
    remained = h % factor
    if remained > 0:
        img = F.pad(img, pad=(0, factor - remained, 0, factor - remained), value=0.5)
    s_crop = ceil(h / factor)
    n_crop = factor**2

    cropped = []
    for i in range(factor):
        for j in range(factor):
            h_loc = i * s_crop
            w_loc = j * s_crop
            cropped.append(img[:, :, h_loc : h_loc + s_crop, w_loc : w_loc + s_crop])
    cropped = torch.cat(cropped)
    data_dec = resize(cropped)
    target_dec = torch.cat([target for _ in range(n_crop)])

    return data_dec, target_dec


def decode(trainset, factor, augment):
    data = []
    target = []
    for img, label in trainset:
        data.append(img)
        target.append(label)

    data = torch.stack(data)
    target = torch.tensor(target)
    data, target = decode_zoom(data, target, factor)
    print("Dataset is decoded! ", data.shape)

    train_transform, _ = transform_cifar(augment=augment, from_tensor=True)
    train_dataset = TensorDataset(data, target, train_transform)

    return train_dataset


def get_condense_dataset(ipc=10, factor=1, augment=False, phase=0):
    if factor == 1:
        init = "random"
        augment_load = augment
        normalize = True
    else:
        init = "mix"
        augment_load = False
        normalize = False

    base_path = "data"
    file_path = f"idc_cifar100/conv3in_grad_mse_nd2000_cut_niter2000_factor{factor}_lr0.005_{init}_ipc{ipc}_20_phase{phase}"
    save_dir = os.path.join(base_path, file_path)

    train_dataset = load_data_path(save_dir, augment_load, normalize=normalize)
    if factor > 1:
        # If factor > 1, augment and normalize here
        train_dataset = decode(train_dataset, factor, augment)

    return train_dataset
