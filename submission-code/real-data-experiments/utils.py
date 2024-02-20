# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from datasets import load_dataset
from collections import OrderedDict
from opacus.validators import ModuleValidator
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
from torchvision import datasets, transforms
from typing import Union, Type, TypeVar

# Attribute target and range for Bayes-SGD AI analysis.
MNIST_ATTRIBUTE_IDX = [0, 0, 0]
MNIST_ATTRIBUTE_RANGE = [0, 1]
PURCHASE100_ATTRIBUTE_IDX = [0]
PURCHASE100_ATTRIBUTE_RANGE = [0, 1]
ADULT_ATTRIBUTE_IDX = [0]
ADULT_ATTRIBUTE_RANGE = np.arange(17, 91)
# Normalized:
ADULT_ATTRIBUTE_RANGE = (ADULT_ATTRIBUTE_RANGE - 38.5816)/13.6404

# Precomputed characteristics of the MNIST dataset.
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081

class SampleConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 8, 2, padding=3)
        self.conv2 = nn.Conv2d(16, 32, 4, 2)
        self.fc1 = nn.Linear(32 * 4 * 4, 32)
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x):
        # x of shape [B, 1, 28, 28]
        x = F.relu(self.conv1(x))  # -> [B, 16, 14, 14]
        x = F.max_pool2d(x, 2, 1)  # -> [B, 16, 13, 13]
        x = F.relu(self.conv2(x))  # -> [B, 32, 5, 5]
        x = F.max_pool2d(x, 2, 1)  # -> [B, 32, 4, 4]
        x = x.view(-1, 32 * 4 * 4)  # -> [B, 512]
        x = F.relu(self.fc1(x))  # -> [B, 32]
        x = self.fc2(x)  # -> [B, 10]
        return x

    def name(self):
        return "SampleConvNet"

def mnist(lr, batch_size, test_batch_size, data_root="./mnist-data"):
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            data_root,
            train=True,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
                ]
            ),
        ),
        batch_size=batch_size,
        num_workers=0,
        pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            data_root,
            train=False,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
                ]
            ),
        ),
        batch_size=test_batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    init_model = SampleConvNet
    init_optimizer = lambda model: optim.SGD(model.parameters(), lr=lr, momentum=0)

    return train_loader, test_loader, init_model, init_optimizer


Y = TypeVar("Y", bound="MLP")

class MLP(nn.Module):
    """
    The fully-connected network architecture from Bao et al. (2022).
    """
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(600, 128), nn.Tanh(),
            nn.Linear(128, 100)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)

    @classmethod
    def load(cls: Type[Y], path: Union[str, os.PathLike]) -> Y:
        model = cls()
        state_dict = torch.load(path)
        new_state_dict = OrderedDict((k.replace('_module.', ''), v) for k, v in state_dict.items())
        model.load_state_dict(new_state_dict)
        model.eval()
        return model


class Purchase100(Dataset):
    """
    Purchase100 dataset pre-processed by Shokri et al.
    (https://github.com/privacytrustlab/datasets/blob/master/dataset_purchase.tgz).
    We save the dataset in a .pickle version because it is much faster to load
    than the original file.
    """
    def __init__(self, dataset_dir: str) -> None:
        import pickle

        dataset_path = os.path.join(dataset_dir, 'purchase100', 'dataset_purchase')

        # Saving the dataset in pickle format because it is quicker to load.
        dataset_path_pickle = dataset_path + '.pickle'

        if not os.path.exists(dataset_path) and not os.path.exists(dataset_path_pickle):
            raise ValueError("Purchase-100 dataset not found.\n"
                             "You may download the dataset from https://www.comp.nus.edu.sg/~reza/files/datasets.html\n"
                            f"and unzip it in the {dataset_dir}/purchase100 directory")

        if not os.path.exists(dataset_path_pickle):
            print('Found the dataset. Saving it in a pickle file that takes less time to load...')
            purchase = np.loadtxt(dataset_path, dtype=int, delimiter=',')
            with open(dataset_path_pickle, 'wb') as f:
                pickle.dump({'dataset': purchase}, f)

        with open(dataset_path_pickle, 'rb') as f:
            dataset = pickle.load(f)['dataset']

        self.labels = list(dataset[:, 0] - 1)
        self.records = torch.FloatTensor(dataset[:, 1:])
        assert len(self.labels) == len(self.records), f'ERROR: {len(self.labels)} and {len(self.records)}'
        print('Successfully loaded the Purchase-100 dataset consisting of',
            f'{len(self.records)} records and {len(self.records[0])}', 'attributes.')

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        return self.records[idx], self.labels[idx]


def purchase100(lr, batch_size, test_batch_size, data_root="./purchase100-data"):
    """Loads the Purchase-100 dataset.
    """
    data = Purchase100(data_root)
    len_training = int(len(data)*.9)
    train_dataset, test_dataset = random_split(data, [len_training, len(data)-len_training])
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
    )

    init_model = MLP
    init_optimizer = lambda model: optim.Adam(model.parameters(), lr=lr)

    return train_loader, test_loader, init_model, init_optimizer

class MLPAdult(nn.Module):
    """
    The fully-connected network architecture from Bao et al. (2022).
    """
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(108, 128),
            nn.Tanh(),
            nn.Linear(128, 2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)

    @classmethod
    def load(cls: Type[Y], path: Union[str, os.PathLike]) -> Y:
        model = cls()
        state_dict = torch.load(path)
        new_state_dict = OrderedDict((k.replace('_module.', ''), v) for k, v in state_dict.items())
        model.load_state_dict(new_state_dict)
        model.eval()
        return model

class Adult(Dataset):
    """
    Adult Census Income dataset.
    """
    def __init__(self, dataset_dir: str) -> None:
        data = load_dataset("scikit-learn/adult-census-income", split="train", cache_dir=dataset_dir)
        data = data.to_pandas()

        # One-hot encoding.
        numeric_subset = data.select_dtypes('number')
        categorical_subset = data.select_dtypes('object')
        categorical_subset = pd.get_dummies(categorical_subset[categorical_subset.columns.drop("income")])

        self.labels = np.array(data["income"]==">50K", dtype=int)
        self.records = torch.FloatTensor(pd.concat([numeric_subset, categorical_subset], axis=1).values)

        # Normalise.
        self.records = (self.records - self.records.mean(axis=0)) / self.records.std(axis=0)

        # Check that normalized attributes for AI analysis are correct.
        attribute_values = set(np.unique(self.records[:, ADULT_ATTRIBUTE_IDX]))
        for a in attribute_values:
            # We check that for each attribute value, at least one from the (normalized) range covers it.
            assert np.sum(np.isclose(a, ADULT_ATTRIBUTE_RANGE, 0.0001, 0.0001)) == 1, f'ERROR: {a} not in {ADULT_ATTRIBUTE_RANGE}'

        print('Successfully loaded the Adult dataset consisting of',
            f'{len(self.records)} records and {len(self.records[0])}', 'attributes.')

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        return self.records[idx], self.labels[idx]


def adult(lr, batch_size, test_batch_size, data_root="./adult-data"):
    """Loads the Adult Census Income dataset.
    """
    data = Adult(data_root)

    # Count labels.
    n_labels = min(sum(data.labels), len(data.labels)-sum(data.labels))

    # Reduce to have a balanced dataset:
    # data = torch.utils.data.Subset(data, np.where(data.labels==0)[0][:n_labels].tolist() + np.where(data.labels==1)[0][:n_labels].tolist())

    len_training = int(len(data)*.8)

    train_dataset, test_dataset = random_split(data, [len_training, len(data)-len_training])

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
    )

    init_optimizer = lambda model: optim.Adam(model.parameters(), lr=lr)

    return train_loader, test_loader, MLPAdult, init_optimizer
