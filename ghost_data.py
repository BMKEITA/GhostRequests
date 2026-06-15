# ghost_data.py
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np

def get_transforms(dataset_name, augment=False):
    if dataset_name == 'mnist':
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    elif dataset_name == 'cifar10':
        if augment:
            return transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.2, 0.2, 0.2),
                transforms.ToTensor(),
                transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010)),
                transforms.RandomErasing(p=0.1)
            ])
        else:
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010))
            ])

def load_datasets(dataset_name, augment=False):
    tf      = get_transforms(dataset_name, augment)
    tf_test = get_transforms(dataset_name, False)
    if dataset_name == 'mnist':
        train = datasets.MNIST('./datasets', train=True,  download=True, transform=tf)
        test  = datasets.MNIST('./datasets', train=False, download=True, transform=tf_test)
    elif dataset_name == 'cifar10':
        train = datasets.CIFAR10('./datasets', train=True,  download=True, transform=tf)
        test  = datasets.CIFAR10('./datasets', train=False, download=True, transform=tf_test)
    return train, test

def iid_split(trainset, n_clients):
    n       = len(trainset)
    indices = torch.randperm(n).tolist()
    size    = n // n_clients
    return [Subset(trainset, indices[i*size:(i+1)*size]) for i in range(n_clients)]

def noniid_split(trainset, n_clients, shards_per_client=2):
    targets    = np.array(trainset.targets)
    sorted_idx = np.argsort(targets)
    n_shards   = n_clients * shards_per_client
    shard_size = len(trainset) // n_shards
    shards     = [sorted_idx[i*shard_size:(i+1)*shard_size] for i in range(n_shards)]
    np.random.shuffle(shards)
    client_indices = []
    for i in range(n_clients):
        idx = np.concatenate(shards[i*shards_per_client:(i+1)*shards_per_client])
        client_indices.append(Subset(trainset, idx.tolist()))
    return client_indices

def get_data_loaders(FL_params):
    train, test = load_datasets(FL_params.data_name, augment=(FL_params.data_name=='cifar10'))
    kwargs      = {'num_workers': 4, 'pin_memory': True} if 'cuda' in FL_params.device else {}
    test_loader = DataLoader(test, batch_size=FL_params.test_batch_size, shuffle=False, **kwargs)
    if FL_params.data_split == 'iid':
        client_sets = iid_split(train, FL_params.N_total_client)
    else:
        client_sets = noniid_split(train, FL_params.N_total_client)
    client_loaders = [
        DataLoader(client_sets[i], FL_params.local_batch_size, shuffle=True, **kwargs)
        for i in range(FL_params.N_total_client)
    ]
    return client_loaders, test_loader, train, test
