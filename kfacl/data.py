import numpy as np
import torch
import torchvision
import os

import torchvision.datasets as datasets

c10_classes = np.array([
    [0, 1, 2, 8, 9],
    [3, 4, 5, 6, 7]
], dtype=np.int32)


def imagenet_loaders(path, batch_size, num_workers, transform_train, transform_test):
    # Path: path to raw-data of imagenet
    # TODO: Validation
    train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(os.path.join(path, 'train'), transform_train),
        batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(os.path.join(path, 'val'), transform_test),
        batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True)
    loaders = {'test': val_loader, 'train': train_loader}
    return loaders


def loaders(dataset, path, batch_size, num_workers, transform_train, transform_test, 
            use_validation=True, val_size=5000, split_classes=None, shuffle_train=True,
            **kwargs):
    if dataset == 'ImageNet':
        return imagenet_loaders(path, batch_size, num_workers, transform_train, transform_test)
    if dataset == 'CamVid':
        raise NotImplementedError

    path = os.path.join(path, dataset.lower())
    
    ds = getattr(torchvision.datasets, dataset)            

    if dataset == 'SVHN':
        raise NotImplementedError
    else:
        ds = getattr(torchvision.datasets, dataset)           

    if dataset == 'STL10':
        train_set = ds(root=path, split='train', download=True, transform=transform_train)
        num_classes = 10
        cls_mapping = np.array([0, 2, 1, 3, 4, 5, 7, 6, 8, 9])
        train_set.labels = cls_mapping[train_set.labels]
    else:
        train_set = ds(root=path, train=True, download=True, transform=transform_train)
        num_classes = max(train_set.targets) + 1

    if use_validation:
        print("Using train (" + str(len(train_set.data)-val_size) + ") + validation (" +str(val_size)+ ")")
        train_set.data = train_set.data[:-val_size]
        train_set.targets = train_set.targets[:-val_size]

        test_set = ds(root=path, train=True, download=True, transform=transform_test)
        test_set.train = False
        test_set.data = test_set.data[-val_size:]
        test_set.targets = test_set.targets[-val_size:]
        # delattr(test_set, 'train_data')
        # delattr(test_set, 'train_labels')
    else:
        print('You are going to run models on the test set. Are you sure?')
        if dataset == 'STL10':
            test_set = ds(root=path, split='test', download=True, transform=transform_test)
            test_set.targets = cls_mapping[test_set.targets]
        else:
            test_set = ds(root=path, train=False, download=True, transform=transform_test)

    if split_classes is not None:
        assert dataset == 'CIFAR10'
        assert split_classes in {0, 1}

        print('Using classes:', end='')
        print(c10_classes[split_classes])
        train_mask = np.isin(train_set.targets, c10_classes[split_classes])
        train_set.data = train_set.data[train_mask, :]
        train_set.targets = np.array(train_set.targets)[train_mask]
        train_set.targets = np.where(train_set.targets[:, None] == c10_classes[split_classes][None, :])[1].tolist()
        print('Train: %d/%d' % (train_set.data.shape[0], train_mask.size))

        test_mask = np.isin(test_set.targets, c10_classes[split_classes])
        test_set.data = test_set.data[test_mask, :]
        test_set.targets = np.array(test_set.targets)[test_mask]
        test_set.targets = np.where(test_set.targets[:, None] == c10_classes[split_classes][None, :])[1].tolist()
        print('Test: %d/%d' % (test_set.data.shape[0], test_mask.size))

    return \
        {
            'train': torch.utils.data.DataLoader(
                train_set,
                batch_size=batch_size,
                shuffle=True and shuffle_train,
                num_workers=num_workers,
                pin_memory=True
            ),
            'test': torch.utils.data.DataLoader(
                test_set,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True
            ),
        }, \
        num_classes

