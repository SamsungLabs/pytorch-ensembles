import torch
import torchvision
import os

def loaders(dataset, path, batch_size, num_workers, transform_train, transform_test, shuffle_train=True):
    ds = getattr(torchvision.datasets, dataset)
    path = os.path.join(path, dataset.lower())
    train_set = ds(path, train=True, download=True, transform=transform_train)
    test_set = ds(path, train=False, download=True, transform=transform_test)
    num_classes = max(train_set.targets) + 1
    
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=shuffle_train,
                                               num_workers=num_workers, pin_memory=True)
        
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False,
                                               num_workers=num_workers, pin_memory=True)

    return ({'train': train_loader, 'test': test_loader}, num_classes)
