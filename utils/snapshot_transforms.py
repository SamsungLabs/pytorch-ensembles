import torchvision.transforms as transforms


"""Specific mean and std values are used for compatibility with trained models
There should not influence the training process much because of the batchnorm
"""

def compose_transform(mean, std, aug):
    if aug:
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

class VGG:
    train = compose_transform(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], aug=True)
    test  = compose_transform(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], aug=False)

class ResNet:
    train = compose_transform(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010], aug=True)
    test  = compose_transform(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010], aug=False)
    
class CIFAR100_CSGMCMC:
    train = compose_transform(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761], aug=True)
    test  = compose_transform(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761], aug=False)

def get_transform(method, model):
    if method == 'fge':
        return VGG
    elif method == 'swag':
        if model == 'VGG16BN':
            return VGG
        else:
            return ResNet
    elif method == 'sse' or method == 'csgld':
        return CIFAR100_CSGMCMC
    else:
        raise NotImplementedError('Unknown method')
