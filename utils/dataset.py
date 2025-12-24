import os
import torchvision
from torchvision import transforms
from .autoaug import CIFAR10Policy
from .random_erasing import RandomErasing



def get_dataset(args):
    mean = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
    std = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
    dir_ = os.path.join(os.getcwd(), 'data')

    if "ViT" not in args.model:
        transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
            ])
        transform_test=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        trainset = torchvision.datasets.CIFAR100(root=dir_, train=True,
                                         download=True, transform=transform)
        testset = torchvision.datasets.CIFAR100(root=dir_, train=False,
                                        download=True, transform=transform_test)
    
    else:
        normalize = [transforms.Normalize(mean=mean, std=std)]
        augmentations = []

        augmentations += [
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4)
                ]
        #autoaugmentation
        augmentations += [
                CIFAR10Policy()
                ]
        augmentations += [
            transforms.ToTensor(),
            *normalize]
        #RandomErasing
        augmentations += [
            RandomErasing(probability = args.re, sh = args.re_sh, r1 = args.re_r1, mean=mean)
            ]
        augmentations = transforms.Compose(augmentations)

        trainset = torchvision.datasets.CIFAR100(
            root=dir_, train=True, download=True, transform=augmentations)

        testset = torchvision.datasets.CIFAR100(
            root=dir_, train=False, download=True, transform=transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            *normalize]))
        

    
    return trainset,testset
        
