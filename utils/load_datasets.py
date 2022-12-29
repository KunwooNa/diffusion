import torch 
import torchvision
from torch.utils.data import Dataset, DataLoader 
import torchvision.datasets as datasets
import torchvision.transforms as transforms 
import os 
import numpy as np 



def get_dataset_and_dataloader(args, configs): 
    """ 
    Get dataset and dataloader that fits the arguments. 
    """
    dataset_type = args.dataset_type 
    image_size = configs.dataset.image_size 
    in_channels = configs.dataset.in_channels 
    batch_size = configs.training.batch_size 
    is_train = True if args.phase == 'train' else False 

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)), 
        transforms.ToTensor(), 
        transforms.Lambda(lambda t: (t * 2) - 1) # Scale between [-1, 1] 
        ])

    if dataset_type == 'folder':
        dataset = datasets.ImageFolder(
            root = args.dataroot, 
            transform = transform
        )
    elif dataset_type == 'built_in':
        if configs.dataset.name == "mnist":
            dataset = torchvision.datasets.MNIST(
                root = os.path.join(args.dataroot, 'mnist'), 
                train = is_train, 
                download = True,
                transform = transform
            )
        elif configs.dataset.name == 'cifar':
            dataset = torchvision.datasets.CIFAR10(
                root = os.path.join(args.dataroot, 'cifar10'), 
                train = is_train, 
                download = True,
                transform = transform
            )
        
        elif configs.dataset.name == 'stanfordcars':
            dataset = torchvision.datasets.StanfordCars(
                root = os.path.join(args.dataroot, 'stancars'), 
                download = True, 
                transform = transform
            )
            
        elif configs.dataset.name == 'celeb': 
            dataset = torchvision.datasets.CelebA(
                root = os.path.join(args.dataroot, 'celeb'), 
                split = 'train',    # this should be modified. 
                download = True, 
                transform = transform 
            )

            if configs.dataset.use_subset == 'true': 
                indices = list(range(len(dataset))) 
                np.random.shuffle(indices) 
                split = int(np.floor(0.01 * len(dataset)))
                picks = indices[:split]
                dataset = torch.utils.data.Subset(
                    dataset, picks
                )
        
        elif configs.dataset.name == 'imagenet':
            dataset = torchvision.datasets.ImageNet(
                root = os.path.join(args.dataroot, 'imagenet'),         
                split = 'train',    # this should be modified. 
                download = True, 
                transform = transform 
                )

        elif configs.dataset.name == 'lsun':
            dataset = torchvision.datasets.LSUN(
                root = os.path.join(args.dataroot, 'lsun'),
                classes = ['classroom_train'], 
                transform = transform
            )
        
        elif configs.dataset.name == 'country':
            dataset = torchvision.datasets.Country211(
                root = os.path.join(args.dataroot, 'country'),
                split = 'train',  
                transform = transform, 
                download = True
            )
        
        elif configs.dataset.name == 'aircraft':
            dataset = torchvision.datasets.FGVCAircraft(
                root = os.path.join(args.dataroot, 'aircraft'),
                split = 'train',  
                transform = transform, 
                download = True
            )
        
        
    dataloader = DataLoader(
        dataset, batch_size = batch_size, shuffle = True, num_workers = configs.dataset.num_workers, pin_memory = True
    )

    return dataset, dataloader 