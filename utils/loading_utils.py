from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode

from data.ds_celeba import PairedCelebA, EvalCelebA
from data.ds_3dident import PairedThreeDIdent, EvalThreeDIdent
from data.ds_inat import PairediNaturalist1M, EvaliNaturalist1M

import random
from PIL import ImageFilter, ImageOps


MEAN_DICT = {'celeba': [0.50815, 0.42195, 0.37636], 
             '3dident': [0.3292, 0.3278, 0.3215],
             'inat': [0.4687, 0.4848, 0.3779]}

SD_DICT = {'celeba': [0.30430, 0.28164, 0.28003], 
           '3dident': [0.0778, 0.0776, 0.0771],
           'inat': [0.2357, 0.2273, 0.2457]}

def get_paired_loader(args, split='train'):
    # Paired loader for SSL training

    mean = MEAN_DICT[args.dataset]
    sd = SD_DICT[args.dataset]
    input_size = args.input_size

    if args.dataset == 'inat':
        transform = {
            'weak': WeakINatTransform(input_size, mean, sd),
            'strong': StrongINatTransform(input_size, mean, sd)
        }[args.transform_type]

        dataset = PairediNaturalist1M(
            root=args.data_dir,
            transform=transform,
            split=split,
            use_standard_pairing=args.use_standard_pairing,
            additional_view=args.additional_view,
            noise_ratio=args.noise_ratio
        )

    elif args.dataset == 'celeba':

        transform = {
            'weak': WeakCelebATransform(input_size, mean, sd),
            'strong': StrongCelebATransform(input_size, mean, sd),
        }[args.transform_type]

        dataset = PairedCelebA(
                root=args.data_dir,
                transform=transform,
                split=split,
                use_standard_pairing=args.use_standard_pairing,
                additional_view=args.additional_view
            )
    

    elif args.dataset == '3dident':
        transform = {
            'weak': Weak3DIdentTransform(input_size, mean, sd),
            'strong': Strong3DIdentTransform(input_size, mean, sd)
        }[args.transform_type]

        dataset = PairedThreeDIdent(
            root=args.data_dir,
            transform=transform,
            split=split,
            use_standard_pairing=args.use_standard_pairing,
            additional_view=args.additional_view
        )
        
    else:
        raise NotImplementedError
    
    if args.num_workers == 0:
        loader = DataLoader(dataset, 
                    batch_size=args.batch_size, 
                    pin_memory=True, 
                    shuffle=True, 
                    num_workers=0)
    else:
        loader = DataLoader(dataset, 
                    batch_size=args.batch_size, 
                    shuffle=True, 
                    pin_memory=True, 
                    num_workers=args.num_workers, 
                    prefetch_factor=args.prefetch_factor,
                    persistent_workers=True)

    return loader

def get_standard_loader(args, split, train):
    # Standard loader for evaluation, including training the linear probe

    mean = MEAN_DICT[args.dataset]
    sd = SD_DICT[args.dataset]
    input_size = args.input_size

    if args.dataset == 'inat':
        # Train using some weak augmentations
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomResizedCrop(input_size, scale=(0.8, 1.0), ratio=(0.9, 1.1), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=sd)
        ])

        # Evaluate using center crop
        test_transform = transforms.Compose([
            transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=sd)
        ])

        transform = train_transform if train else test_transform

        dataset = EvaliNaturalist1M(
            root=args.data_dir,
            transform=transform,
            split=split
        )

    elif args.dataset == 'celeba':

        # Train using some weak augmentations
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomResizedCrop(input_size, scale=(0.8, 1.0), ratio=(0.9, 1.1), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=sd)
        ])
    
        # Evaluate using center crop
        test_transform = transforms.Compose([
            transforms.CenterCrop(178),
            transforms.Resize(input_size, interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=sd)
        ])

        transform = train_transform if train else test_transform

        dataset = EvalCelebA(
            root=args.data_dir,
            transform=transform,
            split=split
        )

    elif args.dataset == '3dident':
        transform = Weak3DIdentTransform(input_size, mean, sd)
        dataset = EvalThreeDIdent(
            root=args.data_dir,
            transform=transform,
            split=split
        )

    else:
        raise NotImplementedError

    loader = DataLoader(dataset, 
        batch_size=args.batch_size if train else args.eval_batch_size, 
        pin_memory=True, 
        num_workers=args.num_workers if train else 2, 
        prefetch_factor=args.prefetch_factor if train else 2,
        shuffle=train, 
        persistent_workers=False)

    return loader

class WeakINatTransform:
    def __init__(self, input_size, mean, sd):
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomResizedCrop(input_size, scale=(0.8, 1.0), ratio=(0.9, 1.1), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=sd)
        ])

    def __call__(self, x1, x2):
        return self.transform(x1), self.transform(x2)


class StrongINatTransform:
    def __init__(self, input_size, mean, sd):
        self.transform1 = transforms.Compose([
            # Geometric
            transforms.RandomHorizontalFlip(p=0.5),

            # Resize and crop,
            transforms.RandomResizedCrop(input_size, interpolation=InterpolationMode.BICUBIC),

            # Color and blur
            transforms.RandomApply([	
                transforms.ColorJitter( # strong color jitter
                    brightness=0.4, 
                    contrast=0.4, 
                    saturation=0.2, 
                    hue=0.1
                )
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(p=0.5),
            Solarization(p=0.2),

            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=sd)
        ])

        self.transform2 = self.transform1

    def __call__(self, x1, x2):
        return self.transform1(x1), self.transform2(x2)


class WeakCelebATransform:
    def __init__(self, input_size, mean, sd):
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomResizedCrop(input_size, scale=(0.8, 1.0), ratio=(0.9, 1.1), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=sd)
        ])

    def __call__(self, x1, x2):
        return self.transform(x1), self.transform(x2)


class StrongCelebATransform:
    def __init__(self, input_size, mean, sd):
        self.transform1 = transforms.Compose([
            # Geometric
            transforms.RandomHorizontalFlip(p=0.5),

            # Resize and crop,
            transforms.RandomResizedCrop(input_size, interpolation=InterpolationMode.BICUBIC),

            # Color and blur
            transforms.RandomApply([	
                transforms.ColorJitter( # strong color jitter
                    brightness=0.4, 
                    contrast=0.4, 
                    saturation=0.2, 
                    hue=0.1
                )
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(p=0.5),
            Solarization(p=0.2),

            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=sd)
        ])

        self.transform2 = self.transform1

    def __call__(self, x1, x2):
        return self.transform1(x1), self.transform2(x2)



class Weak3DIdentTransform:
    def __init__(self, input_size, mean, sd):
        self.transform = transforms.Compose([
            transforms.Resize(input_size, interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=sd)
        ])

    def __call__(self, x):
        return self.transform(x)


class Strong3DIdentTransform:
    def __init__(self, input_size, mean, sd):

        self.transform = transforms.Compose([
            # Geometric
            transforms.RandomHorizontalFlip(p=0.5),

            # Resize and crop,
            transforms.RandomResizedCrop(input_size, interpolation=InterpolationMode.BICUBIC),

            # Color and blur
            transforms.RandomApply([	
                transforms.ColorJitter( # strong color jitter
                    brightness=0.4, 
                    contrast=0.4, 
                    saturation=0.2, 
                    hue=0.1
                )
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(p=0.5),
            #Solarization(p=0.2),

            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=sd)
        ])

    def __call__(self, x):
        return self.transform(x)
    

class GaussianBlur(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            sigma = random.random() * 1.9 + 0.1
            return img.filter(ImageFilter.GaussianBlur(sigma))
        else:
            return img

class Solarization(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img