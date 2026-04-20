import torch
from torch.utils.data import Dataset
from PIL import Image  
import numpy as np
import pickle
import random

class iNaturalist1M(Dataset):
    def __init__(self, root, transform, split):
        self.root = root
        self.transform = transform
        self.split = split

        self.paths = []
        self.cat_labels = []
        self.super_labels = []

        self.num_classes = 5000
        self.num_super_classes = 11

        y_max = 0
        y_super_max = 0

        # Load paths, category, and super-category labels
        with open(f'{root}/{split}.txt', 'r') as f:
            for line in f:
                p, c, s = line.strip().split()
                self.paths.append(p)
                self.cat_labels.append(int(c))
                self.super_labels.append(int(s))
                y_max = max(y_max, int(c))
                y_super_max = max(y_super_max, int(s))

        assert self.num_classes == y_max + 1
        assert self.num_super_classes == y_super_max + 1

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError


class PairediNaturalist1M(iNaturalist1M):
    def __init__(self, root, transform, split, additional_view, use_standard_pairing=False, noise_ratio=0.0):
        super().__init__(root, transform, split)
        self.use_standard_pairing = use_standard_pairing
        self.additional_view = additional_view
        self.noise_ratio = noise_ratio

        self.cat_to_indices = pickle.load(open(f'{self.root}/cat_to_indices.pkl', 'rb'))
        self.super_to_indices = pickle.load(open(f'{self.root}/super_to_indices.pkl', 'rb'))
        
        self.label_types = np.loadtxt(f'{self.root}/label_types_n{noise_ratio}.txt', dtype=int)
        assert self.label_types.shape[0] == len(self.paths)

        # We make sure that cleanly labeled images are paired with other cleanly labeled images.
        # Corrupted images are paired with other corrupted images.
        # This makes sure that we indeed get noise_ratio % of corrupted pairs.

        for key, indices in self.cat_to_indices.items():
            self.cat_to_indices[key] = [i for i in indices if self.label_types[i] == 0]
        print('Cleaning cat_to_indices... done')

        for key, indices in self.super_to_indices.items():
            self.super_to_indices[key] = [i for i in indices if self.label_types[i] == 1]
        print('Cleaning super_to_indices... done')


    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):

        if self.use_standard_pairing:
            # Standard pairing uses the same image for both views
            # We don't use it in this experiment
            idx1, idx2 = idx, idx

        else:
            # Natural pairing pairs up images based on their category or super-category labels
            # Label type decides whether we use the fine-grained (category) or coarse-grained (super-category) labels 
            pool_choice = self.label_types[idx]
            cat = self.cat_labels[idx]
            super_cat = self.super_labels[idx]
            pool = self.cat_to_indices[cat] if pool_choice == 0 else self.super_to_indices[super_cat]
            
            if len(pool) < 2:
                # Fallback to same image if pool is too small
                idx1, idx2 = idx, idx
            else:
                # Randomly sample two different images from the pool
                idx1, idx2 = random.sample(pool, 2)

        path = self.root / self.paths[idx1]
        img1 = Image.open(path).convert('RGB')

        path = self.root / self.paths[idx2]
        img2 = Image.open(path).convert('RGB')

        x1, x2 = self.transform(img1, img2)

        if self.additional_view:
            # Generate one additional augmented view for each image, assuming a symmetric SSL loss
            x3, x4 = self.transform(img1, img2)
        else:
            # Placeholders
            x3, x4 = torch.zeros_like(x1), torch.zeros_like(x2)

        y1 = self.cat_labels[idx1]
        y1_super = self.super_labels[idx1]

        return x1, x2, x3, x4, y1, y1_super


class EvaliNaturalist1M(iNaturalist1M):
    def __init__(self, root, transform, split):
        super().__init__(root, transform, split)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.root / self.paths[idx]
        img = Image.open(path).convert('RGB')

        x = self.transform(img)

        y = self.cat_labels[idx]
        y_super = self.super_labels[idx]

        return x, y, y_super