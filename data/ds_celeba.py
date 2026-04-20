import torch
from torch.utils.data import Dataset
from PIL import Image  
import pandas as pd
import numpy as np


class CelebA(Dataset):
    def __init__(self, root, transform, split):
        self.root = root
        self.df = pd.read_csv(root / f'attributes_{split}.txt', sep='\s+')
        self.transform = transform

        self.num_factors = 40

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError

class PairedCelebA(CelebA):
    def __init__(self, root, transform, split, use_standard_pairing, additional_view):
        super().__init__(root, transform, split)
        self.use_standard_pairing = use_standard_pairing
        self.additional_view = additional_view
        assert split == 'train'  # Only training needs paired data
        self.pairs = np.load(root / f'pairs_{split}.npy', mmap_mode='r')

    def __len__(self):
        return self.pairs.shape[0]

    def __getitem__(self, idx):

        if self.use_standard_pairing:
            # Standard pairing uses the same image for both views
            # Map idx to dataset index to ensure full coverage each epoch
            idx1 = idx % len(self.df)
            idx2 = idx1

        else:
            # Natural pairing uses the pairs specified by self.pairs (same-identity images)
            pair = self.pairs[idx]
            idx1, idx2 = np.random.permutation(pair)

        row1 = self.df.iloc[idx1]
        row2 = self.df.iloc[idx2]

        # The first 40 columns of the dataframe are the attributes
        z1 = row1.iloc[:self.num_factors].to_numpy(dtype=float)
        z2 = row2.iloc[:self.num_factors].to_numpy(dtype=float)
        z_1to2 = torch.FloatTensor(z2 - z1)
        z_2to1 = -z_1to2

        path = self.root / 'img_align_celeba' / row1['filename']
        img1 = Image.open(path).convert('RGB')

        path = self.root / 'img_align_celeba' / row2['filename']
        img2 = Image.open(path).convert('RGB')

        x1, x2 = self.transform(img1, img2)

        if self.additional_view:
            # Generate one additional augmented view for each image, assuming a symmetric SSL loss
            x3, x4 = self.transform(img1, img2)
        else:
            # Placeholders
            x3, x4 = torch.zeros_like(x1), torch.zeros_like(x2)

        return x1, x2, x3, x4, z_1to2, z_2to1, torch.FloatTensor(z1)


class EvalCelebA(CelebA):
    def __init__(self, root, transform, split):
        super().__init__(root, transform, split)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path = self.root / 'img_align_celeba' / self.df.iloc[idx]['filename']
        image = Image.open(path).convert('RGB')

        z = self.df.iloc[idx, :self.num_factors].to_numpy(dtype=float)
        z = torch.FloatTensor(z)

        image = self.transform(image)

        return image, z