import torch
from torch.utils.data import Dataset
from PIL import Image  
import faiss
import numpy as np


class ThreeDIdent(Dataset):
    def __init__(self, root, transform, split):
        self.root = root / split
        self.transform = transform

        self.latents = np.load(self.root / 'raw_latents.npy')

        # Ensure FAISS-compatible dtype
        self.latents = self.latents.astype(np.float32, copy=False)

        self.latent_mean = 0
        self.latent_sd = np.sqrt((2 ** 2) / 12)

        max_length = int(np.ceil(np.log10(len(self.latents))))
        self.image_paths = [
            (self.root / "images" / f"{str(i).zfill(max_length)}.png") for i in range(self.latents.shape[0]) 
        ]

    def __len__(self):
        return self.latents.shape[0]

    def __getitem__(self, idx):
        raise NotImplementedError

class PairedThreeDIdent(ThreeDIdent):
    def __init__(self, root, transform, split, use_standard_pairing, p=0.2, additional_view=False):
        super().__init__(root, transform, split)
        self.use_standard_pairing = use_standard_pairing
        self.p = p
        self.additional_view = additional_view

        # Set up FAISS for quick nearest neighbor search later
        print('Creating FAISS index...')
        self._index = faiss.index_factory(
            self.latents.shape[1], "IVF1024_HNSW32,Flat"
        )
        self._index.efSearch = 8
        self._index.nprobe = 10

        self._index.train(self.latents)
        self._index.add(self.latents)
        print('DONE.')

    def _sample_pair(self, idx):
        # To obtain z2, randomly change some dimensions of z1 and substitute with a random latent u with probability p
        z1 = self.latents[idx]
        i, j = np.random.choice(self.latents.shape[0], size=2, replace=False)
        idx2 = j if i == idx else i
        u = self.latents[idx2]
        shared_dims = np.random.rand(z1.shape[0]) > self.p  
        z2 = np.where(shared_dims, z1, u)
        return z1, z2

    def _search_index(self, z1, z2, index_z1):
        # Search for closest match of z2
        z2_query = np.ascontiguousarray(z2.reshape(1, -1).astype(np.float32))
        _, index_z2 = self._index.search(z2_query, 2)
        
        # Make sure to not use the same image as z1 
        index_z2 = index_z2[0, 0] if index_z2[0, 0] != index_z1 else index_z2[0, 1]

        return index_z2

    def __getitem__(self, idx):
        if self.use_standard_pairing:
            # Standard pairing uses the same image for both views
            z1 = self.latents[idx]
            path_z1 = self.image_paths[idx]
            z2, path_z2 = z1, path_z1

        else:
            # Natural pairing uses different images
            z1, z2 = self._sample_pair(idx)
            index_z1 = idx
            index_z2 = self._search_index(z1, z2, index_z1)

            z1 = self.latents[index_z1]
            z2 = self.latents[index_z2]

            path_z1 = self.image_paths[index_z1]
            path_z2 = self.image_paths[index_z2]

        z1 = (z1 - self.latent_mean) / self.latent_sd
        z2 = (z2 - self.latent_mean) / self.latent_sd

        z_1to2 = torch.FloatTensor(z2 - z1)
        z_2to1 = -z_1to2

        x1 = self.transform(Image.open(path_z1).convert('RGB'))
        x2 = self.transform(Image.open(path_z2).convert('RGB'))

        if self.additional_view:
            # Generate one additional augmented view for each image
            # We don't use this for 3DIdent; having it here for consistency.
            x3 = self.transform(Image.open(path_z1).convert('RGB'))
            x4 = self.transform(Image.open(path_z2).convert('RGB'))
        else:
            x3, x4 = torch.zeros_like(x1), torch.zeros_like(x2)   

        return x1, x2, x3, x4, z_1to2, z_2to1, torch.FloatTensor(z1)


class EvalThreeDIdent(ThreeDIdent):
    def __init__(self, root, transform, split):
        super().__init__(root, transform, split)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        image = Image.open(path).convert('RGB')

        z = self.latents[idx]
        z = (z - self.latent_mean) / self.latent_sd
        z = torch.FloatTensor(z)

        image = self.transform(image)

        return image, z