import torch
import numpy as np
from torch.utils.data.sampler import Sampler

class BatchReplace(Sampler):
    def __init__(self, idx, batch_size):
        self.idx = idx
        self.batch_size = batch_size

    def __iter__(self):
        while True:
            yield np.random.choice(self.idx, self.batch_size, replace=False)

    def __len__(self):
        return len(self.idx)

class SubsetSampler(Sampler):
    def __init__(self, indices):
        self.indices = indices
    def __iter__(self):    
        return iter(self.indices)
    def __len__(self):
        return len(self.indices)