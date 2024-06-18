# -*- coding: utf-8 -*-

import torch
import pickle

from pyLSHash import storage
from torch import nn


class TorchLSHash(nn.Module):
    """ TorchLSHash implments locality sensitive hashing using random projection for
    input vectors of dimension `input_dim` in PyTorch.

    Based off of the pyLSHash library implementation, but converted for use in PyTorch.
    This allows us to perform operations on GPU tensors directly without having to move them to/from CPU.

    Attributes:

    :param hash_size:
        The length of the resulting binary hash in integer. E.g., 32 means the
        resulting binary hash will be 32-bit long.
    :param input_dim:
        The dimension of the input vector. E.g., a grey-scale picture of 30x30
        pixels will have an input dimension of 900.
    :param num_hashtables:
        (optional) The number of hash tables used for multiple lookups.
    :param storage:
        An object to store data
    """

    def __init__(self, hash_size, input_dim, num_hashtables=1,
                 storage_instance: storage.StorageBase = storage.InMemoryStorage(''), device='cpu'):
        super().__init__()
        self.hash_size = hash_size
        self.input_dim = input_dim
        self.num_hashtables = num_hashtables
        self.storage_instance = storage_instance
        self.device = device

        self.uniform_planes = nn.ParameterList([
            nn.Parameter(
                torch.randn(self.hash_size, self.input_dim, device=self.device)) for _ in range(self.num_hashtables)])

    def save_uniform_planes(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.uniform_planes, f)

    def load_uniform_planes(self, filename):
        with open(filename, 'rb') as f:
            self.uniform_planes = pickle.load(f)

    def clear_storage(self):
        self.storage_instance.clear()

    def hash_points(self, planes, input_points: torch.Tensor):
        result = (input_points @ planes.T)
        neg_mask = result < 0
        result[neg_mask] = 0
        result[~neg_mask] = 1
        return result

    def _hash(self, planes, input_point: torch.Tensor):
        projections = torch.bmm(planes.view(planes.shape[0], 1, planes.shape[1]), input_point.view(input_point.shape[0], input_point.shape[1], 1)).squeeze(2)
        # There should be a more efficient way to do this (like packing it into a tensor object)
        # but this is okay for now.
        return "".join(['1' if i > 0 else '0' for i in projections])


def hamming_dist(key1, key2):
    return bin(int(key1, base=2) ^ int(key2, base=2))[2:].count('1')
