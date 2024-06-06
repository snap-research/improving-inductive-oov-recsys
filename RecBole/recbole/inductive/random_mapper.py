from typing import Literal
import torch
from recbole.inductive.abstract_mapper import AbstractInductiveMapper
import numpy as np

class RandomOOVInductiveMapper(AbstractInductiveMapper):
    """
    A class that maps user and item IDs to new IDs in an inductive recommendation system with random out-of-vocabulary (OOV) mapping.
    Note that the mappings are deterministic and depend on the hash function used.

    Args:
        user_features (torch.Tensor): The user features.
        item_features (torch.Tensor): The item features.
        n_original_users (int): The number of original users.
        n_original_items (int): The number of original items.
        n_user_oov_buckets (int): The number of user OOV buckets.
        n_item_oov_buckets (int): The number of item OOV buckets.
        embedding_size (int): The size of the embedding.
        device (torch.device): The device to use for computation.
        prime_pad (bool): Whether to use prime padding.
        hash_function (Literal['mod', 'murmur', 'fast', '3round']): The hash function to use.

    Attributes:
        n_new_users (int): The number of new users.
        n_new_items (int): The number of new items.

    Methods:
        set_train(): Sets the model in training mode.
        set_eval(): Sets the model in evaluation mode.
        _fast_int_hash(x: torch.Tensor) -> torch.Tensor: Applies the fast integer hash function to the input tensor.
        _three_round_int_hash(x: torch.Tensor) -> torch.Tensor: Applies the three-round integer hash function to the input tensor.
        _big_64bit_hash(x: torch.Tensor, n_buckets: int) -> torch.Tensor: Applies the 64-bit hash function to the input tensor.
        _hash_ids(oov_user_ids: torch.Tensor, n_buckets: int) -> torch.Tensor: Hashes the OOV user IDs.
        map_user_ids(user_ids: torch.Tensor) -> torch.Tensor: Maps the user IDs to new IDs.
        map_item_ids(item_ids: torch.Tensor) -> torch.Tensor: Maps the item IDs to new IDs.
    """
class RandomOOVInductiveMapper(AbstractInductiveMapper):
    def __init__(self, user_features, item_features, n_original_users, n_original_items, n_user_oov_buckets, n_item_oov_buckets, embedding_size, device, prime_pad, hash_function: Literal['mod', 'murmur', 'fast', '3round']) -> None:
        super().__init__(user_features, item_features)
        self.n_original_users = n_original_users
        self.n_original_items = n_original_items
        self.n_user_oov_buckets = n_user_oov_buckets
        self.n_item_oov_buckets = n_item_oov_buckets
        self.embedding_size = embedding_size
        self.prime_pad = prime_pad
        self.hash_function = hash_function

    def set_train(self):
        super().set_train()
        self.n_new_users = self.n_original_users * 2
        self.n_new_items = self.n_original_items * 2

    def set_eval(self):
        super().set_eval()
        self.n_new_users = len(self.user_features)
        self.n_new_items = len(self.item_features)

    # Original implementation (in C):
    # uint32_t
    # lowbias32(uint32_t x)
    # {
    #     x ^= x >> 16;
    #     x *= 0x7feb352d;
    #     x ^= x >> 15;
    #     x *= 0x846ca68b;
    #     x ^= x >> 16;
    #     return x;
    # }
    # see https://github.com/skeeto/hash-prospector
    def _fast_int_hash(self, x: torch.Tensor):
        x = x.bitwise_xor(x.bitwise_right_shift(16))
        x *= 0x21f0aaad
        x = x.bitwise_xor(x.bitwise_right_shift(15))
        x *= 0xd35a2d97
        x = x.bitwise_xor(x.bitwise_right_shift(15))
        return x

    def _three_round_int_hash(self, x: torch.Tensor):
        x = x.bitwise_xor(x.bitwise_right_shift(17))
        x *= 0xed5ad4bb
        x = x.bitwise_xor(x.bitwise_right_shift(11))
        x *= 0xac4c1b51
        x = x.bitwise_xor(x.bitwise_right_shift(15))
        x *= 0x31848bab
        x = x.bitwise_xor(x.bitwise_right_shift(14))
        return x

    # Original implementation (in C):
    # uint64_t hash(uint64_t x) {
    #     x = (x ^ (x >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    #     x = (x ^ (x >> 27)) * UINT64_C(0x94d049bb133111eb);
    #     x = x ^ (x >> 31);
    #     return x;
    # }
    def _big_64bit_hash(self, x: torch.Tensor, n_buckets: int):
        orig_device = x.device
        x = x.cpu().numpy().astype(np.uint64)
        x = (x ^ (x >> 30)) * int.from_bytes(b'\xbf\x58\x47\x6d\x1c\xe4\xe5\xb9', byteorder='little', signed=False)
        x = (x ^ (x >> 27)) * int.from_bytes(b'\x94\xd0\x49\xbb\x13\x31\x11\xeb', byteorder='little', signed=False)
        x = (x ^ (x >> 31))
        x %= n_buckets
        return torch.tensor(x.astype(np.int64), dtype=torch.int64, device=orig_device)

    def _hash_ids(self, oov_user_ids: torch.Tensor, n_buckets: int):
        if self.hash_function == 'mod':
            return (oov_user_ids) % n_buckets
        elif self.hash_function == 'fast':
            return self._fast_int_hash(oov_user_ids) % n_buckets
        elif self.hash_function == '3round':
            return self._three_round_int_hash(oov_user_ids) % n_buckets
        elif self.hash_function == '64bit':
            return self._big_64bit_hash(oov_user_ids, n_buckets)
        else:
            raise ValueError(f'Unknown hash function {self.hash_function}')

    def map_user_ids(self, user_ids):
        new_ids = torch.zeros(dtype=user_ids.dtype, device=user_ids.device, size=user_ids.shape)
        correct_mask = user_ids < self.n_original_users
        new_ids[correct_mask] = user_ids[correct_mask]
        inv_mask = ~correct_mask
        new_ids[inv_mask] = self._hash_ids(user_ids[inv_mask] - self.n_original_users, self.n_user_oov_buckets) + self.n_original_users
        return new_ids

    def map_item_ids(self, item_ids):
        new_ids = torch.zeros(dtype=item_ids.dtype, device=item_ids.device, size=item_ids.shape)
        correct_mask = item_ids < self.n_original_items
        new_ids[correct_mask] = item_ids[correct_mask]
        inv_mask = ~correct_mask
        new_ids[inv_mask] = self._hash_ids(item_ids[inv_mask] - self.n_original_items, self.n_item_oov_buckets) + self.n_original_items
        return new_ids
