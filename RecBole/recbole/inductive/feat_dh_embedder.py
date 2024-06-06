import json
from typing import Union
import numpy as np
from recbole.model.abstract_recommender import InductiveContextRecommender, InductiveGeneralRecommender
import os
import torch
from torch import nn
from recbole.inductive.torch_hash import TorchLSHash
import torch.nn.functional as F
from tqdm import tqdm
from .abstract_embedder import AbstractInductiveEmbedder
from csiphash import siphash24
import random
import functorch
from functools import cache
import secrets

class FeatDeepHashEmbedder(AbstractInductiveEmbedder):
    """
    A class that represents a feature-based deep hash embedder for inductive recommendation models.

    Args:
        user_features (pd.DataFrame): The user features dataframe.
        item_features (pd.DataFrame): The item features dataframe.
        n_original_users (int): The number of original users.
        n_original_items (int): The number of original items.
        n_user_oov_buckets (int): The number of out-of-vocabulary (OOV) buckets for users.
        n_item_oov_buckets (int): The number of out-of-vocabulary (OOV) buckets for items.
        embedding_size (int): The size of the embedding vectors.
        device (torch.device): The device to run the model on.
        prime_pad (int): The prime number used for padding.
        num_hashes (int): The number of hashes to use for hashing.
        dhe_layer_size (int): The size of the deep hash embedding (DHE) layer.

    Attributes:
        HASH_KEY_PATH (str): The path to store the hash keys.
        MAX_HASH (int): The maximum hash value.

    Methods:
        get_hash_keys(): Retrieves the hash keys from the file or generates new ones if necessary.
        _hash_id(id: torch.Tensor) -> torch.Tensor: Hashes a single ID using the hash keys.
        _get_hashes(byte_repr: bytes) -> torch.Tensor: Hashes a byte representation using the hash keys.
        _hash_ids(ids: torch.Tensor) -> torch.Tensor: Hashes a batch of IDs using the hash keys.
        _hash_node(nodes: torch.Tensor, lsh: TorchLSHash, feature_mat: torch.Tensor) -> torch.Tensor: Hashes the nodes using the locality-sensitive hashing (LSH) algorithm.
        _hash_users(users: torch.Tensor, feat_lookup_users: torch.Tensor) -> torch.Tensor: Hashes the user IDs and user features.
        _hash_items(items: torch.Tensor, feat_lookup_items: torch.Tensor) -> torch.Tensor: Hashes the item IDs and item features.
        embed_user_ids(old_user_ids: torch.LongTensor, model: Union[InductiveGeneralRecommender, InductiveContextRecommender]) -> torch.Tensor: Embeds the user IDs using the hash-based embedding.
        embed_item_ids(old_item_ids: torch.LongTensor, model: Union[InductiveGeneralRecommender, InductiveContextRecommender]) -> torch.Tensor: Embeds the item IDs using the hash-based embedding.
        embed_all_items(item_embeddings: torch.Tensor, model: InductiveGeneralRecommender): Embeds all items using the hash-based embedding.

    """

    HASH_KEY_PATH = './hash_keys'
    MAX_HASH = 16777216

    def __init__(self, user_features, item_features, n_original_users, n_original_items, n_user_oov_buckets,
                 n_item_oov_buckets, embedding_size, device, prime_pad, num_hashes, dhe_layer_size) -> None:
        """
        Initializes a FeatDeepHashEmbedder instance.

        Args:
            user_features (pd.DataFrame): The user features dataframe.
            item_features (pd.DataFrame): The item features dataframe.
            n_original_users (int): The number of original users.
            n_original_items (int): The number of original items.
            n_user_oov_buckets (int): The number of out-of-vocabulary (OOV) buckets for users.
            n_item_oov_buckets (int): The number of out-of-vocabulary (OOV) buckets for items.
            embedding_size (int): The size of the embedding vectors.
            device (torch.device): The device to run the model on.
            prime_pad (int): The prime number used for padding.
            num_hashes (int): The number of hashes to use for hashing.
            dhe_layer_size (int): The size of the deep hash embedding (DHE) layer.

        """
        super().__init__(user_features, item_features)
        self.n_original_users = n_original_users
        self.n_original_items = n_original_items
        self.n_user_oov_buckets = n_user_oov_buckets
        self.n_item_oov_buckets = n_item_oov_buckets
        self.embedding_size = embedding_size
        self.device = device
        self.prime_pad = prime_pad
        self.num_hashes = num_hashes

        # ... (remaining code)
class FeatDeepHashEmbedder(AbstractInductiveEmbedder):
    HASH_KEY_PATH = './hash_keys'
    MAX_HASH = 16777216

    def __init__(self, user_features, item_features, n_original_users, n_original_items, n_user_oov_buckets,
                 n_item_oov_buckets, embedding_size, device, prime_pad, num_hashes, dhe_layer_size) -> None:
        super().__init__(user_features, item_features)
        self.n_original_users = n_original_users
        self.n_original_items = n_original_items

        self.n_user_oov_buckets = n_user_oov_buckets
        self.n_item_oov_buckets = n_item_oov_buckets
        self.embedding_size = embedding_size
        self.device = device
        self.prime_pad = prime_pad
        self.num_hashes = num_hashes


        user_columns = self.user_features.columns[1:]
        item_columns = self.item_features.columns[1:]
        self.user_feature_mat = torch.hstack([F.normalize(self.user_features[uc].float().view(self.n_new_users, -1), dim=-1) for uc in user_columns]).to(self.device)
        self.item_feature_mat = torch.hstack([F.normalize(self.item_features[ic].float().view(self.n_new_items , -1), dim=-1) for ic in item_columns]).to(self.device)
        self.user_hash_net = nn.Sequential(
            nn.Linear(self.num_hashes + self.user_feature_mat.size(1), dhe_layer_size),
            nn.GELU(),
            nn.Linear(dhe_layer_size, dhe_layer_size),
            nn.GELU(),
            nn.Linear(dhe_layer_size, dhe_layer_size),
            nn.GELU(),
            nn.Linear(dhe_layer_size, self.embedding_size),
            nn.Sigmoid()
        ).to(self.device)
        self.item_hash_net = nn.Sequential(
            nn.Linear(self.num_hashes + self.item_feature_mat.size(1), dhe_layer_size),
            nn.GELU(),
            nn.Linear(dhe_layer_size, dhe_layer_size),
            nn.GELU(),
            nn.Linear(dhe_layer_size, dhe_layer_size),
            nn.GELU(),
            nn.Linear(dhe_layer_size, self.embedding_size),
            nn.Sigmoid()
        ).to(self.device)

        self.hash_keys = self.get_hash_keys()


    def get_hash_keys(self):
        os.makedirs(FeatDeepHashEmbedder.HASH_KEY_PATH, exist_ok=True)
        hash_filename = f'{self.num_hashes}.hashes'
        file_path = os.path.join(FeatDeepHashEmbedder.HASH_KEY_PATH, hash_filename)

        if (os.path.exists(file_path)):
            with open(file_path) as f:
                keys = json.load(f)
                assert(len(keys) == self.num_hashes)
                return [bytes.fromhex(x) for x in keys]

        keys = []
        for _ in range(self.num_hashes):
            key = secrets.token_bytes(16)
            keys.append(key)

        with open(file_path, 'w') as f:
            json.dump([x.hex() for x in keys], f)
        return keys

    def _hash_id(self, id: torch.Tensor) -> torch.Tensor:
        output = torch.empty(self.num_hashes, dtype=torch.double, device=self.device)
        byte_repr = id.item().to_bytes(8, 'little')

        for i, key in enumerate(self.hash_keys):
            output[i] = int.from_bytes(siphash24(key, byte_repr), 'little', signed=False) % FeatDeepHashEmbedder.MAX_HASH
        return output

    @cache
    def _get_hashes(self, byte_repr: bytes) -> torch.Tensor:
        return torch.tensor([int.from_bytes(siphash24(key, byte_repr), 'little', signed=False) % FeatDeepHashEmbedder.MAX_HASH for key in self.hash_keys], dtype=torch.float)

    def _hash_ids(self, ids: torch.Tensor) -> torch.Tensor:
        rows = []

        hfunc = self._get_hashes
        for el in ids:
            byte_repr = el.item().to_bytes(8, 'little')
            rows.append(hfunc(byte_repr))
        return torch.vstack(rows).to(self.device)


    def _hash_node(self, nodes: torch.Tensor, lsh: TorchLSHash, feature_mat: torch.Tensor) -> torch.Tensor:
        assert (lsh.uniform_planes is not None)
        plane = lsh.uniform_planes[0]
        output = lsh.hash_points(plane, feature_mat[nodes]).to(self.device)
        return output

    def _hash_users(self, users: torch.Tensor, feat_lookup_users: torch.Tensor) -> torch.Tensor:
        hashed_values = self._hash_ids(users).float()
        nn_input = torch.hstack((hashed_values, self.user_feature_mat[feat_lookup_users]))
        return self.user_hash_net(nn_input)

    def _hash_items(self, items: torch.Tensor, feat_lookup_items: torch.Tensor) -> torch.Tensor:
        hashed_values = self._hash_ids(items).float()
        nn_input = torch.hstack((hashed_values, self.item_feature_mat[feat_lookup_items]))
        return self.item_hash_net(nn_input)

    def embed_user_ids(self, old_user_ids: torch.LongTensor, model: Union[InductiveGeneralRecommender, InductiveContextRecommender]) -> torch.Tensor:
        if self.training:
            user_ids = old_user_ids.clone()
            user_mask = user_ids >= self.prime_pad
            user_ids[user_mask] = user_ids[user_mask] - self.prime_pad
        else:
            user_ids = old_user_ids
        return self._hash_users(old_user_ids, user_ids)

    def embed_item_ids(self, old_item_ids: torch.LongTensor, model: Union[InductiveGeneralRecommender, InductiveContextRecommender]) -> torch.Tensor:
        if self.training:
            item_ids = old_item_ids.clone()
            item_mask = item_ids >= self.prime_pad
            item_ids[item_mask] = item_ids[item_mask] - self.prime_pad
        else:
            item_ids = old_item_ids
        return self._hash_items(old_item_ids, item_ids)


    def embed_all_items(self, item_embeddings: torch.Tensor, model: InductiveGeneralRecommender):
        raise NotImplementedError()
