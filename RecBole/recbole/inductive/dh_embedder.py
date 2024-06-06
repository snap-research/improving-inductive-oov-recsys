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

class DeepHashEmbedder(AbstractInductiveEmbedder):
    """
    DeepHashEmbedder is a class that implements an inductive embedding method using deep hashing.

    Args:
        user_features (pd.DataFrame): The user features dataframe.
        item_features (pd.DataFrame): The item features dataframe.
        n_original_users (int): The number of original users.
        n_original_items (int): The number of original items.
        n_user_oov_buckets (int): The number of out-of-vocabulary (OOV) buckets for users.
        n_item_oov_buckets (int): The number of out-of-vocabulary (OOV) buckets for items.
        embedding_size (int): The size of the embedding vectors.
        device (torch.device): The device to use for computation.
        prime_pad (bool): Whether to use prime padding.
        num_hashes (int): The number of hashes to use.

    Attributes:
        HASH_KEY_PATH (str): The path to store the hash keys.
        MAX_HASH (int): The maximum hash value.

    Methods:
        get_hash_keys(): Retrieves the hash keys from the file or generates new ones if necessary.
        _hash_id(id: torch.Tensor) -> torch.Tensor: Hashes a single ID.
        _get_hashes(byte_repr: bytes) -> torch.Tensor: Computes the hashes for a byte representation.
        _hash_ids(ids: torch.Tensor) -> torch.Tensor: Hashes a batch of IDs.
        _hash_node(nodes: torch.Tensor, lsh: TorchLSHash, feature_mat: torch.Tensor) -> torch.Tensor: Hashes a batch of nodes.
        _hash_users(users: torch.Tensor) -> torch.Tensor: Hashes a batch of user IDs.
        _hash_items(items: torch.Tensor) -> torch.Tensor: Hashes a batch of item IDs.
        embed_user_ids(user_ids: torch.LongTensor, model: Union[InductiveGeneralRecommender, InductiveContextRecommender]) -> torch.Tensor: Embeds user IDs.
        embed_item_ids(item_ids: torch.LongTensor, model: Union[InductiveGeneralRecommender, InductiveContextRecommender]) -> torch.Tensor: Embeds item IDs.
        embed_all_items(item_embeddings: torch.Tensor, model: InductiveGeneralRecommender): Embeds all item embeddings.

    """

    HASH_KEY_PATH = './hash_keys'
    MAX_HASH = 16777216

    def __init__(self, user_features, item_features, n_original_users, n_original_items, n_user_oov_buckets,
                 n_item_oov_buckets, embedding_size, device, prime_pad, num_hashes) -> None:
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
        self.user_hash_net = nn.Sequential(
            nn.Linear(self.num_hashes, 512),
            nn.GELU(),
            nn.Linear(512, 512),
            nn.GELU(),
            nn.Linear(512, 512),
            nn.GELU(),
            nn.Linear(512, self.embedding_size),
            nn.Sigmoid()
        ).to(self.device)
        self.item_hash_net = nn.Sequential(
            nn.Linear(self.num_hashes, 512),
            nn.GELU(),
            nn.Linear(512, 512),
            nn.GELU(),
            nn.Linear(512, 512),
            nn.GELU(),
            nn.Linear(512, self.embedding_size),
            nn.Sigmoid()
        ).to(self.device)

        self.user_feature_mat = torch.hstack([F.normalize(self.user_features[uc].float().view(self.n_new_users, -1), dim=-1) for uc in user_columns]).to(self.device)
        self.item_feature_mat = torch.hstack([F.normalize(self.item_features[ic].float().view(self.n_new_items , -1), dim=-1) for ic in item_columns]).to(self.device)
        self.hash_keys = self.get_hash_keys()

    def get_hash_keys(self):
        """
        Retrieves the hash keys from the file or generates new ones if necessary.

        Returns:
            List[bytes]: The hash keys.

        """
        os.makedirs(DeepHashEmbedder.HASH_KEY_PATH, exist_ok=True)
        hash_filename = f'{self.num_hashes}.hashes'
        file_path = os.path.join(DeepHashEmbedder.HASH_KEY_PATH, hash_filename)

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
        """
        Hashes a single ID.

        Args:
            id (torch.Tensor): The ID to hash.

        Returns:
            torch.Tensor: The hashed ID.

        """
        output = torch.empty(self.num_hashes, dtype=torch.double, device=self.device)
        byte_repr = id.item().to_bytes(8, 'little')

        for i, key in enumerate(self.hash_keys):
            output[i] = int.from_bytes(siphash24(key, byte_repr), 'little', signed=False) % DeepHashEmbedder.MAX_HASH
        return output

    @cache
    def _get_hashes(self, byte_repr: bytes) -> torch.Tensor:
        """
        Computes the hashes for a byte representation.

        Args:
            byte_repr (bytes): The byte representation.

        Returns:
            torch.Tensor: The computed hashes.

        """
        return torch.tensor([int.from_bytes(siphash24(key, byte_repr), 'little', signed=False) % DeepHashEmbedder.MAX_HASH for key in self.hash_keys], dtype=torch.float)

    def _hash_ids(self, ids: torch.Tensor) -> torch.Tensor:
        """
        Hashes a batch of IDs.

        Args:
            ids (torch.Tensor): The IDs to hash.

        Returns:
            torch.Tensor: The hashed IDs.

        """
        rows = []
        hfunc = self._get_hashes
        for el in ids:
            byte_repr = el.item().to_bytes(8, 'little')
            rows.append(hfunc(byte_repr))
        return torch.vstack(rows).to(self.device)


    def _hash_node(self, nodes: torch.Tensor, lsh: TorchLSHash, feature_mat: torch.Tensor) -> torch.Tensor:
        """
        Hashes a batch of nodes.

        Args:
            nodes (torch.Tensor): The nodes to hash.
            lsh (TorchLSHash): The locality-sensitive hash object.
            feature_mat (torch.Tensor): The feature matrix.

        Returns:
            torch.Tensor: The hashed nodes.

        """
        assert (lsh.uniform_planes is not None)
        plane = lsh.uniform_planes[0]
        output = lsh.hash_points(plane, feature_mat[nodes]).to(self.device)
        return output

    def _hash_users(self, users: torch.Tensor) -> torch.Tensor:
        """
        Hashes a batch of user IDs.

        Args:
            users (torch.Tensor): The user IDs to hash.

        Returns:
            torch.Tensor: The hashed user IDs.

        """
        hashed_values = self._hash_ids(users).float()
        return self.user_hash_net(hashed_values)

    def _hash_items(self, items: torch.Tensor) -> torch.Tensor:
        """
        Hashes a batch of item IDs.

        Args:
            items (torch.Tensor): The item IDs to hash.

        Returns:
            torch.Tensor: The hashed item IDs.

        """
        hashed_values = self._hash_ids(items).float()
        return self.item_hash_net(hashed_values)

    def embed_user_ids(self, user_ids: torch.LongTensor, model: Union[InductiveGeneralRecommender, InductiveContextRecommender]) -> torch.Tensor:
        """
        Embeds user IDs.

        Args:
            user_ids (torch.LongTensor): The user IDs to embed.
            model (Union[InductiveGeneralRecommender, InductiveContextRecommender]): The recommender model.

        Returns:
            torch.Tensor: The embedded user IDs.

        """
        return self._hash_users(user_ids)

    def embed_item_ids(self, item_ids: torch.LongTensor, model: Union[InductiveGeneralRecommender, InductiveContextRecommender]) -> torch.Tensor:
        """
        Embeds item IDs.

        Args:
            item_ids (torch.LongTensor): The item IDs to embed.
            model (Union[InductiveGeneralRecommender, InductiveContextRecommender]): The recommender model.

        Returns:
            torch.Tensor: The embedded item IDs.

        """
        return self._hash_items(item_ids)

    def embed_all_items(self, item_embeddings: torch.Tensor, model: InductiveGeneralRecommender):
        """
        Embeds all item embeddings.

        Args:
            item_embeddings (torch.Tensor): The item embeddings.
            model (InductiveGeneralRecommender): The recommender model.

        Raises:
            NotImplementedError: This method is not implemented.

        """
        raise NotImplementedError()
