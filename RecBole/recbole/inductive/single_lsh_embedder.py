from typing import Union
import numpy as np
import torch
import torch.nn.functional as F
from recbole.inductive.torch_hash import TorchLSHash
from recbole.model.abstract_recommender import InductiveContextRecommender, InductiveGeneralRecommender
from .abstract_embedder import AbstractInductiveEmbedder

class SingleLSHInductiveEmbedder(AbstractInductiveEmbedder):
    """
    A class that implements a LSH-based approach that converts the LSH hash of an ID to a integer.
    It then uses the corresponding OOV embedding for that ID.
    This in contrast to the LSHInductiveEmbedder which uses the sum of multiple OOV embeddings.

    Args:
        user_features (pd.DataFrame): The user features.
        item_features (pd.DataFrame): The item features.
        n_original_users (int): The number of original users.
        n_original_items (int): The number of original items.
        n_user_oov_buckets (int): The number of user out-of-vocabulary (OOV) buckets.
        n_item_oov_buckets (int): The number of item out-of-vocabulary (OOV) buckets.
        embedding_size (int): The size of the embeddings.
        device (torch.device): The device to use for computation.
        prime_pad (int): The prime number used for padding.
        normalization_type (str): The type of normalization to apply.

    Attributes:
        n_original_users (int): The number of original users.
        n_original_items (int): The number of original items.
        n_user_oov_buckets (int): The number of user out-of-vocabulary (OOV) buckets.
        n_item_oov_buckets (int): The number of item out-of-vocabulary (OOV) buckets.
        embedding_size (int): The size of the embeddings.
        device (torch.device): The device to use for computation.
        prime_pad (int): The prime number used for padding.
        user_feature_mat (torch.Tensor): The user feature matrix.
        item_feature_mat (torch.Tensor): The item feature matrix.
        user_bits_req (int): The number of bits required for user hashing.
        item_bits_req (int): The number of bits required for item hashing.
        user_lsh (TorchLSHash): The LSH object for user hashing.
        item_lsh (TorchLSHash): The LSH object for item hashing.

    """

    def __init__(self, user_features, item_features, n_original_users, n_original_items, n_user_oov_buckets,
                 n_item_oov_buckets, embedding_size, device, prime_pad, normalization_type) -> None:
        super().__init__(user_features, item_features)
        self.n_original_users = n_original_users
        self.n_original_items = n_original_items
        self.n_user_oov_buckets = n_user_oov_buckets
        self.n_item_oov_buckets = n_item_oov_buckets
        self.embedding_size = embedding_size
        self.device = device
        self.prime_pad = prime_pad

        user_columns = self.user_features.columns[1:]
        item_columns = self.item_features.columns[1:]

        if normalization_type == 'per-feature':
            self.user_feature_mat = torch.hstack([
                F.normalize(self.user_features[uc].float().view(self.n_new_users, -1), dim=-1) for uc in user_columns
            ]).to(self.device)

            self.item_feature_mat = torch.hstack([
                F.normalize(self.item_features[ic].float().view(self.n_new_items, -1), dim=-1) for ic in item_columns
            ]).to(self.device)

        elif normalization_type in ('global', 'none'):
            self.user_feature_mat = torch.hstack(
                [self.user_features[uc].float().view(self.n_new_users, -1) for uc in user_columns]).to(self.device)

            self.item_feature_mat = torch.hstack(
                [self.item_features[ic].float().view(self.n_new_items, -1) for ic in item_columns]).to(self.device)

        else:
            raise ValueError(f'Invalid normalization type: {normalization_type}')

        self.user_bits_req = int(np.ceil(np.log2(self.n_user_oov_buckets)))
        self.item_bits_req = int(np.ceil(np.log2(self.n_item_oov_buckets)))
        self.user_lsh = TorchLSHash(hash_size=self.user_bits_req, input_dim=self.user_feature_mat.size(1), device=self.device)
        self.item_lsh = TorchLSHash(hash_size=self.item_bits_req, input_dim=self.item_feature_mat.size(1), device=self.device)

    def _hash_node(self, nodes: torch.Tensor, lsh: TorchLSHash, feature_mat: torch.Tensor, n_buckets: int) -> torch.Tensor:
        assert (lsh.uniform_planes is not None)
        plane = lsh.uniform_planes[0].data
        output = lsh.hash_points(plane, feature_mat[nodes]).to(self.device)
        node_ids = (2 ** output).sum(axis=1).long() % n_buckets
        return node_ids

    def _hash_users(self, users: torch.Tensor) -> torch.Tensor:
        return self._hash_node(users, self.user_lsh, self.user_feature_mat, self.n_user_oov_buckets)

    def _hash_items(self, items: torch.Tensor) -> torch.Tensor:
        return self._hash_node(items, self.item_lsh, self.item_feature_mat, self.n_item_oov_buckets)

    def embed_user_ids(self, user_ids: torch.LongTensor, model: Union[InductiveGeneralRecommender, InductiveContextRecommender]) -> torch.Tensor:
        if self.training:
            user_mask = user_ids >= self.prime_pad
            user_ids[user_mask] = user_ids[user_mask] - self.prime_pad
        hashed_users = self._hash_users(user_ids)
        user_embeddings = model.user_oov_buckets(hashed_users)
        return user_embeddings

    def embed_item_ids(self, item_ids: torch.LongTensor, model: Union[InductiveGeneralRecommender, InductiveContextRecommender]) -> torch.Tensor:
        if self.training:
            item_mask = item_ids >= self.prime_pad
            item_ids[item_mask] = item_ids[item_mask] - self.prime_pad
        hashed_items = self._hash_items(item_ids)
        item_embeddings = model.item_oov_buckets(hashed_items)
        return item_embeddings

    def embed_all_items(self, item_embeddings: torch.Tensor, model: InductiveGeneralRecommender):
        raise NotImplementedError()
        hashed_items = self._hash_items(list(range(self.n_original_items, self.n_new_items)))
        new_embed = self.oov_item_embed[hashed_items].mean(axis=-1)
        return torch.vstack((item_embeddings, new_embed))
