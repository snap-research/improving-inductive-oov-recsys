from typing import Union
import numpy as np
from recbole.model.abstract_recommender import InductiveContextRecommender, InductiveGeneralRecommender
import torch
# from pyLSHash import LSHash
from recbole.inductive.torch_hash import TorchLSHash
import torch.nn.functional as F
from .abstract_embedder import AbstractInductiveEmbedder


class LSHInductiveEmbedder(AbstractInductiveEmbedder):
    """
    Class for performing inductive embedding using Locality Sensitive Hashing (LSH).

    Args:
        user_features (DataFrame): DataFrame containing user features.
        item_features (DataFrame): DataFrame containing item features.
        n_original_users (int): Number of original users.
        n_original_items (int): Number of original items.
        n_user_oov_buckets (int): Number of out-of-vocabulary buckets for users.
        n_item_oov_buckets (int): Number of out-of-vocabulary buckets for items.
        embedding_size (int): Size of the embedding vectors.
        device (str): Device to use for computation (e.g., 'cpu', 'cuda').
        prime_pad (int): Padding prime number to add to inductive user and item IDs.

    Attributes:
        n_original_users (int): Number of original users.
        n_original_items (int): Number of original items.
        n_user_oov_buckets (int): Number of out-of-vocabulary buckets for users.
        n_item_oov_buckets (int): Number of out-of-vocabulary buckets for items.
        embedding_size (int): Size of the embedding vectors.
        device (str): Device to use for computation (e.g., 'cpu', 'cuda').
        prime_pad (int): Padding prime number to add to inductive user and item IDs.
        user_feature_mat (torch.Tensor): Tensor containing user feature matrix.
        item_feature_mat (torch.Tensor): Tensor containing item feature matrix.
        user_lsh (TorchLSHash): LSH object for hashing user features.
        item_lsh (TorchLSHash): LSH object for hashing item features.

    Methods:
        _hash_node(nodes, lsh, feature_mat): Hashes the given nodes using the provided LSH object and feature matrix.
        _hash_users(users): Hashes the given user nodes using the user LSH object and user feature matrix.
        _hash_items(items): Hashes the given item nodes using the item LSH object and item feature matrix.
        embed_user_ids(user_ids, model): Embeds user IDs using the provided model and user LSH object.
        embed_item_ids(item_ids, model): Embeds item IDs using the provided model and item LSH object.
        embed_all_items(item_embeddings, model): Embeds all items using the provided item embeddings and model.
    """

    def __init__(self, user_features, item_features, n_original_users, n_original_items, n_user_oov_buckets,
                 n_item_oov_buckets, embedding_size, device, prime_pad, normalization_type, feature_cache) -> None:
        """
        Initialize the LSHEmbedder object.

        Args:
            user_features (DataFrame): DataFrame containing user features.
            item_features (DataFrame): DataFrame containing item features.
            n_original_users (int): Number of original users.
            n_original_items (int): Number of original items.
            n_user_oov_buckets (int): Number of out-of-vocabulary buckets for users.
            n_item_oov_buckets (int): Number of out-of-vocabulary buckets for items.
            embedding_size (int): Size of the embedding vectors.
            device (str): Device to use for computation (e.g., 'cpu', 'cuda').
            prime_pad (int): Padding prime number to add to inductive user and item IDs.
            normalization_type (str): Type of normalization to apply to the feature matrices.

        Returns:
            None
        """
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

        if feature_cache.has_cached():
            self.user_feature_mat, self.item_feature_mat = feature_cache.get_cached()
        else:
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

            if normalization_type == 'global':
                self.user_feature_mat = F.normalize(self.user_feature_mat, dim=-1)
                self.item_feature_mat = F.normalize(self.item_feature_mat, dim=-1)

            feature_cache.add_to_cache(self.user_feature_mat, self.item_feature_mat)

        self.user_lsh = TorchLSHash(hash_size=n_user_oov_buckets,
                                    input_dim=self.user_feature_mat.size(1),
                                    device=self.device)

        self.item_lsh = TorchLSHash(hash_size=n_item_oov_buckets,
                                    input_dim=self.item_feature_mat.size(1),
                                    device=self.device)

    def _hash_node(self, nodes: torch.Tensor, lsh: TorchLSHash, feature_mat: torch.Tensor) -> torch.Tensor:
        """
        Hashes the given nodes using the provided LSH object and feature matrix.

        Args:
            nodes (torch.Tensor): The nodes to be hashed.
            lsh (TorchLSHash): The LSH object used for hashing.
            feature_mat (torch.Tensor): The feature matrix containing the node features.

        Returns:
            torch.Tensor: The hashed representation of the nodes.
        """
        assert (lsh.uniform_planes is not None)
        plane = lsh.uniform_planes[0].data
        output = lsh.hash_points(plane, feature_mat[nodes]).to(self.device)
        return output

    def _hash_users(self, users: torch.Tensor) -> torch.Tensor:
        """ Hashes the given user nodes using the user LSH object and user feature matrix. """
        return self._hash_node(users, self.user_lsh, self.user_feature_mat)

    def _hash_items(self, items: torch.Tensor) -> torch.Tensor:
        """ Hashes the given item nodes using the item LSH object and item feature matrix. """
        return self._hash_node(items, self.item_lsh, self.item_feature_mat)

    def embed_user_ids(self, user_ids: torch.LongTensor, model: Union[InductiveGeneralRecommender,
                                                                      InductiveContextRecommender]) -> torch.Tensor:
        """
        Embeds user IDs using the provided model and user LSH object.

        Args:
            user_ids (torch.LongTensor): The user IDs to be embedded.
            model (Union[InductiveGeneralRecommender, InductiveContextRecommender]): The model used for embedding.

        Returns:
            torch.Tensor: The embedded user IDs.
        """
        if self.training:
            user_mask = user_ids >= self.prime_pad
            user_ids[user_mask] = user_ids[user_mask] - self.prime_pad
        hashed_users = self._hash_users(user_ids)
        user_embeddings = model.user_oov_buckets.weight
        new_embed = (hashed_users @ user_embeddings) / hashed_users.sum(dim=1).view(-1, 1)
        return new_embed

    def embed_item_ids(self, item_ids: torch.LongTensor, model: Union[InductiveGeneralRecommender,
                                                                      InductiveContextRecommender]) -> torch.Tensor:
        """
        Embeds item IDs using the provided model and item LSH object.

        Args:
            item_ids (torch.LongTensor): The item IDs to be embedded.
            model (Union[InductiveGeneralRecommender, InductiveContextRecommender]): The model used for embedding.

        Returns:
            torch.Tensor: The embedded item IDs.
        """
        if self.training:
            item_mask = item_ids >= self.prime_pad
            item_ids[item_mask] = item_ids[item_mask] - self.prime_pad
        hashed_items = self._hash_items(item_ids)
        item_embeddings = model.item_oov_buckets.weight
        new_embed = (hashed_items @ item_embeddings) / hashed_items.sum(dim=1).view(-1, 1)
        return new_embed

    def embed_all_items(self, item_embeddings: torch.Tensor, model: InductiveGeneralRecommender):
        """
        Embeds all items using the provided item embeddings and model.

        Args:
            item_embeddings (torch.Tensor): The item embeddings.
            model (InductiveGeneralRecommender): The model used for embedding.

        Returns:
            torch.Tensor: The embedded item IDs.
        """
        raise NotImplementedError()
