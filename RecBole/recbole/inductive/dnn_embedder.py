from typing import Union
from recbole.model.abstract_recommender import InductiveContextRecommender, InductiveGeneralRecommender
import torch
from torch import nn
import torch.nn.functional as F
from .abstract_embedder import AbstractInductiveEmbedder

class DNNEmbedder(AbstractInductiveEmbedder):
    """
    A class that represents a deep neural network (DNN) embedder for inductive recommendation.

    Args:
        user_features (pd.DataFrame): The user features.
        item_features (pd.DataFrame): The item features.
        n_original_users (int): The number of original users.
        n_original_items (int): The number of original items.
        n_user_oov_buckets (int): The number of out-of-vocabulary (OOV) buckets for users.
        n_item_oov_buckets (int): The number of out-of-vocabulary (OOV) buckets for items.
        embedding_size (int): The size of the embedding.
        device (torch.device): The device to use for computation.
        prime_pad (int): The prime padding value.
        dhe_layer_size (int): The size of the DHE (Deep Hashing Embedding) layer.

    Attributes:
        n_original_users (int): The number of original users.
        n_original_items (int): The number of original items.
        n_user_oov_buckets (int): The number of out-of-vocabulary (OOV) buckets for users.
        n_item_oov_buckets (int): The number of out-of-vocabulary (OOV) buckets for items.
        embedding_size (int): The size of the embedding.
        device (torch.device): The device to use for computation.
        prime_pad (int): The prime padding value.
        user_feature_mat (torch.Tensor): The user feature matrix.
        item_feature_mat (torch.Tensor): The item feature matrix.
        user_hash_net (nn.Sequential): The user hash network.
        item_hash_net (nn.Sequential): The item hash network.

    Methods:
        _hash_users(users, feat_lookup_users): Hashes the user features.
        _hash_items(items, feat_lookup_items): Hashes the item features.
        embed_user_ids(old_user_ids, model): Embeds the user IDs.
        embed_item_ids(old_item_ids, model): Embeds the item IDs.
        embed_all_items(item_embeddings, model): Embeds all items.

    Raises:
        NotImplementedError: If the method `embed_all_items` is not implemented.

    """

    def __init__(self, user_features, item_features, n_original_users, n_original_items, n_user_oov_buckets,
                 n_item_oov_buckets, embedding_size, device, prime_pad, dhe_layer_size) -> None:
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
        self.user_feature_mat = torch.hstack([F.normalize(self.user_features[uc].float().view(self.n_new_users, -1), dim=-1) for uc in user_columns]).to(self.device)
        self.item_feature_mat = torch.hstack([F.normalize(self.item_features[ic].float().view(self.n_new_items , -1), dim=-1) for ic in item_columns]).to(self.device)
        self.user_hash_net = nn.Sequential(
            nn.Linear(self.user_feature_mat.size(1), dhe_layer_size),
            nn.GELU(),
            nn.Linear(dhe_layer_size, dhe_layer_size),
            nn.GELU(),
            nn.Linear(dhe_layer_size, dhe_layer_size),
            nn.GELU(),
            nn.Linear(dhe_layer_size, self.embedding_size),
            nn.Sigmoid()
        ).to(self.device)
        self.item_hash_net = nn.Sequential(
            nn.Linear(self.item_feature_mat.size(1), dhe_layer_size),
            nn.GELU(),
            nn.Linear(dhe_layer_size, dhe_layer_size),
            nn.GELU(),
            nn.Linear(dhe_layer_size, dhe_layer_size),
            nn.GELU(),
            nn.Linear(dhe_layer_size, self.embedding_size),
            nn.Sigmoid()
        ).to(self.device)

    def _hash_users(self, users: torch.Tensor, feat_lookup_users: torch.Tensor) -> torch.Tensor:
        return self.user_hash_net(self.user_feature_mat[feat_lookup_users])

    def _hash_items(self, items: torch.Tensor, feat_lookup_items: torch.Tensor) -> torch.Tensor:
        return self.item_hash_net(self.item_feature_mat[feat_lookup_items])

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
