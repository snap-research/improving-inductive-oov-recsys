from copy import deepcopy
from typing import Union
import numpy as np
from recbole.model.abstract_recommender import InductiveContextRecommender, InductiveGeneralRecommender
from recbole.model.context_aware_recommender.dcnv2 import DCNV2
from recbole.model.context_aware_recommender.widedeep import WideDeep
from recbole.model.context_aware_recommender.xdeepfm import xDeepFM
from recbole.model.general_recommender.bpr import BPR
from recbole.model.general_recommender.directau import DirectAU
from recbole.model.layers import InductiveFMFirstOrderLinear
import torch
from recbole.inductive.torch_hash import TorchLSHash
import torch.nn.functional as F
from .abstract_embedder import AbstractInductiveEmbedder
import scann


class KNNInductiveEmbedder(AbstractInductiveEmbedder):
    """
    KNNInductiveEmbedder is a class that implements the inductive embedding using K-nearest neighbors (KNN) approach.

    Args:
        user_features (pd.DataFrame): The user features dataframe.
        item_features (pd.DataFrame): The item features dataframe.
        n_original_users (int): The number of original users.
        n_original_items (int): The number of original items.
        n_user_oov_buckets (int): The number of out-of-vocabulary (OOV) buckets for users.
        n_item_oov_buckets (int): The number of out-of-vocabulary (OOV) buckets for items.
        embedding_size (int): The size of the embedding.
        device (torch.device): The device to use for computation.
        prime_pad (int): The prime padding value.
        n_neighbors (int, optional): The number of nearest neighbors to consider. Defaults to 2.

    Attributes:
        n_original_users (int): The number of original users.
        n_original_items (int): The number of original items.
        n_user_oov_buckets (int): The number of out-of-vocabulary (OOV) buckets for users.
        n_item_oov_buckets (int): The number of out-of-vocabulary (OOV) buckets for items.
        embedding_size (int): The size of the embedding.
        device (torch.device): The device to use for computation.
        n_neighbors (int): The number of nearest neighbors to consider.
        prime_pad (int): The prime padding value.
        user_feature_mat (np.ndarray): The normalized user feature matrix.
        item_feature_mat (np.ndarray): The normalized item feature matrix.
        user_searcher (scann.scann_ops_pybind.scann): The user searcher object.
        item_searcher (scann.scann_ops_pybind.scann): The item searcher object.

    """

    def __init__(self,
                 user_features,
                 item_features,
                 n_original_users,
                 n_original_items,
                 n_user_oov_buckets,
                 n_item_oov_buckets,
                 embedding_size,
                 device,
                 prime_pad,
                 n_neighbors=2) -> None:
        super().__init__(user_features, item_features)
        self.n_original_users = n_original_users
        self.n_original_items = n_original_items
        self.n_user_oov_buckets = n_user_oov_buckets
        self.n_item_oov_buckets = n_item_oov_buckets
        self.embedding_size = embedding_size
        self.device = device
        self.n_neighbors = n_neighbors
        self.prime_pad = prime_pad

        user_columns = self.user_features.columns[1:]
        item_columns = self.item_features.columns[1:]
        self.user_feature_mat = F.normalize(
            torch.hstack([
                F.normalize(self.user_features[uc].float().view(self.n_new_users, -1), dim=-1) for uc in user_columns
            ])).cpu().numpy()
        self.item_feature_mat = F.normalize(
            torch.hstack([
                F.normalize(self.item_features[ic].float().view(self.n_new_items, -1), dim=-1) for ic in item_columns
            ])).cpu().numpy()

        # We have currently fixed these hyperparameters for simplicity.
        # In the future, these should be moved to a configuration file.
        self.user_searcher = scann.scann_ops_pybind.builder(self.user_feature_mat[:n_original_users], 10, "dot_product") \
            .tree(num_leaves=2000, num_leaves_to_search=100, training_sample_size=250000) \
            .score_ah(2, anisotropic_quantization_threshold=0.2) \
            .reorder(100) \
            .build()
        self.item_searcher = scann.scann_ops_pybind.builder(self.item_feature_mat[:n_original_items], 10, "dot_product") \
            .tree(num_leaves=2000, num_leaves_to_search=100, training_sample_size=250000) \
            .score_ah(2, anisotropic_quantization_threshold=0.2) \
            .reorder(100) \
            .build()

    def __deepcopy__(self, memo):
        return KNNInductiveEmbedder(deepcopy(self.user_features, memo), deepcopy(self.item_features, memo),
                                    self.n_original_users, self.n_original_items, self.n_user_oov_buckets,
                                    self.n_item_oov_buckets, self.embedding_size, self.device, self.prime_pad)

    def _hash_node(self, nodes: torch.Tensor, searcher, feature_mat: torch.Tensor) -> torch.Tensor:
        target_idxs, _ = searcher.search_batched(feature_mat[nodes.cpu().numpy()], final_num_neighbors=self.n_neighbors)
        return torch.tensor(target_idxs.astype(np.int64)).to(self.device)

    def _hash_users(self, users: torch.Tensor) -> torch.Tensor:
        return self._hash_node(users, self.user_searcher, self.user_feature_mat)

    def _hash_items(self, items: torch.Tensor) -> torch.Tensor:
        return self._hash_node(items, self.item_searcher, self.item_feature_mat)

    def embed_user_ids(self, user_ids: torch.LongTensor, model: Union[InductiveGeneralRecommender,
                                                                      InductiveContextRecommender]) -> torch.Tensor:
        if self.training:
            user_mask = user_ids >= self.prime_pad
            user_ids[user_mask] = user_ids[user_mask] - self.prime_pad
        hashed_users = self._hash_users(user_ids)

        if isinstance(model, (BPR, DirectAU)):
            weight_mat = model.user_embedding.weight
        elif isinstance(model, (DCNV2, WideDeep, xDeepFM, InductiveFMFirstOrderLinear)):
            weight_mat = model.token_embedding_table.embedding.weight[model.token_field_offsets[0]:model.
                                                                      token_field_offsets[1]]
        else:
            raise ValueError('Unknown model type')

        select_embeddings = weight_mat[hashed_users.ravel()]
        return torch.vstack([x.mean(dim=0) for x in select_embeddings.split(2)])

    def embed_item_ids(self, item_ids: torch.LongTensor, model: Union[InductiveGeneralRecommender,
                                                                      InductiveContextRecommender]) -> torch.Tensor:
        if self.training:
            item_mask = item_ids >= self.prime_pad
            item_ids[item_mask] = item_ids[item_mask] - self.prime_pad
        hashed_items = self._hash_items(item_ids)

        if isinstance(model, (BPR, DirectAU)):
            weight_mat = model.item_embedding.weight
        elif isinstance(model, (DCNV2, WideDeep, xDeepFM, InductiveFMFirstOrderLinear)):
            if len(model.token_field_offsets) == 2:
                weight_mat = model.token_embedding_table.embedding.weight[model.token_field_offsets[1]:]
            else:
                weight_mat = model.token_embedding_table.embedding.weight[model.token_field_offsets[1]:model.
                                                                          token_field_offsets[2]]
        else:
            raise ValueError('Unknown model type')

        select_embeddings = weight_mat[hashed_items.ravel()]
        return torch.vstack([x.mean(dim=0) for x in select_embeddings.split(2)])

    def embed_all_items(self, item_embeddings: torch.Tensor, model: InductiveGeneralRecommender):
        raise NotImplementedError()
