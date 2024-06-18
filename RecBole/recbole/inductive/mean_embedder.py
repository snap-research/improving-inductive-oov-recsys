from typing import Optional, Union
from recbole.model.abstract_recommender import InductiveContextRecommender, InductiveGeneralRecommender
from recbole.model.context_aware_recommender.dcnv2 import DCNV2
from recbole.model.context_aware_recommender.widedeep import WideDeep
from recbole.model.context_aware_recommender.xdeepfm import xDeepFM
from recbole.model.general_recommender.bpr import BPR
from recbole.model.general_recommender.directau import DirectAU
from recbole.model.layers import InductiveFMFirstOrderLinear
import torch
from .abstract_embedder import AbstractInductiveEmbedder

class MeanEmbedder(AbstractInductiveEmbedder):
    """
    A class that calculates the mean embeddings for user and item IDs.

    Args:
        user_features (torch.Tensor): The user features.
        item_features (torch.Tensor): The item features.
        n_original_users (int): The number of original users.
        n_original_items (int): The number of original items.
        n_user_oov_buckets (int): The number of user out-of-vocabulary (OOV) buckets.
        n_item_oov_buckets (int): The number of item out-of-vocabulary (OOV) buckets.
        embedding_size (int): The size of the embeddings.
        device (torch.device): The device to use for calculations.

    Attributes:
        user_feat_mean (Optional[torch.Tensor]): The mean user feature tensor.
        item_feat_mean (Optional[torch.Tensor]): The mean item feature tensor.
        n_original_users (int): The number of original users.
        n_original_items (int): The number of original items.
    """

    def __init__(self, user_features, item_features, n_original_users, n_original_items, n_user_oov_buckets,
                 n_item_oov_buckets, embedding_size, device) -> None:
        super().__init__(user_features, item_features)
        self.user_feat_mean: Optional[torch.Tensor] = None
        self.item_feat_mean: Optional[torch.Tensor] = None
        self.n_original_users = n_original_users
        self.n_original_items = n_original_items

    @torch.no_grad()
    def embed_user_ids(self, user_ids: torch.LongTensor, model: Union[InductiveGeneralRecommender, InductiveContextRecommender]) -> torch.Tensor:
        """
        Embeds user IDs using the mean user feature tensor.

        Args:
            user_ids (torch.LongTensor): The user IDs to embed.
            model (Union[InductiveGeneralRecommender, InductiveContextRecommender]): The model used for embedding.

        Returns:
            torch.Tensor: The embedded user IDs.
        """
        if isinstance(model, (BPR, DirectAU)):
            if self.user_feat_mean is None:
                self.user_feat_mean = torch.mean(model.user_embedding.weight, dim=0)
            return self.user_feat_mean.repeat(len(user_ids), 1)
        elif isinstance(model, (DCNV2, WideDeep, xDeepFM, InductiveFMFirstOrderLinear)):
            if self.user_feat_mean is None:
                self.user_feat_mean = model.token_embedding_table.embedding.weight[model.token_field_offsets[0]:model.token_field_offsets[1]].mean(dim=0)
            return self.user_feat_mean.repeat(len(user_ids), 1)
        raise ValueError('Invalid model type for mean embedder')

    @torch.no_grad()
    def embed_item_ids(self, item_ids: torch.LongTensor, model: Union[InductiveGeneralRecommender, InductiveContextRecommender]) -> torch.Tensor:
        """
        Embeds item IDs using the mean item feature tensor.

        Args:
            item_ids (torch.LongTensor): The item IDs to embed.
            model (Union[InductiveGeneralRecommender, InductiveContextRecommender]): The model used for embedding.

        Returns:
            torch.Tensor: The embedded item IDs.
        """
        if isinstance(model, (BPR, DirectAU)):
            if self.item_feat_mean is None:
                self.item_feat_mean = torch.mean(model.item_embedding.weight, dim=0)
            return self.item_feat_mean.repeat(len(item_ids), 1)
        elif isinstance(model, (DCNV2, WideDeep, xDeepFM, InductiveFMFirstOrderLinear)):
            if self.item_feat_mean is None:
                if len(model.token_field_offsets) == 2:
                    token_weight = model.token_embedding_table.embedding.weight[model.token_field_offsets[1]:]
                else:
                    token_weight = model.token_embedding_table.embedding.weight[model.token_field_offsets[1]:model.token_field_offsets[2]]
                self.item_feat_mean = token_weight.mean(dim=0)
            return self.item_feat_mean.repeat(len(item_ids), 1)
        raise ValueError('Invalid model type for mean embedder')
