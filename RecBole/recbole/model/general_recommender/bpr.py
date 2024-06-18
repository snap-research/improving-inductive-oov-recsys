# -*- coding: utf-8 -*-
# @Time   : 2020/6/25
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn

# UPDATE:
# @Time   : 2020/9/16
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn

r"""
BPR
################################################
Reference:
    Steffen Rendle et al. "BPR: Bayesian Personalized Ranking from Implicit Feedback." in UAI 2009.
"""

from typing import Union
from recbole.inductive.abstract_embedder import AbstractInductiveEmbedder
from recbole.inductive.abstract_mapper import AbstractInductiveMapper
import torch
import torch.nn as nn

from recbole.model.abstract_recommender import GeneralRecommender, InductiveGeneralRecommender
from recbole.model.init import xavier_normal_initialization
from recbole.model.loss import BPRLoss
from recbole.utils import InputType


class BPR(InductiveGeneralRecommender):
    r"""BPR is a basic matrix factorization model that be trained in the pairwise way."""
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset, inductive_mapper: Union[AbstractInductiveMapper, None] = None, inductive_embedder: Union[AbstractInductiveEmbedder, None] = None):
        super(BPR, self).__init__(config, dataset, inductive_mapper, inductive_embedder)

        # load parameters info
        self.embedding_size = config["embedding_size"]

        # define layers and loss
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size)
        self.loss = BPRLoss()

        # parameters initialization
        self.apply(xavier_normal_initialization)

    def get_user_embedding(self, new_user_ids):
        r"""Get a batch of user embedding tensor according to input user's id.

        Args:
            user (torch.LongTensor): The input tensor that contains user's id, shape: [batch_size, ]

        Returns:
            torch.FloatTensor: The embedding tensor of a batch of user, shape: [batch_size, embedding_size]
        """
        if self.inductive_mapper is not None:
            new_user_ids = self.inductive_mapper.map_user_ids(new_user_ids)

        in_vocab_users = new_user_ids < self.n_users
        n_oov_users = new_user_ids.size(0) - in_vocab_users.sum()
        # if self.inductive_embedder is not None:
        #     user_e = self.inductive_embedder.embed_user_ids(new_user_ids, self)
        if n_oov_users > 0:
            # consider placing on CPU first and moving to GPU later to save memory
            user_e = torch.zeros((new_user_ids.shape[0], self.embedding_size), device=self.device)
            in_vocab_user_e = self._user_id_lookup(new_user_ids[in_vocab_users])
            user_e[in_vocab_users] = in_vocab_user_e

            oov_mask = ~in_vocab_users
            if oov_mask.sum() > 0:
                if self.inductive_embedder is not None:
                    user_e[oov_mask] = self.inductive_embedder.embed_user_ids(new_user_ids[oov_mask], self)
                else:
                    user_e[oov_mask] = self.user_oov_buckets(new_user_ids[oov_mask] - self.n_users)
        else:
            user_e = self._user_id_lookup(new_user_ids)
        return user_e

    def _user_id_lookup(self, user_ids):
        return self.user_embedding(user_ids)

    def _item_id_lookup(self, item_ids):
        return self.item_embedding(item_ids)

    def freeze_non_oov_layers(self):
        self.user_embedding.weight.requires_grad = False
        self.item_embedding.weight.requires_grad = False

    def unfreeze_non_oov_layers(self):
        self.user_embedding.weight.requires_grad = True
        self.item_embedding.weight.requires_grad = True

    def get_item_embedding(self, item):
        r"""Get a batch of item embedding tensor according to input item's id.

        Args:
            item (torch.LongTensor): The input tensor that contains item's id, shape: [batch_size, ]

        Returns:
            torch.FloatTensor: The embedding tensor of a batch of item, shape: [batch_size, embedding_size]
        """
        if self.inductive_mapper is not None:
            item = self.inductive_mapper.map_item_ids(item)

        # if self.inductive_embedder is not None:
        #     return self.inductive_embedder.embed_item_ids(item, self)

        in_vocab_items = item < self.n_items
        # n_oov_items = self.n_items - in_vocab_items.sum()
        item_e = torch.zeros((item.shape[0], self.embedding_size), device=self.device)
        oov_mask = ~in_vocab_items

        samp = self.item_embedding.weight.cpu()[item.cpu()[in_vocab_items.cpu()]]
        in_vocab_item_e = self._item_id_lookup(item[in_vocab_items])
        item_e[in_vocab_items] = in_vocab_item_e

        if oov_mask.sum() > 0:
            if self.inductive_embedder is not None:
                item_e[oov_mask] = self.inductive_embedder.embed_item_ids(item[oov_mask], self)
            else:
                item_e[oov_mask] = self.item_oov_buckets(item[oov_mask] - self.n_items)

        # all_item_e = self._item_id_lookup(item)
        return item_e

    def forward(self, user, item):
        user_e = self.get_user_embedding(user)
        item_e = self.get_item_embedding(item)
        return user_e, item_e

    def calculate_loss(self, interaction):
        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        user_e, pos_e = self.forward(user, pos_item)
        neg_e = self.get_item_embedding(neg_item)
        pos_item_score, neg_item_score = torch.mul(user_e, pos_e).sum(dim=1), torch.mul(
            user_e, neg_e
        ).sum(dim=1)
        loss = self.loss(pos_item_score, neg_item_score)
        return loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        user_e, item_e = self.forward(user, item)
        return torch.mul(user_e, item_e).sum(dim=1)

    def ind_full_sort_predict(self, interaction, item_ids):
        user = interaction[self.USER_ID]
        user_e = self.get_user_embedding(user)
        all_item_e = self.get_item_embedding(item_ids)
        score = torch.matmul(user_e, all_item_e.transpose(0, 1))
        return score.view(-1)

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        user_e = self.get_user_embedding(user)
        all_item_e = self.item_embedding.weight
        score = torch.matmul(user_e, all_item_e.transpose(0, 1))
        return score.view(-1)