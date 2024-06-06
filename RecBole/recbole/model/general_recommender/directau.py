import numpy as np
import scipy.sparse as sp

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Union
from recbole.inductive.abstract_embedder import AbstractInductiveEmbedder
from recbole.inductive.abstract_mapper import AbstractInductiveMapper

from recbole.model.abstract_recommender import InductiveGeneralRecommender
from recbole.model.init import xavier_normal_initialization
from recbole.utils import InputType
import copy


class DirectAU(InductiveGeneralRecommender):
    input_type = InputType.POINTWISE

    def __init__(self, config, dataset, inductive_mapper: Union[AbstractInductiveMapper, None] = None, inductive_embedder: Union[AbstractInductiveEmbedder, None] = None):
        super(DirectAU, self).__init__(config, dataset, inductive_mapper, inductive_embedder)

        # load parameters info
        self.embedding_size = config['embedding_size']
        self.gamma = config['gamma']
        self.detach = config['detach'] if 'detach' in config else False 

        # define layers and loss
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size)
        
        # storage variables for full sort evaluation acceleration
        self.restore_user_e = None
        self.restore_item_e = None
        self.other_parameter_name = ['restore_user_e', 'restore_item_e']
        
        # parameters initialization
        self.apply(xavier_normal_initialization)

    def freeze_non_oov_layers(self):
        self.user_embedding.weight.requires_grad = False
        self.item_embedding.weight.requires_grad = False

    def unfreeze_non_oov_layers(self):
        self.user_embedding.weight.requires_grad = True
        self.item_embedding.weight.requires_grad = True

    def get_norm_adj_mat(self):
        # build adj matrix
        A = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.n_users), [1] * inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row + self.n_users, inter_M_t.col), [1] * inter_M_t.nnz)))
        A._update(data_dict)
        # norm adj matrix
        sumArr = (A > 0).sum(axis=1)
        # add epsilon to avoid divide by zero Warning
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        # covert norm_adj matrix to tensor
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor([row, col])
        data = torch.FloatTensor(L.data)
        SparseL = torch.sparse.FloatTensor(i, data, torch.Size(L.shape))
        return SparseL

    def forward(self, user, item):
        user_e = self.get_user_embedding(user)
        item_e = self.get_item_embedding(item)
        return F.normalize(user_e, dim=-1), F.normalize(item_e, dim=-1)

    @staticmethod
    def alignment(x, y, alpha=2):
        return (x - y).norm(p=2, dim=1).pow(alpha).mean()

    @staticmethod
    def uniformity(x, t=2):
        return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()

    def calculate_loss(self, interaction):
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None

        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]

        user_e, item_e = self.forward(user, item)
        align = self.alignment(user_e, item_e)
        uniform = self.gamma * (self.uniformity(user_e) + self.uniformity(item_e)) / 2

        return align + uniform

    def _user_id_lookup(self, user_ids):
        return self.user_embedding(user_ids)

    def _item_id_lookup(self, item_ids):
        return self.item_embedding(item_ids)

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

        # items_clone = item.cpu().clone()
        in_vocab_items = item < self.n_items
        # n_oov_items = self.n_items - in_vocab_items.sum()
        item_e = torch.zeros((item.shape[0], self.embedding_size), device=self.device)
        oov_mask = ~in_vocab_items

        # in_vocab_copy = item[in_vocab_items].clone().cpu()
        # samp = self.item_embedding.weight[item[in_vocab_items]]
        in_vocab_item_e = self._item_id_lookup(item[in_vocab_items])
        item_e[in_vocab_items] = in_vocab_item_e

        if oov_mask.sum() > 0:
            if self.inductive_embedder is not None:
                item_e[oov_mask] = self.inductive_embedder.embed_item_ids(item[oov_mask], self)
            else:
                item_e[oov_mask] = self.item_oov_buckets(item[oov_mask] - self.n_items)

        # all_item_e = self._item_id_lookup(item)
        return item_e

    def predict(self, interaction):
        # interaction_clone = copy.deepcopy(interaction.cpu())
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        # user_e = self.user_embedding(user)
        # item_e = self.item_embedding(item)
        user_e, item_e = self.forward(user, item)
        return torch.mul(user_e, item_e).sum(dim=1)

    def full_sort_predict(self, interaction):
        raise NotImplementedError()
        user = interaction[self.USER_ID]
        if self.restore_user_e is None or self.restore_item_e is None:
            self.restore_user_e, self.restore_item_e = self.get_all_embeddings()
        user_e = self.restore_user_e[user]
        all_item_e = self.restore_item_e
        score = torch.matmul(user_e, all_item_e.transpose(0, 1))
        return score.view(-1)

    def ind_full_sort_predict(self, interaction, item_ids):
        user = interaction[self.USER_ID]
        user_e = self.get_user_embedding(user)
        all_item_e = self.get_item_embedding(item_ids)
        score = torch.matmul(user_e, all_item_e.transpose(0, 1))
        return score.view(-1)

    # def get_all_embeddings(self):
    #     user_embeddings = self.user_embedding.weight
    #     item_embeddings = self.item_embedding.weight
    #     return user_embeddings, item_embeddings


class MFEncoder(nn.Module):
    def __init__(self, user_num, item_num, emb_size):
        super(MFEncoder, self).__init__()
        self.user_embedding = nn.Embedding(user_num, emb_size)
        self.item_embedding = nn.Embedding(item_num, emb_size)

    def forward(self, user_id, item_id):
        u_embed = self.user_embedding(user_id)
        i_embed = self.item_embedding(item_id)
        return u_embed, i_embed

    def get_all_embeddings(self):
        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_embedding.weight
        return user_embeddings, item_embeddings
