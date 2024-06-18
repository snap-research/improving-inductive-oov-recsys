from typing import Union
from recbole.model.abstract_recommender import InductiveContextRecommender, InductiveGeneralRecommender
import torch
from .abstract_embedder import AbstractInductiveEmbedder

class ZeroEmbedder(AbstractInductiveEmbedder):
    """
    A class that represents a zero embedding strategy for inductive recommendation models.

    This class inherits from the AbstractInductiveEmbedder class and provides methods to embed user and item IDs with zero vectors.

    Args:
        user_features (torch.Tensor): The user features tensor.
        item_features (torch.Tensor): The item features tensor.
        n_original_users (int): The number of original users.
        n_original_items (int): The number of original items.
        embedding_size (int): The size of the embedding vectors.
        device (torch.device): The device to use for tensor operations.

    Attributes:
        zero_vec (torch.Tensor): The zero vector used for embedding.
        n_original_users (int): The number of original users.
        n_original_items (int): The number of original items.

    Methods:
        embed_user_ids(user_ids, model): Embeds user IDs with zero vectors.
        embed_item_ids(item_ids, model): Embeds item IDs with zero vectors.
    """

    def __init__(self, user_features, item_features, n_original_users, n_original_items, embedding_size, device) -> None:
        super().__init__(user_features, item_features)
        self.zero_vec = torch.zeros(embedding_size, device=device)
        self.n_original_users = n_original_users
        self.n_original_items = n_original_items

    def embed_user_ids(self, user_ids: torch.LongTensor, model: Union[InductiveGeneralRecommender, InductiveContextRecommender]) -> torch.Tensor:
        """
        Embeds user IDs with zero vectors.

        Args:
            user_ids (torch.LongTensor): The user IDs to embed.
            model (Union[InductiveGeneralRecommender, InductiveContextRecommender]): The inductive recommendation model.

        Returns:
            torch.Tensor: The embedded user IDs with zero vectors.
        """
        return self.zero_vec.repeat(len(user_ids), 1)

    def embed_item_ids(self, item_ids: torch.LongTensor, model: Union[InductiveGeneralRecommender, InductiveContextRecommender]) -> torch.Tensor:
        """
        Embeds item IDs with zero vectors.

        Args:
            item_ids (torch.LongTensor): The item IDs to embed.
            model (Union[InductiveGeneralRecommender, InductiveContextRecommender]): The inductive recommendation model.

        Returns:
            torch.Tensor: The embedded item IDs with zero vectors.
        """
        return self.zero_vec.repeat(len(item_ids), 1)
