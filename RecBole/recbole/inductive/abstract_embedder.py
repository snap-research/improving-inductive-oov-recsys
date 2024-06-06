import torch
from typing import Union
from torch import nn

class AbstractInductiveEmbedder(nn.Module):
    """
    Abstract base class for inductive embedders.

    Args:
        user_features (list): List of user features.
        item_features (list): List of item features.
    """

    def __init__(self, user_features, item_features) -> None:
        super().__init__()
        self.user_features = user_features
        self.item_features = item_features
        self.n_new_users = len(user_features)
        self.n_new_items = len(item_features)
        self.training = False

    def set_train(self):
        """
        Sets the model to training mode.
        """
        self.training = True

    def set_eval(self):
        """
        Sets the model to evaluation mode.
        """
        self.training = False

    def embed_user_ids(self, user_ids: torch.LongTensor, model) -> torch.Tensor:
        """
        Embeds user IDs.

        Args:
            user_ids (torch.LongTensor): Tensor of user IDs.
            model: The model to use for embedding.

        Returns:
            torch.Tensor: Tensor of embedded user IDs.
        """
        raise NotImplementedError()

    def embed_item_ids(self, item_ids: torch.LongTensor, model) -> torch.Tensor:
        """
        Embeds item IDs.

        Args:
            item_ids (torch.LongTensor): Tensor of item IDs.
            model: The model to use for embedding.

        Returns:
            torch.Tensor: Tensor of embedded item IDs.
        """
        raise NotImplementedError()

    def map_all_item_embeddings(self, item_embeddings: Union[torch.nn.Parameter, torch.Tensor]) -> torch.Tensor:
        """
        Maps all item embeddings.

        Args:
            item_embeddings (Union[torch.nn.Parameter, torch.Tensor]): Item embeddings.

        Returns:
            torch.Tensor: Mapped item embeddings.
        """
        raise NotImplementedError()
