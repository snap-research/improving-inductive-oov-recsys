import torch
from typing import Union
from torch import nn

class AbstractInductiveMapper(nn.Module):
    """
    Abstract base class for inductive mappers.

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
        Sets the training mode to True.
        """
        self.training = True

    def set_eval(self):
        """
        Sets the training mode to False.
        """
        self.training = False

    def map_user_ids(self, user_ids: torch.Tensor) -> torch.Tensor:
        """
        Maps user ids to their corresponding embeddings.

        Args:
            user_ids (torch.Tensor): Tensor containing user ids.

        Returns:
            torch.Tensor: Tensor containing user embeddings.
        """
        raise NotImplementedError()

    def map_item_ids(self, item_ids: torch.Tensor) -> torch.Tensor:
        """
        Maps item ids to their corresponding embeddings.

        Args:
            item_ids (torch.Tensor): Tensor containing item ids.

        Returns:
            torch.Tensor: Tensor containing item embeddings.
        """
        raise NotImplementedError()

    def map_all_item_embeddings(self, item_embeddings: Union[torch.nn.Parameter, torch.Tensor]) -> torch.Tensor:
        """
        Maps all item embeddings.

        Args:
            item_embeddings (Union[torch.nn.Parameter, torch.Tensor]): Tensor containing item embeddings.

        Returns:
            torch.Tensor: Tensor containing mapped item embeddings.
        """
        raise NotImplementedError()
