from typing import Literal, Tuple, Union
import torch
import torch.nn.functional as F
from torch import Tensor

class InductiveCollectorFilter():
    """
    This class represents an inductive collector filter.

    Methods:
        apply_score_filter(score_mat: Tensor) -> Tensor:
            Apply score filter to the given score matrix.

        map_users(users: Tensor) -> Tensor:
            Map the given users tensor.

        map_items(items: Tensor) -> Tensor:
            Map the given items tensor.

        map_user_items(user_ids: Tensor, users: Tensor, item_ids: Tensor, items: Tensor, is_ranking_model: bool = False) -> Tuple[Tensor, Tensor]:
            Map the given user and item tensors.

    """

    def __init__(self) -> None:
        pass

    def apply_score_filter(self, score_mat: Tensor) -> Tensor:
        return score_mat

    def map_users(self, users: Tensor) -> Tensor:
        return users

    def map_items(self, items: Tensor) -> Tensor:
        return items

    def map_user_items(self, user_ids: Tensor, users: Tensor, item_ids: Tensor, items: Tensor, is_ranking_model: bool = False) -> Tuple[Tensor, Tensor]:
        return users, items


class UserCollectorFilter(InductiveCollectorFilter):
    """
    A class representing a user collector filter for inductive recommendation.

    This class is responsible for filtering and mapping users in the recommendation process.

    Args:
        target_users (Tensor): The target users to be considered for filtering and mapping.
        n_new_users (int): The number of new users.

    Attributes:
        target_users (Tensor): The target users to be considered for filtering and mapping.
        device: The device on which the target users tensor is located.
        user_mapping (Tensor): A tensor mapping the target users to their corresponding indices.
    """

    def __init__(self, target_users: Tensor, n_new_users: int) -> None:
        super().__init__()
        self.target_users = target_users
        self.device = target_users.device
        # TODO: consider a sparse tensor
        self.user_mapping = torch.full((n_new_users,), -1, dtype=torch.long, device=self.device)
        self.user_mapping[self.target_users] = torch.arange(len(self.target_users))

    def apply_score_filter(self, score_mat: Tensor) -> Tensor:
        """
        Apply score filtering to the score matrix.

        Args:
            score_mat (Tensor): The score matrix to be filtered.

        Returns:
            Tensor: The filtered score matrix.
        """
        return score_mat[self.target_users]

    def map_users(self, users: Tensor) -> Tensor:
        """
        Map the given users to their corresponding indices.

        Args:
            users (Tensor): The users to be mapped.

        Returns:
            Tensor: The mapped users.
        """
        out = self.user_mapping[users]
        # also remove invalid users
        return out[out != -1]


class FastUserCollectorFilter(InductiveCollectorFilter):
    """Optimized version of UserCollectorFilter for old/new user splits.
    """

    def __init__(self, n_old_users: int, return_old=True) -> None:
        super().__init__()
        self.n_old_users = n_old_users
        self.return_old = return_old

    def apply_score_filter(self, score_mat: Tensor) -> Tensor:
        return score_mat[:self.n_old_users] if self.return_old else score_mat[self.n_old_users:]

    def map_users(self, users: Tensor) -> Tensor:
        if self.return_old:
            return users[users < self.n_old_users]
        return users[users >= self.n_old_users] - self.n_old_users


class FastItemCollectorFilter(InductiveCollectorFilter):
    """Optimized version of ItemCollectorFilter for old/new item splits.
    """
    def __init__(self, n_old_items: int, return_old=True) -> None:
        super().__init__()
        self.n_old_items = n_old_items
        self.return_old = return_old

    def apply_score_filter(self, score_mat: Tensor) -> Tensor:
        return score_mat[:, :self.n_old_items] if self.return_old else score_mat[:, self.n_old_items:]

    def map_items(self, items: Tensor) -> Tensor:
        if self.return_old:
            return items[items < self.n_old_items]
        return items[items >= self.n_old_items] - self.n_old_items


class FastUserItemCollectorFilter(InductiveCollectorFilter):
    """
    A class that represents a fast user-item collector filter for inductive recommendation.

    This class extends the `InductiveCollectorFilter` class and provides methods to apply label and score filters,
    as well as map user-item pairs based on certain conditions.

    A value of None means that the corresponding filter is disabled.

    Args:
        n_old_users (int): The number of old users.
        n_old_items (int): The number of old items.
        return_old_users (bool, optional): Whether to return old users. Defaults to None.
        return_old_items (bool, optional): Whether to return old items. Defaults to None.
    """

    def __init__(self, n_old_users: int, n_old_items: int, return_old_users=None, return_old_items=None) -> None:
        super().__init__()
        self.n_old_users = n_old_users
        self.n_old_items = n_old_items
        self.return_old_users = return_old_users
        self.return_old_items = return_old_items
        self.last_users = None

    def apply_label_filter(self, labels: Tensor) -> Tensor:
        if self.overall_mask is None:
            return labels
        return labels[self.overall_mask]

    def apply_score_filter(self, score_mat: Tensor) -> Tensor:
        if self.last_users is None:
            raise ValueError("Must call map_user_items first")

        if self.return_old_users is None and self.return_old_items is None:
            return score_mat

        if score_mat.ndim == 1:
            if self.overall_mask is None:
                return score_mat
            return score_mat[self.overall_mask]

        if self.return_old_items is None:
            return score_mat[self.last_users.unique(sorted=True)]

        if self.return_old_users:
            score_mat[:, self.n_old_items:] = -float('inf')
        else:
            score_mat[:, :self.n_old_items] = -float('inf')

        return score_mat[self.last_users.unique(sorted=True)]

    def _compute_ranking_mask(self, user_ids, item_ids):
        if self.return_old_items is None:
            item_mask = None
        elif self.return_old_items:
            item_mask = item_ids < self.n_old_items
        else:
            item_mask = item_ids >= self.n_old_items

        if self.return_old_users is None:
            user_mask = None
        if self.return_old_users:
            user_mask = user_ids < self.n_old_users
        else:
            user_mask = user_ids >= self.n_old_users

        if user_mask is None and item_mask is None:
            self.overall_mask = None
            return

        if user_mask is None:
            self.overall_mask = item_mask
        elif item_mask is None:
            self.overall_mask = user_mask
        else:
            self.overall_mask = user_mask & item_mask

    def map_user_items(self, user_ids: Tensor, users: Tensor, item_ids: Tensor, items: Tensor, is_ranking_model=False) -> Tuple[Tensor, Tensor]:
        users = users.clone()
        items = items.clone()
        uids = user_ids[users]
        transform_users, transform_items = False, False

        if is_ranking_model:
            self._compute_ranking_mask(user_ids, item_ids)

        if self.return_old_items is None:
            item_mask = None
        elif self.return_old_items:
            item_mask = items < self.n_old_items
        else:
            item_mask = items >= self.n_old_items
            transform_items = True

        if self.return_old_users is None:
            user_mask = None
        if self.return_old_users:
            user_mask = uids < self.n_old_users
            transform_users = True
        else:
            user_mask = uids >= self.n_old_users
            transform_users = True

        if user_mask is None and item_mask is None:
            self.last_users = users
            return users, items

        if user_mask is None:
            overall_mask = item_mask
        elif item_mask is None:
            overall_mask = user_mask
        else:
            overall_mask = user_mask & item_mask

        n_transformed = overall_mask.sum()
        users = users[overall_mask]
        items = items[overall_mask]
        self.last_users = users

        if transform_users and n_transformed >= 1:
            max_id = users.max(dim=-1).values.item()
            mapping = torch.full((max_id + 1,), -1, dtype=torch.long, device=users.device)
            unique_users = users.unique(sorted=True)
            mapping[unique_users] = torch.arange(unique_users.size(0))
            users = mapping[users]

        if transform_items and n_transformed >= 1:
            items = items - self.n_old_items
        return users, items


class ItemCollectorFilter(InductiveCollectorFilter):
    """
    A class representing an item collector filter.

    This filter is used to apply score filtering and map items based on a target set of items.

    Args:
        target_items (Tensor): The target set of items.

    Attributes:
        device: The device on which the target items are located.
        target_items (Tensor): The target set of items.

    Methods:
        apply_score_filter: Applies score filtering to the given score matrix.
        map_items: Maps the given items based on the target set of items.
    """

    def __init__(self, target_items: Tensor) -> None:
        super().__init__()
        self.device = target_items.device
        self.target_items = target_items

    def apply_score_filter(self, score_mat: Tensor) -> Tensor:
        """
        Applies score filtering to the given score matrix.

        This method sets the scores of the target items in the score matrix to -inf.

        Args:
            score_mat (Tensor): The score matrix.

        Returns:
            Tensor: The score matrix with the target items filtered.
        """
        # TODO(willshiao): consider removing items rather than setting to -inf
        out_mat = score_mat.clone()
        out_mat[:, self.target_items] = -float('inf')
        return out_mat

    def map_items(self, items: Tensor) -> Tensor:
        """
        Maps the given items based on the target set of items.

        This method returns only the items that are present in the target set.

        Args:
            items (Tensor): The items to be mapped.

        Returns:
            Tensor: The mapped items.
        """
        out_mask = torch.isin(items, self.target_items, assume_unique=True)
        return items[out_mask]


class MultiCollectorFilter(InductiveCollectorFilter):
    """
    A class representing a multi-filter collector for inductive recommendation.

    This class applies multiple filters to score matrices, user tensors, and item tensors.

    Args:
        filters (list[InductiveCollectorFilter]): A list of filters to be applied.

    Attributes:
        filters (list[InductiveCollectorFilter]): The list of filters to be applied.

    """

    def __init__(self, filters: list[InductiveCollectorFilter]) -> None:
        super().__init__()
        self.filters = filters

    def apply_score_filter(self, score_mat: Tensor) -> Tensor:
        """
        Apply score filters to the score matrix.

        Args:
            score_mat (Tensor): The score matrix to be filtered.

        Returns:
            Tensor: The filtered score matrix.

        """
        for f in self.filters:
            score_mat = f.apply_score_filter(score_mat)
        return score_mat

    def map_users(self, users: Tensor) -> Tensor:
        """
        Map users using the filters.

        Args:
            users (Tensor): The user tensor to be mapped.

        Returns:
            Tensor: The mapped user tensor.

        """
        for f in self.filters:
            users = f.map_users(users)
        return users

    def map_items(self, items: Tensor) -> Tensor:
        """
        Map items using the filters.

        Args:
            items (Tensor): The item tensor to be mapped.

        Returns:
            Tensor: The mapped item tensor.

        """
        for f in self.filters:
            items = f.map_items(items)
        return items
