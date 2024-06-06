from recbole.data.dataset import Dataset
from typing import Dict, Optional, Union, Literal
import numpy as np
from recbole.utils.enum_type import FeatureType
import torch

class InductiveDataset(Dataset):
    """
    Dataset class for inductive recommendation models.

    Args:
        config (dict): The configuration dictionary.
        removal_setting (Optional[Literal['remove_old', 'remove_new']]): The removal setting for filtering data. Defaults to None.

    Attributes:
        remove_old (bool): Flag indicating whether to remove old data.
        remove_new (bool): Flag indicating whether to remove new data.
        id2id_mapping (Optional[Dict[str, Dict[int, int]]]): Mapping of IDs from the original dataset to the inductive dataset.
        train_ufeatures (Optional[Dict[str, torch.Tensor]]): User features for training.
        train_ifeatures (Optional[Dict[str, torch.Tensor]]): Item features for training.
        orig_dataset (Optional[Dataset]): The original dataset.
        USER_ID (str): The field name for user IDs.
        ITEM_ID (str): The field name for item IDs.
        feat_name_list (List[str]): List of feature names.
        field2id_token (Dict[str, Dict[int, int]]): Mapping of field names to ID tokens.
        field2token_id (Dict[str, Dict[str, int]]): Mapping of field names to token IDs.
        user_feat (Dict[str, torch.Tensor]): User features.
        item_feat (Dict[str, torch.Tensor]): Item features.
        field2type (Dict[str, FeatureType]): Mapping of field names to feature types.
        inter_feat (Optional[pd.DataFrame]): Intermediate features.
        benchmark_filename_list (Optional[List[str]]): List of benchmark filenames.
        file_size_list (List[int]): List of file sizes.
        time_field (str): The field name for time.

    """

    def __init__(self, config, removal_setting: Optional[Literal['remove_old', 'remove_new']] = None):
        self.remove_old = (removal_setting == 'remove_old')
        self.remove_new = (removal_setting == 'remove_new')
        self.id2id_mapping: Optional[Dict[str, Dict[int, int]]] = None
        self.train_ufeatures: Optional[Dict[str, torch.Tensor]] = None
        self.train_ifeatures: Optional[Dict[str, torch.Tensor]] = None
        self.orig_dataset: Optional[Dataset] = None
        self.USER_ID = config['USER_ID_FIELD']
        self.ITEM_ID = config['ITEM_ID_FIELD']

        super().__init__(config)

    def _data_processing(self):
        """Data preprocessing, including:

        - Data filtering
        - Remap ID
        - Missing value imputation
        - Normalization
        - Preloading weights initialization
        """
        self.feat_name_list = self._build_feat_name_list()
        self._data_filtering()

        self._remap_ID_all()

        self._user_item_feat_preparation()
        self._fill_nan()
        self._set_label_by_threshold()
        self._normalize()
        self._discretization()
        self._preload_weight_matrix()

    def set_orig_dataset(self, orig_dataset: Dataset):
        self.orig_dataset = orig_dataset

    def remap_features(self):
        if self.orig_dataset is None:
            raise ValueError('The original dataset has not been set.')
        train_ufeatures = self.orig_dataset.user_feat
        train_ifeatures = self.orig_dataset.item_feat

        cross_feature_mapping = {}
        id2id_mapping = {}
        for field_name in self.orig_dataset.field2id_token.keys():
            if np.array_equal(self.orig_dataset.field2id_token[field_name], self.field2id_token[field_name]):
                continue
            if field_name in (self.USER_ID, self.ITEM_ID):
                continue
            cross_feature_mapping[field_name] = {}
            orig_vals = self.orig_dataset.field2id_token[field_name]
            ind_vals = self.field2id_token[field_name]
            for i in range(len(ind_vals)):
                if i >= len(orig_vals):
                    cross_feature_mapping[field_name][ind_vals[i]] = orig_vals[0]
                else:
                    cross_feature_mapping[field_name][ind_vals[i]] = orig_vals[i]
            self.field2id_token[field_name] = orig_vals

        for field_name in self.orig_dataset.field2token_id.keys():
            if np.array_equal(self.orig_dataset.field2token_id[field_name], self.field2token_id[field_name]):
                continue
            if field_name in (self.USER_ID, self.ITEM_ID):
                continue
            id2id_mapping[field_name] = {}

            orig_vals = self.orig_dataset.field2token_id[field_name]
            ind_vals = self.field2token_id[field_name]
            missing_vals = []
            for token_name, ind_id in ind_vals.items():
                if token_name in orig_vals:
                    id2id_mapping[field_name][ind_id] = orig_vals[token_name]
                else:
                    id2id_mapping[field_name][ind_id] = 0
                    missing_vals.append(token_name)
            self.field2token_id[field_name] = orig_vals
            self.field2token_id[field_name].update({ mv: 0 for mv in missing_vals })
            print('Missing vals for field:', field_name, ':', missing_vals)

        assert(self.user_feat is not None and self.item_feat is not None)
        assert(train_ufeatures is not None and train_ifeatures is not None)

        # Adjust feature shapes to ensure they are the same.
        for fname in id2id_mapping.keys():
            if fname in self.user_feat:
                print('Adjusting user feat:', fname)
                self.user_feat[fname] = self.user_feat[fname].apply_(lambda x: id2id_mapping[fname][x])
                # adjust shape if different from transductive
                if self.user_feat[fname].ndim > 1 and self.user_feat[fname].size(1) != train_ufeatures[fname].size(1):
                    print('Adjusting the size of feature:', fname)
                    self.user_feat[fname] = self.user_feat[fname][:, :train_ufeatures[fname].size(1)]

                if not torch.equal(train_ufeatures[fname][1:], self.user_feat[fname][1:train_ufeatures[fname].size(0)]):
                    print('Value mismatch: ', fname)
            elif fname in self.item_feat:
                print('Adjusting item feat:', fname)
                self.item_feat[fname] = self.item_feat[fname].apply_(lambda x: id2id_mapping[fname][x])
                # adjust shape if different from transductive
                if self.item_feat[fname].ndim > 1 and self.item_feat[fname].size(1) != train_ifeatures[fname].size(1):
                    print('Adjusting the size of feature:', fname)
                    self.item_feat[fname] = self.item_feat[fname][:, :train_ifeatures[fname].size(1)]

        # Remap mean interpolated columns to be the same
        for fname, ftype in self.field2type.items():
            if ftype != FeatureType.FLOAT:
                continue

            if fname in self.user_feat:
                train_feat = train_ufeatures[fname]
                ind_feat: torch.Tensor = self.user_feat[fname][1:train_feat.size(0)]
            elif fname in self.item_feat:
                train_feat = train_ifeatures[fname]
                ind_feat: torch.Tensor = self.item_feat[fname][1:train_feat.size(0)]
            else:
                print('Skipping feature:', fname)
                continue

            mismatch_mask = (train_feat[1:] != ind_feat)
            mismatch_vals = ind_feat[mismatch_mask]
            n_mismatches = torch.sum(mismatch_mask).item()
            if n_mismatches == 0:
                continue

            # Make sure that all missing values are the same
            assert(torch.all(mismatch_vals == mismatch_vals[0]))
            orig_vals = train_feat[1:][mismatch_mask]
            # Make sure that all original values are the same
            assert(torch.all(orig_vals == orig_vals[0]))
            # Replace all missing values with the same value
            ind_feat[mismatch_mask] = orig_vals[0]

        print('Done remapping features.')

    def build(self):
        """Processing dataset according to evaluation setting, including Group, Order and Split.
        See :class:`~recbole.config.eval_setting.EvalSetting` for details.

        Returns:
            list: List of built :class:`Dataset`.
        """
        self._change_feat_format()
        self.remap_features()

        if self.benchmark_filename_list is not None:
            assert(self.inter_feat is not None)

            self._drop_unused_col()
            cumsum = list(np.cumsum(self.file_size_list))
            datasets = [
                self.copy(self.inter_feat[start:end])
                for start, end in zip([0] + cumsum[:-1], cumsum)
            ]
            return datasets

        # ordering
        ordering_args = self.config["eval_args"]["order"]
        if ordering_args == "RO":
            self.shuffle()
        elif ordering_args == "TO":
            self.sort(by=self.time_field)
        else:
            raise NotImplementedError(
                f"The ordering_method [{ordering_args}] has not been implemented."
            )

        # splitting & grouping
        split_args = self.config["eval_args"]["split"]
        if split_args is None:
            raise ValueError("The split_args in eval_args should not be None.")
        if not isinstance(split_args, dict):
            raise ValueError(f"The split_args [{split_args}] should be a dict.")

        split_mode = list(split_args.keys())[0]
        assert len(split_args.keys()) == 1
        group_by = self.config["eval_args"]["group_by"]
        if split_mode == "RS":
            if not isinstance(split_args["RS"], list):
                raise ValueError(f'The value of "RS" [{split_args}] should be a list.')
            if group_by is None or group_by.lower() == "none":
                datasets = self.split_by_ratio(split_args["RS"], group_by=None)
            elif group_by == "user":
                datasets = self.split_by_ratio(
                    split_args["RS"], group_by=self.uid_field
                )
            else:
                raise NotImplementedError(
                    f"The grouping method [{group_by}] has not been implemented."
                )
        elif split_mode == "LS":
            datasets = self.leave_one_out(
                group_by=self.uid_field, leave_one_mode=split_args["LS"]
            )
        else:
            raise NotImplementedError(
                f"The splitting_method [{split_mode}] has not been implemented."
            )

        return datasets