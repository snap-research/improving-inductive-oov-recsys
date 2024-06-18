import copy
import importlib
import os
import pickle
import torch

from logging import getLogger
from recbole.data.utils import create_dataset, data_preparation
from recbole.inductive.evaluator import InductiveEvaluator
from recbole.inductive.get_inductive import get_inductive_embedder, get_inductive_mapper
from recbole.model.abstract_recommender import InductiveContextRecommender, InductiveGeneralRecommender
from recbole.utils.argument_list import dataset_arguments
from recbole.utils.enum_type import ModelType
from recbole.utils.logger import set_color
from recbole.utils.utils import get_model, init_seed


def create_id_mapping(series):
    """
    Create a mapping of unique values in the given series to unique IDs.

    Args:
        series (pandas.Series): The series containing the values to be mapped.

    Returns:
        dict: A dictionary mapping each unique value in the series to a unique ID.
    """
    count = 1  # due to RecBole creating a dummy entry at 0
    id_mapping = {}
    for i in series:
        if i not in id_mapping:
            id_mapping[i] = count
            count += 1
    return id_mapping


def create_ind_dataset(config):
    """Create dataset according to :attr:`config['model']` and :attr:`config['MODEL_TYPE']`.
    If :attr:`config['dataset_save_path']` file exists and
    its :attr:`config` of dataset is equal to current :attr:`config` of dataset.
    It will return the saved dataset in :attr:`config['dataset_save_path']`.

    Args:
        config (Config): An instance object of Config, used to record parameter information.

    Returns:
        Dataset: Constructed dataset.
    """
    dataset_module = importlib.import_module("recbole.data.dataset")
    if hasattr(dataset_module, config["model"] + "Dataset"):
        dataset_class = getattr(dataset_module, config["model"] + "Dataset")
    else:
        model_type = config["MODEL_TYPE"]
        type2class = {
            ModelType.GENERAL: "Dataset",
            ModelType.SEQUENTIAL: "SequentialDataset",
            ModelType.CONTEXT: "Dataset",
            ModelType.KNOWLEDGE: "KnowledgeBasedDataset",
            ModelType.TRADITIONAL: "Dataset",
            ModelType.DECISIONTREE: "Dataset",
        }
        dataset_class = getattr(dataset_module, type2class[model_type])

    default_file = os.path.join(config["checkpoint_dir"], f'{config["dataset"]}-{dataset_class.__name__}.pth')
    file = config["dataset_save_path"] or default_file
    if os.path.exists(file):
        with open(file, "rb") as f:
            dataset = pickle.load(f)
        dataset_args_unchanged = True
        for arg in dataset_arguments + ["seed", "repeatable"]:
            if config[arg] != dataset.config[arg]:
                dataset_args_unchanged = False
                break
        if dataset_args_unchanged:
            logger = getLogger()
            logger.info(set_color("Load filtered dataset from", "pink") + f": [{file}]")
            return dataset

    dataset = dataset_class(config)
    if config["save_dataset"]:
        dataset.save()
    return dataset


def perform_inductive_eval(orig_dataset,
                           checkpoint_path,
                           mapper_path,
                           embedder_path,
                           oov_eval_batch_size=-1,
                           show_progress=True):
    stored_dict = torch.load(checkpoint_path)
    config = stored_dict['config']
    init_seed(config['seed'], config['reproducibility'])

    config.compatibility_settings()

    train_ufeatures = orig_dataset.get_user_feature()
    train_ifeatures = orig_dataset.get_item_feature()

    print('Preparing for inductive evaluation...')
    inductive_config = copy.deepcopy(stored_dict['config'])
    inductive_config['dataset'] = config['dataset'] + '_ind'
    inductive_config['benchmark_filename'] = ['train', 'empty', 'test_filt']
    inductive_config['topk'] = [3, 5, 10, 20]
    if oov_eval_batch_size > 0:
        print('Using OOV eval batch size: ', oov_eval_batch_size)
        inductive_config['eval_batch_size'] = oov_eval_batch_size
    ind_dataset = create_dataset(inductive_config, inductive=True)
    ind_dataset.set_orig_dataset(orig_dataset)
    _, _, test_data = data_preparation(inductive_config, ind_dataset)

    ind_ufeatures = ind_dataset.get_user_feature()
    ind_ifeatures = ind_dataset.get_item_feature()

    # A series of assertions and checks to ensure that we aren't scrambling any user rows or columns.
    # This can happen if the dataset is not loaded or processed correctly.
    for col in ind_ufeatures.columns[1:]:
        if len(ind_ufeatures[col].size()) < 2:
            if not torch.equal(train_ufeatures[col][1:], ind_ufeatures[col][1:train_ufeatures[col].size(0)]):
                print('Value mismatch: ', col)
            continue
        if train_ufeatures[col].size()[1:] != ind_ufeatures[col].size()[1:]:
            print('Size mismatch: ', col)
            continue
        if not torch.equal(train_ufeatures[col][1:], ind_ufeatures[col][1:train_ufeatures[col].size(0)]):
            print('Value mismatch: ', col)

    # We repeat the same check for the item rows/columns.
    for col in ind_ifeatures.columns[1:]:
        if len(ind_ifeatures[col].size()) < 2:
            if not torch.equal(train_ifeatures[col][1:], ind_ifeatures[col][1:train_ifeatures[col].size(0)]):
                print('Value mismatch: ', col)
            continue
        if train_ifeatures[col].size()[1:] != ind_ifeatures[col].size()[1:]:
            print('Size mismatch: ', col)
            continue
        if not torch.equal(train_ifeatures[col][1:], ind_ifeatures[col][1:train_ifeatures[col].size(0)]):
            print('Value mismatch: ', col)

    # Now, we fetch the inductive mapper and/or embedder. They are None if we don't have either.
    mapper = get_inductive_mapper(inductive_config,
                                  ind_dataset,
                                  user_num=orig_dataset.user_num,
                                  item_num=orig_dataset.item_num)
    embedder = get_inductive_embedder(inductive_config,
                                      ind_dataset,
                                      user_num=orig_dataset.user_num,
                                      item_num=orig_dataset.item_num,
                                      mode='inductive')
    model_class = get_model(config['model'])

    # We check the model class to see if it is inductive or not to try and preserve backwards compatibility.
    if issubclass(model_class, (InductiveGeneralRecommender, InductiveContextRecommender)):
        model = model_class(config, orig_dataset, inductive_mapper=mapper, inductive_embedder=embedder)
    else:
        model = model_class(config, orig_dataset)

    # Load the model parameters
    model = model.to(config['device'])
    model.load_state_dict(stored_dict['state_dict'])
    # This step should no longer be necessary for newer models since we store the other parameters in the model itself.
    # We keep this for backwards compatibility.
    model.load_other_parameter(stored_dict.get('other_parameter'))
    model.eval()

    print('Processing inductive dataset')
    # Now, we evaluate the model on the inductive dataset.
    with torch.no_grad():
        ie = InductiveEvaluator(model, inductive_config, orig_dataset.user_num, orig_dataset.item_num)
        return ie.evaluate_model(test_data, inductive_config, show_progress=True, inductive=True)
