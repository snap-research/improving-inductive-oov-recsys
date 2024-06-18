# This script is based off of the original run_recbole.py script but has been heavily modified
# to facilitate inductive evaluation of models.
# The script is designed to be run as a standalone script and can be used to train and evaluate models.

import json
import os
import shutil
import sys
from logging import getLogger
import uuid
from recbole.inductive.get_inductive import get_inductive_embedder, get_inductive_mapper
from recbole.model.abstract_recommender import InductiveContextRecommender, InductiveGeneralRecommender
import torch

import wandb
from utils.parse import custom_parse_args
from google.cloud import storage
from recbole.config import Config
from recbole.data import (
    create_dataset,
    data_preparation,
)
from recbole.data.transform import construct_transform
from recbole.utils import (
    init_logger,
    get_model,
    get_trainer,
    init_seed,
    set_color,
    get_flops,
    get_environment,
    get_local_time
)
from perform_hashing import perform_inductive_eval

# default bucket used for saving to GCS
GCS_BUCKET_DIRECTORY = 'recsys-weights'

def run_recbole(
    model=None, dataset=None, config_file_list=None, config_dict=None, saved=True
):
    r'''A fast running api, which includes the complete process of
    training and testing a model on a specified dataset

    Args:
        model (str, optional): Model name. Defaults to ``None``.
        dataset (str, optional): Dataset name. Defaults to ``None``.
        config_file_list (list, optional): Config files used to modify experiment parameters. Defaults to ``None``.
        config_dict (dict, optional): Parameters dictionary used to modify experiment parameters. Defaults to ``None``.
        saved (bool, optional): Whether to save the model. Defaults to ``True``.
    '''
    # configuration initialization
    config = Config(
        model=model,
        dataset=dataset,
        config_file_list=config_file_list,
        config_dict=config_dict,
    )
    init_seed(config['seed'], config['reproducibility'])

    # logger initialization
    init_logger(config)
    logger = getLogger()
    logger.info(sys.argv)
    logger.info(config)

    # dataset filtering
    dataset = create_dataset(config)
    logger.info(dataset)

    # dataset splitting
    train_data, valid_data, test_data = data_preparation(config, dataset)

    # model loading and initialization
    init_seed(config['seed'] + config['local_rank'], config['reproducibility'])
    model_class = get_model(config['model'])

    ind_mapper = None
    ind_embedder = None
    if issubclass(model_class, (InductiveGeneralRecommender, InductiveContextRecommender)):
        ind_mapper = get_inductive_mapper(config, dataset)
        ind_embedder = get_inductive_embedder(config, dataset)
        model = model_class(config, train_data._dataset, inductive_mapper=ind_mapper, inductive_embedder=ind_embedder)
    else:
        model = model_class(config, train_data._dataset)
    model = model.to(config['device'])
    logger.info(model)

    transform = construct_transform(config)
    flops = get_flops(model, dataset, config['device'], logger, transform)
    logger.info(set_color('FLOPs', 'blue') + f': {flops}')

    # trainer loading and initialization
    trainer = get_trainer(config['MODEL_TYPE'], config['model'], '-ind' in config['dataset'])(config, model)
    saved_model_file_name = '{}-{}-{}.pth'.format(trainer.config['model'], get_local_time(), str(uuid.uuid4())[:6])
    saved_embedder_file_name = '{}-{}-{}.pth'.format('embedder', get_local_time(), str(uuid.uuid4())[:6])
    saved_mapper_file_name = '{}-{}-{}.pth'.format('mapper', get_local_time(), str(uuid.uuid4())[:6])

    dataset_dir_path = os.path.join(trainer.checkpoint_dir, config['dataset'])
    os.makedirs(dataset_dir_path, exist_ok=True)
    saved_model_file = os.path.join(dataset_dir_path, saved_model_file_name)
    trainer.saved_model_file = saved_model_file

    saved_model_file = os.path.join(dataset_dir_path, saved_model_file_name)
    saved_embedder_file = os.path.join(dataset_dir_path, saved_embedder_file_name)
    saved_mapper_file = os.path.join(dataset_dir_path, saved_mapper_file_name)

    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, saved=saved, show_progress=config['show_progress']
    )

    if ind_mapper is not None:
        torch.save(model.inductive_mapper.state_dict(), saved_mapper_file)
    if ind_embedder is not None:
        torch.save(model.inductive_embedder.state_dict(), saved_embedder_file)

    has_backup = False

    # save to GCS (if enabled)
    try:
        if 'gcs_bucket_name' in config:
            bucket_name = config['gcs_bucket_name']
            storage_client = storage.Client()
            bucket = storage_client.bucket(bucket_name)
            blob = bucket.blob(os.path.join('recsys-mkiii-weights', config['dataset'], saved_model_file_name))
            generation_match_precondition = 0
            blob.upload_from_filename(saved_model_file, if_generation_match=generation_match_precondition)
            has_backup = True
            if config['log_wandb']:
                wandb.log({ 'file_backup_medium': 'gcs' }, commit=False)
    except Exception as e:
        print('Failed to upload to GCS: ', e)

    try:
        backup_path = config['nfs_backup_path']
        backup_dir = os.path.join(backup_path, config['dataset'])
        os.makedirs(backup_dir, exist_ok=True)

        print('Backing up to NFS...')
        shutil.copy(saved_model_file, backup_dir)
        has_backup = True
        if config['log_wandb']:
            wandb.log({ 'file_backup_medium': 'nfs' }, commit=False)
    except Exception as e:
            print('Failed to write to NFS: ', e)

    if not has_backup:
        try:
            print('Trying to backup to local...')
            current_script_dir = os.path.dirname(os.path.realpath(__file__))
            backup_dir = os.path.join(current_script_dir, 'saved/', config['dataset'])
            os.makedirs(backup_dir, exist_ok=True)

            backup_path = os.path.join(backup_dir, saved_model_file_name)
            shutil.copy(saved_model_file, backup_path)

            has_backup = True
            if config['log_wandb']:
                wandb.log({ 'file_backup_medium': 'local' }, commit=False)
            print('Backup saved to: ', backup_path)
        except Exception as e:
            print('Failed to write to local: ', e)

    if not has_backup:
        print('Failed to backup to any medium!')
        wandb.alert(title='Backup Failure', text='Failed to backup to any medium!', level='error')

    # model evaluation
    test_result = trainer.evaluate(
        test_data, load_best_model=saved, show_progress=config['show_progress']
    )

    environment_tb = get_environment(config)
    logger.info(
        'The running environment of this training is as follows:\n'
        + environment_tb.draw()
    )

    logger.info(set_color('best valid ', 'yellow') + f': {best_valid_result}')
    logger.info(set_color('test result', 'yellow') + f': {test_result}')
    if config['log_wandb']:
        wandb.log({ 'test_'+k: v for k, v in dict(test_result).items() })
        wandb.log({
            'best_valid_score':best_valid_score,
            'checkpoint_path': saved_model_file,
            'embedder_path': saved_embedder_file,
            'mapper_path': saved_mapper_file
        })

    return {
        'best_valid_score': best_valid_score,
        'valid_score_bigger': config['valid_metric_bigger'],
        'best_valid_result': best_valid_result,
        'test_result': test_result,
        'checkpoint_path': saved_model_file,
        'embedder_path': saved_embedder_file,
        'mapper_path': saved_mapper_file,
        'dataset': dataset
    }, config


if __name__ == '__main__':
    args = custom_parse_args()

    if args.get('model_eval_type', 'retrieval') == 'retrieval':
        args['valid_metric'] = 'MRR@20'
        args['metrics'] = ['Recall', 'MRR', 'NDCG', 'Hit', 'Precision']
    elif args['model_eval_type'] == 'ranking':
        args['valid_metric'] = 'RMSE'
        args['metrics'] = ['AUC', 'RMSE']
    else:
        raise NotImplementedError('Unknown model type: ', args['model_eval_type'])

    args['eval_args'] = {
        'split': {
            'RS': [0.88, 0.02, 0.1]
        },
        'group_by': None,
        'order': 'TO',
        'mode': 'uni250'
    }
    args['eval_batch_size'] = int(1e5)
    args['topk'] = [10, 20]
    args['train_neg_sample_args'] = { 'distribution': 'uniform', 'sample_num': 1, 'alpha': 1.0, 'dynamic': False, 'candidate_num': 0 }
    args['oov_neg_sample_args'] = { 'distribution': 'uniform', 'sample_num': 1, 'alpha': 1.0, 'dynamic': False, 'candidate_num': 0 }
    args['test_neg_sample_args'] = { 'distribution': 'uniform', 'sample_num': 1, 'alpha': 1.0, 'dynamic': False, 'candidate_num': 0 }
    args['threshold'] = None
    args['fixed_dataset_issue'] = True
    args['reproducibility'] = True
    # check for override env vars
    if 'GPU_OVERRIDE' in os.environ:
        args['gpu_id'] = os.environ['GPU_OVERRIDE']

    # check for dataset config
    dataset_name = args['dataset']
    dataset_config = os.path.join('./dataset_configs', f'{dataset_name}.json')
    if os.path.exists(dataset_config):
        print('Dataset config file found:', dataset_config)
        with open(dataset_config, 'rb') as f:
            config_dict = json.load(f)
        args = { **config_dict, **args }
    else:
        print('Dataset config file not found, using defaults...')

    recbole_results, config = run_recbole(
        model=args['model'], dataset=args['dataset'], config_dict=args
    )

    if args['inductive_eval']:
        print('Performing inductive evaluation...')
        inductive_results = perform_inductive_eval(orig_dataset=recbole_results['dataset'],
                                                   checkpoint_path=recbole_results['checkpoint_path'],
                                                   mapper_path=recbole_results['mapper_path'],
                                                   embedder_path=recbole_results['embedder_path'],
                                                   oov_eval_batch_size=config['oov_eval_batch_size'])

        logger = getLogger()
        logger.info(set_color('inductive test results', 'yellow') + f': {inductive_results}')

        if config['log_wandb'] and wandb.run is not None:
            ind_values = { f'ind_{k}': v for k, v in inductive_results.items() }
            wandb.log(ind_values)
            for k, v in ind_values.items():
                wandb.run.summary[k] = v

            wandb.finish()
