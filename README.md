# Improving Out-of-Vocabulary Handling in Recommendation Systems

This repository contains the code for the paper ["Improving Out-of-Vocabulary Handling in Recommendation Systems"](https://arxiv.org/abs/2403.18280).

## Setup

First, install our local copy of RecBole. You will need to uninstall the original RecBole first (if already installed). (This framework is a major extension to RecBole https://github.com/RUCAIBox/RecBole. Recbole provides the basic framework to evaluate on recommendation system models with standard datasets. This framework enables inductive training and evaluation, as well as datasets.)

```bash
pip uninstall recbole
cd RecBole
pip install -e .
```

Then, you can run the code:

```bash
cd ../src
python run_recbole.py --dataset [DATASET]
```

The `run_recbole.py` script accepts the following general parameters:

- `--dataset`: Mandatory. The dataset name to train and evaluate on.
- `--model`: Mandatory. The model type. The currently supported types are `BPR`, `DirectAU`, `DCNV2`, `WideDeep`, and `xDeepFM`.
- `--checkpoint_dir`: Directory to store the model weights.
- `--embedding_size`: The embedding size to use.
- `--train_batch_size`: The training batch size.
- `--eval_batch_size`: The evaluation batch size.
- `--weight_decay`: Weight decay to use during training (if any).
- `--gcs_bucket_name`: The name of the GCS bucket you would like to write model weights to, if applicable. Note that you need write permissions on the bucket and should already be authenticated.
- `--learning_rate`: The model learning rate. Defaults to 
- `--log_wandb`: Whether or not to log results to Weights and Biases.
- `--model_eval_type`: The model evaluation metrics type - either `retrieval` or `ranking`.

There are also inductive-specific parameters that can be passed in:
- `--add_oov_buckets`: Whether or not to add OOV buckets to the model. This should be true if using any trainable OOV methods.
- `--inductive_embedder`: The inductive embedder to use. The currently supported types are `lsh`, `slsh`, `knn`, `dnn`, `dhe`, `fdhe`, `zero`, and `mean`.
- `--inductive_mapper`: The inductive mapper to use. The only currently supported type is `random`.
- `--inductive_eval`: Whether or not to perform inductive evaluation (versus training only)
- `--user_oov_buckets`: The number of user OOV buckets to use.
- `--item_oov_buckets`: The number of item OOV buckets to use.
- `--oov_feature_mask_rate`: The rate at which to mask OOV features during training.
- `--oov_freeze_embedding`: Whether or not to freeze IV embeddings during OOV training.
- `--oov_freeze_skip_optim`: Whether or not to also freeze the optimizer parameters.
- `--train_oov`: Whether or not to train OOV embeddings at all. Should be true for all embedders except for `zero` and `mean`.
- `--oov_only_epoch`: Whether or not to split OOV samples out into its own epoch.
- `--oov_train_ratio`: Ratio of IV samples used for OOV training at every epoch.
- `--oov_normalization_type`: Feature normalization type. Can be one of three options: per-feature, global, none. Not implemented for all OOV embedders.

Note that any empty parameters will use the default parameters found in `RecBole/recbole/properties/overall.yaml` unless overriden by model-specific or dataset-specific configuration files. You can also pass in any model-specific hyperparameters here, which can be found in the `RecBole/recbole/properties/model/MODEL_NAME.yaml` files.

## Citation

If you use this code in your research, please cite the following paper:

```tex
@article{shiao2024improving,
  title={Improving Out-of-Vocabulary Handling in Recommendation Systems},
  author={Shiao, William and Ju, Mingxuan and Guo, Zhichun and Chen, Xin and Papalexakis, Evangelos and Zhao, Tong and Shah, Neil and Liu, Yozen},
  journal={arXiv preprint arXiv:2403.18280},
  year={2024}
}
```
