from typing import Union
from recbole.inductive.dh_embedder import DeepHashEmbedder
from recbole.inductive.feat_dh_embedder import FeatDeepHashEmbedder
from recbole.inductive.dnn_embedder import DNNEmbedder
from recbole.inductive.lsh_embedder import LSHInductiveEmbedder
from recbole.inductive.zero_embedder import ZeroEmbedder
from recbole.inductive.abstract_mapper import AbstractInductiveMapper
from recbole.inductive.mean_embedder import MeanEmbedder
from recbole.inductive.knn_embedder import KNNInductiveEmbedder
from recbole.inductive.random_mapper import RandomOOVInductiveMapper
from recbole.inductive.single_lsh_embedder import SingleLSHInductiveEmbedder
from recbole.inductive.feature_cache import InductiveFeatureCache

feat_cache = InductiveFeatureCache()

def get_inductive_mapper(config,
                         dataset,
                         user_num=None,
                         item_num=None,
                         embedding_size=None,
                         first_order=False) -> Union[AbstractInductiveMapper, None]:
    if embedding_size is None:
        embedding_size = config['embedding_size']

    if config['inductive_mapper'] == 'random':
        return RandomOOVInductiveMapper(user_features=dataset.get_user_feature(),
                                        item_features=dataset.get_item_feature(),
                                        n_original_users=user_num or dataset.user_num,
                                        n_original_items=item_num or dataset.item_num,
                                        n_user_oov_buckets=config['user_oov_buckets'],
                                        n_item_oov_buckets=config['item_oov_buckets'],
                                        embedding_size=embedding_size,
                                        device=config['device'],
                                        prime_pad=config['oov_prime_pad'],
                                        hash_function=config['oov_hash_function'])
    return None


def get_inductive_embedder(config, dataset, mode='transductive', user_num=None, item_num=None, embedding_size=None, first_order=False):
    """Factory method to get the inductive embedder based on the config.
    We use a global variable feat_cache to store the feature cache, so that we can reuse the cache when the mode is the same.
    This helps facilitate the sharing of the feature matrix across multiple inductive embedder instances.
    """
    # Note: the use of this global variable is not ideal and could be problematic
    # if we are working with multiple datasets at the same time.
    global feat_cache
    if feat_cache.get_mode() != mode:
        # reset feature cache if mode has changed
        del feat_cache
        feat_cache = InductiveFeatureCache(mode=mode)

    if embedding_size is None:
        embedding_size = config['embedding_size']
    if config['inductive_embedder'] == 'knn':
        return KNNInductiveEmbedder(user_features=dataset.get_user_feature(),
                                    item_features=dataset.get_item_feature(),
                                    n_original_users=user_num or dataset.user_num,
                                    n_original_items=item_num or dataset.item_num,
                                    n_user_oov_buckets=config['user_oov_buckets'],
                                    n_item_oov_buckets=config['item_oov_buckets'],
                                    embedding_size=embedding_size,
                                    device=config['device'],
                                    prime_pad=config['oov_prime_pad'],
                                    n_neighbors=config['oov_knn_num_neighbors'])
    if config['inductive_embedder'] == 'lsh':
        return LSHInductiveEmbedder(user_features=dataset.get_user_feature(),
                                    item_features=dataset.get_item_feature(),
                                    n_original_users=user_num or dataset.user_num,
                                    n_original_items=item_num or dataset.item_num,
                                    n_user_oov_buckets=config['user_oov_buckets'],
                                    n_item_oov_buckets=config['item_oov_buckets'],
                                    embedding_size=embedding_size,
                                    device=config['device'],
                                    prime_pad=config['oov_prime_pad'],
                                    normalization_type=config['oov_normalization_type'],
                                    feature_cache=feat_cache)
    elif config['inductive_embedder'] == 'slsh':
        return SingleLSHInductiveEmbedder(user_features=dataset.get_user_feature(),
                                          item_features=dataset.get_item_feature(),
                                          n_original_users=user_num or dataset.user_num,
                                          n_original_items=item_num or dataset.item_num,
                                          n_user_oov_buckets=config['user_oov_buckets'],
                                          n_item_oov_buckets=config['item_oov_buckets'],
                                          embedding_size=embedding_size,
                                          device=config['device'],
                                          prime_pad=config['oov_prime_pad'],
                                          normalization_type=config['oov_normalization_type'])
    elif config['inductive_embedder'] == 'dhe':
        return DeepHashEmbedder(user_features=dataset.get_user_feature(),
                                item_features=dataset.get_item_feature(),
                                n_original_users=user_num or dataset.user_num,
                                n_original_items=item_num or dataset.item_num,
                                n_user_oov_buckets=config['user_oov_buckets'],
                                n_item_oov_buckets=config['item_oov_buckets'],
                                embedding_size=embedding_size,
                                device=config['device'],
                                prime_pad=config['oov_prime_pad'],
                                num_hashes=config['dhe_num_hashes'])
    elif config['inductive_embedder'] == 'fdhe':
        return FeatDeepHashEmbedder(user_features=dataset.get_user_feature(),
                                    item_features=dataset.get_item_feature(),
                                    n_original_users=user_num or dataset.user_num,
                                    n_original_items=item_num or dataset.item_num,
                                    n_user_oov_buckets=config['user_oov_buckets'],
                                    n_item_oov_buckets=config['item_oov_buckets'],
                                    embedding_size=embedding_size,
                                    device=config['device'],
                                    prime_pad=config['oov_prime_pad'],
                                    num_hashes=config['dhe_num_hashes'],
                                    dhe_layer_size=config['dhe_layer_size'])
    elif config['inductive_embedder'] == 'dnn':
        return DNNEmbedder(user_features=dataset.get_user_feature(),
                                    item_features=dataset.get_item_feature(),
                                    n_original_users=user_num or dataset.user_num,
                                    n_original_items=item_num or dataset.item_num,
                                    n_user_oov_buckets=config['user_oov_buckets'],
                                    n_item_oov_buckets=config['item_oov_buckets'],
                                    embedding_size=embedding_size,
                                    device=config['device'],
                                    prime_pad=config['oov_prime_pad'],
                                    dhe_layer_size=config['dhe_layer_size'])
    elif config['inductive_embedder'] == 'mean':
        return MeanEmbedder(user_features=dataset.get_user_feature(),
                            item_features=dataset.get_item_feature(),
                            n_original_users=user_num or dataset.user_num,
                            n_original_items=item_num or dataset.item_num,
                            n_user_oov_buckets=config['user_oov_buckets'],
                            n_item_oov_buckets=config['item_oov_buckets'],
                            embedding_size=embedding_size,
                            device=config['device'])
    elif config['inductive_embedder'] == 'zero':
        return ZeroEmbedder(user_features=dataset.get_user_feature(),
                            item_features=dataset.get_item_feature(),
                            n_original_users=user_num or dataset.user_num,
                            n_original_items=item_num or dataset.item_num,
                            embedding_size=embedding_size,
                            device=config['device'])
    return None
