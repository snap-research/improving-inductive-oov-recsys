from recbole.data.interaction import Interaction
from recbole.inductive.collector_filter import FastUserCollectorFilter, FastItemCollectorFilter, MultiCollectorFilter, FastUserItemCollectorFilter
from recbole.inductive.filtered_collector import FilteredCollector
from recbole.utils.enum_type import EvaluatorType
import torch
from functools import partial
import numpy as np
from tqdm import tqdm
from recbole.evaluator.collector import Collector
from recbole.evaluator.evaluator import Evaluator
from recbole.utils.logger import set_color
from recbole.utils.utils import get_gpu_usage
from recbole.data.dataloader import FullSortEvalDataLoader

class InductiveEvaluator:
    """This class is a slim version of the Trainer class that only contains the evaluation-related code.
    The evaluation methods have been modified to add support for inductive evaluation.
    """
    def __init__(self, model, config, n_old_users, n_old_items, feature_extractor=None) -> None:
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.gpu_available = torch.cuda.is_available()
        self.config = config
        self.test_batch_size = config['test_batch_size']
        self.test_batch_size = config['eval_batch_size']

        # define basic filters
        self.old_user_filter = FastUserItemCollectorFilter(n_old_users, n_old_items, return_old_users=True, return_old_items=None)
        self.new_user_filter = FastUserItemCollectorFilter(n_old_users, n_old_items, return_old_users=False, return_old_items=None)
        self.old_item_filter = FastUserItemCollectorFilter(n_old_users, n_old_items, return_old_users=None, return_old_items=True)
        self.new_item_filter = FastUserItemCollectorFilter(n_old_users, n_old_items, return_old_users=None, return_old_items=False)

        # define composite filters
        self.old_old_filter = FastUserItemCollectorFilter(n_old_users, n_old_items, return_old_users=True, return_old_items=True)
        self.old_new_filter = FastUserItemCollectorFilter(n_old_users, n_old_items, return_old_users=True, return_old_items=False)
        self.new_old_filter = FastUserItemCollectorFilter(n_old_users, n_old_items, return_old_users=False, return_old_items=True)
        self.new_new_filter = FastUserItemCollectorFilter(n_old_users, n_old_items, return_old_users=False, return_old_items=False)

        self.collectors = {
            'overall': Collector(config),
            'old_users': FilteredCollector(config, self.old_user_filter, 'old_users'),
            'new_users': FilteredCollector(config, self.new_user_filter, 'new_users'),
            'old_old': FilteredCollector(config, self.old_old_filter, 'old_old'),
            'old_new': FilteredCollector(config, self.old_new_filter, 'old_new'),
            'new_old': FilteredCollector(config, self.new_old_filter, 'new_old'),
            'new_new': FilteredCollector(config, self.new_new_filter, 'new_new'),
        }
        self.evaluators = {
            'overall': Evaluator(config),
            'old_users': Evaluator(config),
            'new_users': Evaluator(config),
            'old_old': Evaluator(config),
            'old_new': Evaluator(config),
            'new_old': Evaluator(config),
            'new_new': Evaluator(config)
        }

        self.USER_ID = config['USER_ID_FIELD']
        self.ITEM_ID = config['ITEM_ID_FIELD']
        self.item_tensor = None
        self.model = model
        self.evaluator = Evaluator(config)
        self.tot_item_num = None
        self.feature_extractor = feature_extractor
        self.item_range = None
        self.n_old_users = n_old_users
        self.n_old_items = n_old_items


    def eval_batch(self, batched_data, inductive=False):
        interaction, history_index, positive_u, positive_i = batched_data
        if (interaction[self.USER_ID] < self.n_old_users).any() or (interaction[self.ITEM_ID < self.n_old_items]).any():
            print('found old item')
        try:
            # Note: interaction without item ids
            if inductive:
                scores = self.model.ind_full_sort_predict(interaction.to(self.device), self.item_range)
            else:
                scores = self.model.full_sort_predict(interaction.to(self.device))
        except NotImplementedError:
            inter_len = len(interaction)
            new_inter = interaction.to(self.device).repeat_interleave(self.tot_item_num)
            batch_size = len(new_inter)
            new_inter.update(self.item_tensor.repeat(inter_len))
            if batch_size <= self.test_batch_size:
                scores = self.model.predict(new_inter)
            else:
                raise ValueError()
                scores = self._spilt_predict(new_inter, batch_size)

        scores = scores.view(-1, self.tot_item_num)
        scores[:, 0] = -np.inf
        if history_index is not None:
            scores[history_index] = -np.inf
        return interaction, scores, positive_u, positive_i

    def _spilt_predict(self, interaction, batch_size):
        spilt_interaction = dict()
        for key, tensor in interaction.interaction.items():
            spilt_interaction[key] = tensor.split(self.test_batch_size, dim=0)
        num_block = (batch_size + self.test_batch_size - 1) // self.test_batch_size
        result_list = []
        for i in range(num_block):
            current_interaction = dict()
            for key, spilt_tensor in spilt_interaction.items():
                current_interaction[key] = spilt_tensor[i]
            result = self.model.predict(
                Interaction(current_interaction).to(self.device)
            )
            if len(result.shape) == 0:
                result = result.unsqueeze(0)
            result_list.append(result)
        return torch.cat(result_list, dim=0)


    def neg_sample_batch_eval(self, batched_data, inductive=False):
        interaction, row_idx, positive_u, positive_i = batched_data
        batch_size = interaction.length
        if batch_size <= self.test_batch_size:
            origin_scores = self.model.predict(interaction.to(self.device))
        else:
            origin_scores = self._spilt_predict(interaction, batch_size)

        if self.config['eval_type'] == EvaluatorType.VALUE:
            return interaction, origin_scores, positive_u, positive_i
        elif self.config['eval_type'] == EvaluatorType.RANKING:
            col_idx = interaction[self.config['ITEM_ID_FIELD']]
            batch_user_num = positive_u[-1] + 1
            scores = torch.full(
                (batch_user_num, self.tot_item_num), -np.inf, device=self.device
            )
            scores[row_idx, col_idx] = origin_scores
            return interaction, scores, positive_u, positive_i


    def evaluate_model(self, eval_data, config, show_progress=True, inductive=False):
        self.model.eval()

        self.tot_item_num = eval_data._dataset.item_num
        self.item_range = torch.arange(self.tot_item_num, device=self.device)

        if isinstance(eval_data, FullSortEvalDataLoader):
            eval_func = partial(self.eval_batch, inductive=inductive)
            # item_tensor = eval_data._dataset.get_item_feature().to(self.device)
        else:
            eval_func = partial(self.neg_sample_batch_eval, inductive=inductive)

        iter_data = (
            tqdm(
                eval_data,
                total=len(eval_data),
                ncols=100,
                desc=set_color(f"Evaluate   ", "pink"),
            )
            if show_progress
            else eval_data
        )

        num_sample = 0
        for batch_idx, batched_data in enumerate(iter_data):
            num_sample += len(batched_data)
            interaction, scores, positive_u, positive_i = eval_func(batched_data)
            if self.gpu_available and show_progress:
                iter_data.set_postfix_str(
                    set_color("GPU RAM: " + get_gpu_usage(self.device), "yellow")
                )
            for _, collector in self.collectors.items():
                collector.eval_batch_collect(
                    scores, interaction, positive_u, positive_i
                )

        results = {}
        for name, collector in self.collectors.items():
            collector.model_collect(self.model)
            struct = collector.get_data_struct()
            # we assume the the metrics don't maintain state (true for current ones).
            # this allows us to reuse the same evaluator for different models.
            results[name] = self.evaluators[name].evaluate(struct)

        return results
