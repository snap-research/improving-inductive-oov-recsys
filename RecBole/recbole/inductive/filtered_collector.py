from recbole.evaluator.collector import Collector
from recbole.inductive.collector_filter import InductiveCollectorFilter
import torch

class FilteredCollector(Collector):
    """An extension of the Collector class that filters the evaluation data before collecting it.
    This allows us to evaluate the performance of our model on a subset of the data.
    """
    def __init__(self, config, ind_filter: InductiveCollectorFilter, name=None):
        super().__init__(config)
        self.filter = ind_filter
        self.USER_ID = config['USER_ID_FIELD']
        self.ITEM_ID = config['ITEM_ID_FIELD']
        self.name = name
        self.model_eval_type = config['model_eval_type']
        self.is_ranking_model = self.model_eval_type == 'ranking'

    def eval_batch_collect(
        self,
        orig_scores_tensor: torch.Tensor,
        interaction,
        orig_positive_u: torch.Tensor,
        orig_positive_i: torch.Tensor,
    ):
        """Collect the evaluation resource from batched eval data and batched model output.
        Args:
            orig_scores_tensor (Torch.Tensor): the output tensor of model with the shape of `(N, )`
            interaction(Interaction): batched eval data.
            orig_positive_u(Torch.Tensor): the row index of positive items for each user.
            orig_positive_i(Torch.Tensor): the positive item id for each user.
        """
        positive_u, positive_i = self.filter.map_user_items(interaction[self.USER_ID], orig_positive_u, interaction[self.ITEM_ID], orig_positive_i, is_ranking_model=self.is_ranking_model)
        # no users after filtering? don't update then
        if positive_u.size(0) == 0:
            return None
        scores_tensor = self.filter.apply_score_filter(orig_scores_tensor)

        if self.model_eval_type != 'ranking':
            score_perms = torch.randperm(scores_tensor.size(1), device=scores_tensor.device)
            scores_perturbed = scores_tensor[:,score_perms]
            # scores_inv = torch.argsort(score_perms)

        if self.register.need("rec.items"):
            # get topk
            _, topk_idx = torch.topk(
                scores_perturbed, max(self.topk), dim=-1
            )  # n_users x k
            topk_idx = score_perms[topk_idx]
            self.data_struct.update_tensor("rec.items", topk_idx)

        if self.register.need("rec.topk"):
            _, topk_idx = torch.topk(
                scores_perturbed, max(self.topk), dim=-1
            )  # n_users x k
            topk_idx = score_perms[topk_idx]
            pos_matrix = torch.zeros_like(scores_tensor, dtype=torch.int)
            pos_matrix[positive_u, positive_i] = 1
            pos_len_list = pos_matrix.sum(dim=1, keepdim=True)
            pos_idx = torch.gather(pos_matrix, dim=1, index=topk_idx)
            result = torch.cat((pos_idx, pos_len_list), dim=1)

            self.data_struct.update_tensor("rec.topk", result)

        if self.register.need("rec.meanrank"):
            raise NotImplementedError("rec.meanrank is not implemented for filtered collector")

        if self.register.need("rec.score"):
            self.data_struct.update_tensor("rec.score", scores_tensor)

        if self.register.need("data.label"):
            self.label_field = self.config["LABEL_FIELD"]
            labels = interaction[self.label_field].to(self.device)
            if self.model_eval_type == 'ranking':
                labels = self.filter.apply_label_filter(labels)

            self.data_struct.update_tensor(
                "data.label", labels
            )

        return True