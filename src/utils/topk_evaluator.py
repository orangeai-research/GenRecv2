# coding: utf-8
# @email: enoche.chow@gmail.com
"""
################################
"""
import os
import numpy as np
import pandas as pd
import torch
from collections import Counter
from utils.metrics import metrics_dict
from torch.nn.utils.rnn import pad_sequence
from utils.utils import get_local_time


# These metrics are typical in topk recommendations
topk_metrics = {metric.lower(): metric for metric in ['Recall', 'Recall2', 'Precision', 'NDCG', 'MAP']}


def cal_gini(d_counter):
    '''
    Improving Item-side Fairness of Multimodal Recommendation via Modality Debiasing
    https://dl.acm.org/doi/pdf/10.1145/3589334.3648156
    '''
    cum_degree = np.cumsum(sorted(np.append(d_counter, 0)))
    sum_degree = cum_degree[-1]
    xarray = np.array(range(0, len(cum_degree))) / (len(cum_degree) - 1)
    yarray = cum_degree / sum_degree
    B = np.trapz(yarray, x=xarray)
    A = 0.5 - B
    G = A / (A + B)
    return G


class TopKEvaluator(object):
    r"""TopK Evaluator is mainly used in ranking tasks. Now, we support six topk metrics which
    contain `'Hit', 'Recall', 'MRR', 'Precision', 'NDCG', 'MAP'`.

    Note:
        The metrics used calculate group-based metrics which considers the metrics scores averaged
        across users. Some of them are also limited to k.

    """

    def __init__(self, config):
        self.config = config
        self.metrics = config['metrics']
        self.topk = config['topk']
        self.save_recom_result = config['save_recommended_topk']
        self.pop_items = config['pop_items'] if 'pop_items' in config else None # popular group
        self.warm_users = config['warm_users'] if 'warm_users' in config else None # warm-niche group
        self.few_shot_users = config['few_shot_users'] if 'few_shot_users' in config else None # few-shot group
        self.pop_mask = None
        self._check_args()

    def collect(self, interaction, scores_tensor, full=False):
        """collect the topk intermediate result of one batch, this function mainly
        implements padding and TopK finding. It is called at the end of each batch

        Args:
            interaction (Interaction): :class:`AbstractEvaluator` of the batch
            scores_tensor (tensor): the tensor of model output with size of `(N, )`
            full (bool, optional): whether it is full sort. Default: False.

        """
        user_len_list = interaction.user_len_list
        if full is True:
            scores_matrix = scores_tensor.view(len(user_len_list), -1)
        else:
            scores_list = torch.split(scores_tensor, user_len_list, dim=0)
            scores_matrix = pad_sequence(scores_list, batch_first=True, padding_value=-np.inf)  # nusers x items

        # get topk
        _, topk_index = torch.topk(scores_matrix, max(self.topk), dim=-1)  # nusers x k

        return topk_index

    def evaluate(self, batch_matrix_list, eval_data, is_test=False, idx=0):
        """calculate the metrics of all batches. It is called at the end of each epoch

        Args:
            batch_matrix_list (list): the results of all batches
            eval_data (Dataset): the class of test data
            is_test: in testing?

        Returns:
            dict: such as ``{'Hit@20': 0.3824, 'Recall@20': 0.0527, 'Hit@10': 0.3153, 'Recall@10': 0.0329}``

        """
        pos_items = eval_data.get_eval_items()
        pos_len_list = eval_data.get_eval_len_list()
        topk_index = torch.cat(batch_matrix_list, dim=0).cpu().numpy()
        # if save recommendation result?
        if self.save_recom_result and is_test:
            dataset_name = self.config['dataset']
            model_name = self.config['model']
            max_k = max(self.topk)
            dir_name = os.path.abspath(self.config['recommend_topk'])
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)
            file_path = os.path.join(dir_name, '{}-{}-idx{}-top{}-{}.csv'.format(
                model_name, dataset_name, idx, max_k, get_local_time()))
            x_df = pd.DataFrame(topk_index)
            x_df.insert(0, 'id', eval_data.get_eval_users())
            x_df.columns = ['id']+['top_'+str(i) for i in range(max_k)]
            x_df = x_df.astype(int)
            x_df.to_csv(file_path, sep='\t', index=False)
        assert len(pos_len_list) == len(topk_index)
        # if recom right?
        bool_rec_matrix = []
        for m, n in zip(pos_items, topk_index):
            bool_rec_matrix.append([True if i in m else False for i in n])
        bool_rec_matrix = np.asarray(bool_rec_matrix)

        # get metrics
        metric_dict = {}
        result_list = self._calculate_metrics(pos_len_list, bool_rec_matrix)
        for metric, value in zip(self.metrics, result_list):
            for k in self.topk:
                key = '{}@{}'.format(metric, k)
                metric_dict[key] = round(value[k - 1], 4)

        # Popularity Group Evaluation
        if self.pop_items is not None and is_test:
            pop_pos_len_list = []
            pop_bool_rec_matrix = []
            niche_pos_len_list = []
            niche_bool_rec_matrix = []

            for i, (gt_items, recommended_items) in enumerate(zip(pos_items, topk_index)):
                # 拆分正样本为「主流物品」和「小众物品」
                gt_pop = [item for item in gt_items if item in self.pop_items]
                gt_niche = [item for item in gt_items if item not in self.pop_items]
                # 主流物品命中标记
                if len(gt_pop) > 0:
                    pop_hits = [True if item in gt_pop else False for item in recommended_items]
                    pop_pos_len_list.append(len(gt_pop))
                    pop_bool_rec_matrix.append(pop_hits)
                 # 小众物品命中标记
                if len(gt_niche) > 0:
                    niche_hits = [True if item in gt_niche else False for item in recommended_items]
                    niche_pos_len_list.append(len(gt_niche))
                    niche_bool_rec_matrix.append(niche_hits)
            # 计算主流物品指标并存入结果字典
            if len(pop_pos_len_list) > 0:
                pop_res = self._calculate_metrics(np.array(pop_pos_len_list), np.array(pop_bool_rec_matrix))
                for metric, value in zip(self.metrics, pop_res):
                    for k in self.topk:
                        metric_name = topk_metrics.get(metric, metric)
                        key = 'Pop_{}@{}'.format(metric_name, k)
                        metric_dict[key] = round(value[k - 1], 4)
            # 计算小众物品指标并存入结果字典
            if len(niche_pos_len_list) > 0:
                niche_res = self._calculate_metrics(np.array(niche_pos_len_list), np.array(niche_bool_rec_matrix))
                for metric, value in zip(self.metrics, niche_res):
                    for k in self.topk:
                        metric_name = topk_metrics.get(metric, metric)
                        key = 'Niche_{}@{}'.format(metric_name, k)
                        metric_dict[key] = round(value[k - 1], 4)

        # Cold-Start User Evaluation
        if self.warm_users is not None and is_test:
            eval_users = eval_data.get_eval_users()
            # Check if eval_users is a tensor or numpy array and convert to list/numpy for faster lookup if needed
            if isinstance(eval_users, torch.Tensor):
                eval_users = eval_users.cpu().numpy()
            
            # Identify Warm Users (Present in warm_users set)
            is_warm_mask = np.array([u in self.warm_users for u in eval_users])
            # Identify Cold Users (NOT in warm_users set -> covers count<=5 and count==0)
            is_cold_mask = ~is_warm_mask
            
            if is_cold_mask.any():
                cold_pos_len = np.array(pos_len_list)[is_cold_mask]
                cold_bool_rec = bool_rec_matrix[is_cold_mask]
                
                if len(cold_pos_len) > 0:
                    cold_res = self._calculate_metrics(cold_pos_len, cold_bool_rec)
                    for metric, value in zip(self.metrics, cold_res):
                        for k in self.topk:
                            metric_name = topk_metrics.get(metric, metric)
                            key = 'Cold_{}@{}'.format(metric_name, k)
                            metric_dict[key] = round(value[k - 1], 4)

            # Warm users metrics
            if is_warm_mask.any():
                warm_pos_len = np.array(pos_len_list)[is_warm_mask]
                warm_bool_rec = bool_rec_matrix[is_warm_mask]
                
                if len(warm_pos_len) > 0:
                    warm_res = self._calculate_metrics(warm_pos_len, warm_bool_rec)
                    for metric, value in zip(self.metrics, warm_res):
                        for k in self.topk:
                            metric_name = topk_metrics.get(metric, metric)
                            key = 'Warm_{}@{}'.format(metric_name, k)
                            metric_dict[key] = round(value[k - 1], 4)

        # Few-Shot User Evaluation
        if self.few_shot_users is not None and is_test:
            eval_users = eval_data.get_eval_users()
            if isinstance(eval_users, torch.Tensor):
                eval_users = eval_users.cpu().numpy()
            
            is_few_shot_mask = np.array([u in self.few_shot_users for u in eval_users])
            
            if is_few_shot_mask.any():
                few_shot_pos_len = np.array(pos_len_list)[is_few_shot_mask]
                few_shot_bool_rec = bool_rec_matrix[is_few_shot_mask]
                
                if len(few_shot_pos_len) > 0:
                    few_shot_res = self._calculate_metrics(few_shot_pos_len, few_shot_bool_rec)
                    for metric, value in zip(self.metrics, few_shot_res):
                        for k in self.topk:
                            metric_name = topk_metrics.get(metric, metric)
                            key = 'FewShot_{}@{}'.format(metric_name, k)
                            metric_dict[key] = round(value[k - 1], 4)

        # Diversity and Tail metrics
        '''
        1.Coverage@k（覆盖率）：
            计算逻辑：推荐列表中出现的唯一物品数 / 数据集中总物品数。
            指标意义：值越大，说明模型推荐的物品越丰富，多样性越强，避免推荐结果过于集中。
        2.Gini@k（基尼系数）：
            计算逻辑：基于推荐物品的交互频次分布计算，取值范围 [0,1]。
            指标意义：越接近 0，说明推荐物品的流行度分布越均匀，流行度偏差越小；越接近 1，说明模型过度推荐主流物品，偏差严重。
        3.Tail%@k（长尾占比）：
            计算逻辑：推荐列表中长尾物品（非主流物品）的数量 / 推荐物品总数量。
            指标意义：值越大，说明模型对长尾 / 小众物品的推荐占比越高，去偏效果越好。
        '''
        if is_test:
            item_num = eval_data.dataset.item_num
            
            # Precompute pop_mask if needed
            if self.pop_items is not None and self.pop_mask is None:
                self.pop_mask = np.zeros(item_num, dtype=bool)
                # pop_items is a set of item IDs
                # Ensure indices are within bounds
                pop_indices = [i for i in self.pop_items if i < item_num]
                self.pop_mask[pop_indices] = True

            for k in self.topk:
                # 提取每个用户的前 k 个推荐物品
                # topk_index is (n_users, max_k)
                # We need the first k items for each user
                current_k_items = topk_index[:, :k]
                rec_items = current_k_items.flatten()
                
                # 1. 计算覆盖率（Coverage@k）：推荐列表覆盖的唯一物品数 / 总物品数
                # Coverage & Gini
                rec_count = np.bincount(rec_items, minlength=item_num)
                
                # Coverage
                coverage = np.count_nonzero(rec_count) / item_num
                metric_dict[f'Coverage@{k}'] = round(coverage, 4)
                
                 # 2. 计算基尼系数（Gini@k）：衡量推荐物品流行度分布均匀性（越小偏差越小）
                # Gini
                sorted_counts = np.sort(rec_count)
                n = item_num
                cum_counts = np.cumsum(sorted_counts)
                sum_counts = cum_counts[-1]
                if sum_counts > 0:
                    index = np.arange(1, n + 1)
                    gini = (2 * np.sum(index * sorted_counts)) / (n * sum_counts) - (n + 1) / n
                    metric_dict[f'Gini@{k}'] = round(gini, 4)
                else:
                    metric_dict[f'Gini@{k}'] = 0.0

                # Coverage2 & Gini2 (from new_gini.py logic)
                # Note: new_gini logic calculates gini based on active items in recommendation list, 
                # and coverage based on total dataset items.
                num_count = Counter(rec_items.tolist())
                num_list = [i[1] for i in num_count.items()]
                if len(num_list) > 0:
                    gini2 = cal_gini(num_list)
                    metric_dict[f'Gini2@{k}'] = round(gini2, 4)
                    
                    coverage2 = len(num_list) / item_num
                    metric_dict[f'Coverage2@{k}'] = round(coverage2, 4)
                else:
                    metric_dict[f'Gini2@{k}'] = 0.0
                    metric_dict[f'Coverage2@{k}'] = 0.0

                 # 3. 计算长尾物品占比（Tail%@k）：推荐列表中长尾物品数 / 总推荐物品数
                # Tail Percentage
                if self.pop_mask is not None:
                    is_pop = self.pop_mask[rec_items]
                    # Tail items are those NOT in pop_items (False in pop_mask)
                    tail_count = (~is_pop).sum()
                    tail_pct = tail_count / len(rec_items)
                    metric_dict[f'Tail%@{k}'] = round(tail_pct, 4)

        return metric_dict

    def _check_args(self):
        # Check metrics
        if isinstance(self.metrics, (str, list)):
            if isinstance(self.metrics, str):
                self.metrics = [self.metrics]
        else:
            raise TypeError('metrics must be str or list')

        # Convert metric to lowercase
        for m in self.metrics:
            if m.lower() not in topk_metrics:
                raise ValueError("There is no user grouped topk metric named {}!".format(m))
        self.metrics = [metric.lower() for metric in self.metrics]

        # Check topk:
        if isinstance(self.topk, (int, list)):
            if isinstance(self.topk, int):
                self.topk = [self.topk]
            for topk in self.topk:
                if topk <= 0:
                    raise ValueError(
                        'topk must be a positive integer or a list of positive integers, but get `{}`'.format(topk))
        else:
            raise TypeError('The topk must be a integer, list')

    def _calculate_metrics(self, pos_len_list, topk_index):
        """integrate the results of each batch and evaluate the topk metrics by users

        Args:
            pos_len_list (list): a list of users' positive items
            topk_index (np.ndarray): a matrix which contains the index of the topk items for users
        Returns:
            np.ndarray: a matrix which contains the metrics result
        """
        result_list = []
        for metric in self.metrics:
            metric_fuc = metrics_dict[metric.lower()]
            result = metric_fuc(topk_index, pos_len_list)
            result_list.append(result)
        return np.stack(result_list, axis=0)

    def __str__(self):
        mesg = 'The TopK Evaluator Info:\n' + '\tMetrics:[' + ', '.join(
            [topk_metrics[metric.lower()] for metric in self.metrics]) \
               + '], TopK:[' + ', '.join(map(str, self.topk)) + ']'
        return mesg
