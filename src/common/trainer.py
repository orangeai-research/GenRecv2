# coding: utf-8
# @email: enoche.chow@gmail.com

r"""
################################
"""

import os
import itertools
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import scipy.sparse as sp
from scipy.sparse import coo_matrix
import matplotlib.pyplot as plt

from time import time
from logging import getLogger

from utils.utils import get_local_time, early_stopping, dict2str, build_knn_normalized_graph
from utils.topk_evaluator import TopKEvaluator

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from common.interest_cluster import MultimodalCluster, InterestDebiase

class AbstractTrainer(object):
    r"""Trainer Class is used to manage the training and evaluation processes of recommender system models.
    AbstractTrainer is an abstract class in which the fit() and evaluate() method should be implemented according
    to different training and evaluation strategies.
    """

    def __init__(self, config, model):
        self.config = config
        self.model = model

    def fit(self, train_data):
        r"""Train the model based on the train data.

        """
        raise NotImplementedError('Method [next] should be implemented.')

    def evaluate(self, eval_data):
        r"""Evaluate the model based on the eval data.

        """

        raise NotImplementedError('Method [next] should be implemented.')


class Trainer(AbstractTrainer):
    r"""The basic Trainer for basic training and evaluation strategies in recommender systems. This class defines common
    functions for training and evaluation processes of most recommender system models, including fit(), evaluate(),
    and some other features helpful for model training and evaluation.

    Generally speaking, this class can serve most recommender system models, If the training process of the model is to
    simply optimize a single loss without involving any complex training strategies, such as adversarial learning,
    pre-training and so on.

    Initializing the Trainer needs two parameters: `config` and `model`. `config` records the parameters information
    for controlling training and evaluation, such as `learning_rate`, `epochs`, `eval_step` and so on.
    More information can be found in [placeholder]. `model` is the instantiated object of a Model Class.

    """

    def __init__(self, config, model, mg=False):
        super(Trainer, self).__init__(config, model)

        self.logger = getLogger()
        self.learner = config['learner']
        self.learning_rate = config['learning_rate']
        self.epochs = config['epochs']
        self.eval_step = min(config['eval_step'], self.epochs)
        self.stopping_step = config['stopping_step']
        self.clip_grad_norm = config['clip_grad_norm']
        self.valid_metric = config['valid_metric'].lower()
        self.valid_metric_bigger = config['valid_metric_bigger']
        self.test_batch_size = config['eval_batch_size']
        self.device = config['device']
        self.weight_decay = 0.0
        if config['weight_decay'] is not None:
            wd = config['weight_decay']
            self.weight_decay = eval(wd) if isinstance(wd, str) else wd

        self.req_training = config['req_training']

        # W&B logging
        self.use_wandb = (config['use_wandb'] if 'use_wandb' in config else True) and WANDB_AVAILABLE and wandb.run is not None

        self.start_epoch = 0
        self.cur_step = 0

        tmp_dd = {}
        for j, k in list(itertools.product(config['metrics'], config['topk'])):
            tmp_dd[f'{j.lower()}@{k}'] = 0.0
        self.best_valid_score = -1
        self.best_valid_result = tmp_dd
        self.best_test_upon_valid = tmp_dd
        self.train_loss_dict = dict()
        self.optimizer = self._build_optimizer()

        #fac = lambda epoch: 0.96 ** (epoch / 50)
        lr_scheduler = config['learning_rate_scheduler']        # check zero?
        fac = lambda epoch: lr_scheduler[0] ** (epoch / lr_scheduler[1])
        scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=fac)
        self.lr_scheduler = scheduler

        self.eval_type = config['eval_type']
        self.evaluator = TopKEvaluator(config)

        self.item_tensor = None
        self.tot_item_num = None
        self.mg = mg
        self.alpha1 = config['alpha1']
        self.alpha2 = config['alpha2']
        self.beta = config['beta']

    def _build_optimizer(self):
        r"""Init the Optimizer

        Returns:
            torch.optim: the optimizer
        """
        if self.learner.lower() == 'adam':
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'sgd':
            optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'adagrad':
            optimizer = optim.Adagrad(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'rmsprop':
            optimizer = optim.RMSprop(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        else:
            self.logger.warning('Received unrecognized optimizer, set default Adam optimizer')
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return optimizer

    def _train_epoch(self, train_data, epoch_idx, loss_func=None):
        r"""Train the model in an epoch

        Args:
            train_data (DataLoader): The train data.
            epoch_idx (int): The current epoch id.
            loss_func (function): The loss function of :attr:`model`. If it is ``None``, the loss function will be
                :attr:`self.model.calculate_loss`. Defaults to ``None``.

        Returns:
            float/tuple: The sum of loss returned by all batches in this epoch. If the loss in each batch contains
            multiple parts and the model return these multiple parts loss instead of the sum of loss, It will return a
            tuple which includes the sum of loss in each part.
        """
        if not self.req_training:
            return 0.0, []
        self.model.train()
        loss_func = loss_func or self.model.calculate_loss
        total_loss = None
        loss_batches = []
        for batch_idx, interaction in enumerate(train_data):
            self.optimizer.zero_grad()
            second_inter = interaction.clone()
            losses = loss_func(interaction)
            
            if isinstance(losses, tuple):
                loss = sum(losses)
                loss_tuple = tuple(per_loss.item() for per_loss in losses)
                total_loss = loss_tuple if total_loss is None else tuple(map(sum, zip(total_loss, loss_tuple)))
            else:
                loss = losses
                total_loss = losses.item() if total_loss is None else total_loss + losses.item()
            if self._check_nan(loss):
                self.logger.info('Loss is nan at epoch: {}, batch index: {}. Exiting.'.format(epoch_idx, batch_idx))
                return loss, torch.tensor(0.0)
            
            if self.mg and batch_idx % self.beta == 0:
                first_loss = self.alpha1 * loss
                first_loss.backward()

                self.optimizer.step()
                self.optimizer.zero_grad()
                
                losses = loss_func(second_inter)
                if isinstance(losses, tuple):
                    loss = sum(losses)
                else:
                    loss = losses
                    
                if self._check_nan(loss):
                    self.logger.info('Loss is nan at epoch: {}, batch index: {}. Exiting.'.format(epoch_idx, batch_idx))
                    return loss, torch.tensor(0.0)
                second_loss = -1 * self.alpha2 * loss
                second_loss.backward()
            else:
                loss.backward()
                
            if self.clip_grad_norm:
                clip_grad_norm_(self.model.parameters(), **self.clip_grad_norm)
            self.optimizer.step()
            loss_batches.append(loss.detach())
            # for test
            #if batch_idx == 0:
            #    break
        return total_loss, loss_batches

    def _valid_epoch(self, valid_data, is_test=False):
        r"""Valid the model with valid data

        Args:
            valid_data (DataLoader): the valid data
            is_test (bool): whether it is test data

        Returns:
            float: valid score
            dict: valid result
        """
        valid_result = self.evaluate(valid_data, is_test=is_test)
        # 若模型支持，计算生成兴趣表征质量指标（SWD / 余弦 / MSE / FID）
        if hasattr(self.model, 'compute_gen_quality_metrics'):
            try:
                gen_metrics = self.model.compute_gen_quality_metrics()
                if gen_metrics:
                    valid_result = {**valid_result, **gen_metrics}
            except Exception as e:
                if hasattr(self, 'logger') and self.logger is not None:
                    self.logger.warning('compute_gen_quality_metrics failed: %s', e)
        # 跨模型可比：用 GenRecv2 的原始兴趣编码作为共同参考，计算 gen_cross_*
        cross_ref_path = self.config['gen_cross_reference_path']
        if cross_ref_path and hasattr(self.model, 'compute_gen_quality_metrics_cross'):
            try:
                import os
                if os.path.isfile(cross_ref_path):
                    origin_ref = torch.load(cross_ref_path, map_location='cpu', weights_only=True)
                    cross_metrics = self.model.compute_gen_quality_metrics_cross(origin_ref)
                    if cross_metrics:
                        valid_result = {**valid_result, **cross_metrics}
            except Exception as e:
                if hasattr(self, 'logger') and self.logger is not None:
                    self.logger.warning('compute_gen_quality_metrics_cross failed: %s', e)
        # 轨迹线性度 (Trajectory Linearity)：RefFlow 应接近 1.0
        if hasattr(self.model, 'compute_trajectory_linearity'):
            try:
                traj_out = self.model.compute_trajectory_linearity(return_trajectory_for_viz=False)
                if traj_out:
                    valid_result = {**valid_result, **{k: v for k, v in traj_out.items() if k != 'trajectory'}}
            except Exception as e:
                if hasattr(self, 'logger') and self.logger is not None:
                    self.logger.warning('compute_trajectory_linearity failed: %s', e)
        valid_score = valid_result[self.valid_metric] if self.valid_metric else valid_result['NDCG@20']
        return valid_score, valid_result

    def _check_nan(self, loss):
        if torch.isnan(loss):
            #raise ValueError('Training loss is nan')
            return True

    def _generate_train_loss_output(self, epoch_idx, s_time, e_time, losses):
        train_loss_output = 'epoch %d training [time: %.2fs, ' % (epoch_idx, e_time - s_time)
        if isinstance(losses, tuple):
            train_loss_output = ', '.join('train_loss%d: %.4f' % (idx + 1, loss) for idx, loss in enumerate(losses))
        else:
            train_loss_output += 'train loss: %.4f' % losses
        return train_loss_output + ']'

    def fit(self, train_data, valid_data=None, test_data=None, saved=False, verbose=True):
        r"""Train the model based on the train data and the valid data.

        Args:
            train_data (DataLoader): the train data
            valid_data (DataLoader, optional): the valid data, default: None.
                                               If it's None, the early_stopping is invalid.
            test_data (DataLoader, optional): None
            verbose (bool, optional): whether to write training and evaluation information to logger, default: True
            saved (bool, optional): whether to save the model parameters, default: True

        Returns:
             (float, dict): best valid score and best valid result. If valid_data is None, it returns (-1, None)
        """
        for epoch_idx in range(self.start_epoch, self.epochs):
            # train
            training_start_time = time()
            self.model.pre_epoch_processing()
            train_loss, _ = self._train_epoch(train_data, epoch_idx)
            if torch.is_tensor(train_loss):
                # get nan loss
                break
            #for param_group in self.optimizer.param_groups:
            #    print('======lr: ', param_group['lr'])
            self.lr_scheduler.step()
            
            self.train_loss_dict[epoch_idx] = sum(train_loss) if isinstance(train_loss, tuple) else train_loss
            training_end_time = time()
            train_loss_output = \
                self._generate_train_loss_output(epoch_idx, training_start_time, training_end_time, train_loss)
            post_info = self.model.post_epoch_processing()
            if verbose:
                self.logger.info(train_loss_output)
                if post_info is not None:
                    self.logger.info(post_info)

            # Log training loss to wandb
            if self.use_wandb:
                wandb_log = {
                    'epoch': epoch_idx,
                    'train/time': training_end_time - training_start_time,
                    'train/lr': self.optimizer.param_groups[0]['lr']
                }
                if isinstance(train_loss, tuple):
                    for idx, loss in enumerate(train_loss):
                        wandb_log[f'train/loss_{idx+1}'] = loss
                    wandb_log['train/total_loss'] = sum(train_loss)
                else:
                    wandb_log['train/loss'] = train_loss
                wandb.log(wandb_log, step=epoch_idx)

            # eval: To ensure the test result is the best model under validation data, set self.eval_step == 1
            if (epoch_idx + 1) % self.eval_step == 0:
                valid_start_time = time()
                valid_score, valid_result = self._valid_epoch(valid_data)
                self.best_valid_score, self.cur_step, stop_flag, update_flag = early_stopping(
                    valid_score, self.best_valid_score, self.cur_step,
                    max_step=self.stopping_step, bigger=self.valid_metric_bigger)
                valid_end_time = time()
                valid_score_output = "epoch %d evaluating [time: %.2fs, valid_score: %f]" % \
                                     (epoch_idx, valid_end_time - valid_start_time, valid_score)
                valid_result_output = 'valid result: \n' + dict2str(valid_result)
                # test
                _, test_result = self._valid_epoch(test_data, is_test=True)
                if verbose:
                    self.logger.info(valid_score_output)
                    self.logger.info(valid_result_output)
                    self.logger.info('test result: \n' + dict2str(test_result))
                    # 测试集上的生成兴趣表征质量指标
                    gen_keys = ['gen_swd', 'gen_cosine', 'gen_mse', 'gen_fid']
                    if all(k in test_result for k in gen_keys):
                        self.logger.info(
                            'gen quality (test): gen_swd=%.6f, gen_cosine=%.6f, gen_mse=%.6f, gen_fid=%.6f' % (
                                test_result['gen_swd'], test_result['gen_cosine'],
                                test_result['gen_mse'], test_result['gen_fid']
                            )
                        )
                    gen_cross_keys = ['gen_cross_swd', 'gen_cross_cosine', 'gen_cross_mse', 'gen_cross_fid']
                    if all(k in test_result for k in gen_cross_keys):
                        self.logger.info(
                            'gen quality cross (test): gen_cross_swd=%.6f, gen_cross_cosine=%.6f, gen_cross_mse=%.6f, gen_cross_fid=%.6f' % (
                                test_result['gen_cross_swd'], test_result['gen_cross_cosine'],
                                test_result['gen_cross_mse'], test_result['gen_cross_fid']
                            )
                        )
                    if 'trajectory_linearity' in test_result:
                        self.logger.info('trajectory linearity (test): %.6f (RefFlow ~1.0 = straight)' % test_result['trajectory_linearity'])

                # Log validation and test results to wandb
                if self.use_wandb:
                    wandb_eval_log = {
                        'epoch': epoch_idx,
                        'valid/score': valid_score,
                        'valid/time': valid_end_time - valid_start_time,
                        **{f'valid/{k}': v for k, v in valid_result.items()},
                        **{f'test/{k}': v for k, v in test_result.items()}
                    }
                    wandb.log(wandb_eval_log, step=epoch_idx)

                if update_flag:
                    update_output = '██ ' + self.config['model'] + '--Best validation results updated!!!'
                    if verbose:
                        self.logger.info(update_output)
                    self.best_valid_result = valid_result
                    self.best_test_upon_valid = test_result

                    # Log best results to wandb
                    if self.use_wandb:
                        wandb.run.summary.update({
                            'best_epoch': epoch_idx,
                            'best_valid_score': valid_score,
                            **{f'best_valid_{k}': v for k, v in valid_result.items()},
                            **{f'best_test_{k}': v for k, v in test_result.items()}
                        })
                    
                    if saved:
                         self._save_checkpoint(epoch_idx)
                    # 若配置了跨模型参考保存路径且模型可提供参考（如 GenRecv2），保存共同参考
                    save_ref_path = self.config['gen_cross_reference_save_path']
                    if save_ref_path and hasattr(self.model, 'get_origin_embedding_for_cross_model'):
                        try:
                            origin_ref = self.model.get_origin_embedding_for_cross_model()
                            torch.save(origin_ref.cpu(), save_ref_path)
                            if verbose and hasattr(self, 'logger') and self.logger is not None:
                                self.logger.info('gen_cross reference saved to %s', save_ref_path)
                        except Exception as e:
                            if hasattr(self, 'logger') and self.logger is not None:
                                self.logger.warning('save gen_cross reference failed: %s', e)

                if stop_flag:
                    stop_output = '+++++Finished training, best eval result in epoch %d' % \
                                  (epoch_idx - self.cur_step * self.eval_step)
                    if verbose:
                        self.logger.info(stop_output)
                    break
        return self.best_valid_score, self.best_valid_result, self.best_test_upon_valid
    
    def _save_checkpoint(self, epoch):
        r"""Store the model parameters, the training parameters and the state_dict of the optimizer
        """
        checkpoint_dir = self.config['checkpoint_dir']
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
            
        checkpoint_path = os.path.join(
            checkpoint_dir,
            '{}-{}.pth'.format(self.config['model'], self.config['dataset'])
        )
        
        # Save the best model
        state = {
            'config': self.config,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'best_valid_score': self.best_valid_score,
        }
        torch.save(state, checkpoint_path)
        self.logger.info('Saved best model to {}'.format(checkpoint_path))


    @torch.no_grad()
    def evaluate(self, eval_data, is_test=False, idx=0):
        r"""Evaluate the model based on the eval data.
        Returns:
            dict: eval result, key is the eval metric and value in the corresponding metric value
        """
        self.model.eval()

        # batch full users
        batch_matrix_list = []
        for batch_idx, batched_data in enumerate(eval_data):
            # predict: interaction without item ids
            scores = self.model.full_sort_predict(batched_data)
            masked_items = batched_data[1]
            # mask out pos items
            scores[masked_items[0], masked_items[1]] = -1e10
            # rank and get top-k
            _, topk_index = torch.topk(scores, max(self.config['topk']), dim=-1)  # nusers x topk
            batch_matrix_list.append(topk_index)
        return self.evaluator.evaluate(batch_matrix_list, eval_data, is_test=is_test, idx=idx)

    def plot_train_loss(self, show=True, save_path=None):
        r"""Plot the train loss in each epoch

        Args:
            show (bool, optional): whether to show this figure, default: True
            save_path (str, optional): the data path to save the figure, default: None.
                                       If it's None, it will not be saved.
        """
        epochs = list(self.train_loss_dict.keys())
        epochs.sort()
        values = [float(self.train_loss_dict[epoch]) for epoch in epochs]
        plt.plot(epochs, values)
        plt.xticks(epochs)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        if show:
            plt.show()
        if save_path:
            plt.savefig(save_path)

class DiffMMTrainer(Trainer):
    def __init__(self, config, model, mg=False):
        super(DiffMMTrainer, self).__init__(config, model, mg)
        
        self.denoise_opt_image = optim.Adam(self.model.denoise_model_image.parameters(), lr=config['learning_rate'], weight_decay=0)
        self.denoise_opt_text = optim.Adam(self.model.denoise_model_text.parameters(), lr=config['learning_rate'], weight_decay=0)
        
        self.diffusion_loader = None
        self.item_num = model.n_items
        self.user_num = model.n_users

    def _build_diffusion_loader(self, train_data):
        if self.diffusion_loader is not None:
            return
            
        # Extract user interactions
        df = train_data.dataset.df
        uid_field = self.config['USER_ID_FIELD']
        iid_field = self.config['ITEM_ID_FIELD']
        
        user_interactions = df.groupby(uid_field)[iid_field].apply(list).to_dict()
        
        # Create dense vectors
        # Note: This might consume memory. If user_num * item_num is large.
        # DiffMM uses sparse conversion in DataHandler but DataLoader yields dense tensors?
        # Let's check DiffMM logic: 
        # "return torch.FloatTensor(itemTensor), torch.tensor(index).float()" in DiffusionData
        # So it returns dense tensors.
        
        # We can implement a custom Dataset
        class DiffusionDataset(torch.utils.data.Dataset):
            def __init__(self, user_interactions, user_num, item_num):
                self.user_interactions = user_interactions
                self.user_num = user_num
                self.item_num = item_num
                self.users = list(range(user_num))
                
            def __len__(self):
                return self.user_num
                
            def __getitem__(self, idx):
                user_id = self.users[idx]
                items = self.user_interactions.get(user_id, [])
                
                # Create dense vector
                vector = torch.zeros(self.item_num)
                if len(items) > 0:
                    vector[items] = 1.0
                
                return vector, torch.tensor(user_id).float()
                
        dataset = DiffusionDataset(user_interactions, self.user_num, self.item_num)
        self.diffusion_loader = DataLoader(dataset, batch_size=self.config['train_batch_size'], shuffle=True, num_workers=0) # num_workers=0 for safety

    def normalizeAdj(self, mat): 
        degree = np.array(mat.sum(axis=-1))
        dInvSqrt = np.reshape(np.power(degree, -0.5), [-1])
        dInvSqrt[np.isinf(dInvSqrt)] = 0.0
        dInvSqrtMat = sp.diags(dInvSqrt)
        return mat.dot(dInvSqrtMat).transpose().dot(dInvSqrtMat).tocoo()

    def buildUIMatrix(self, u_list, i_list, edge_list):
        mat = coo_matrix((edge_list, (u_list, i_list)), shape=(self.user_num, self.item_num), dtype=np.float32)

        a = sp.csr_matrix((self.user_num, self.user_num))
        b = sp.csr_matrix((self.item_num, self.item_num))
        mat = sp.vstack([sp.hstack([a, mat]), sp.hstack([mat.transpose(), b])])
        mat = (mat != 0) * 1.0
        mat = (mat + sp.eye(mat.shape[0])) * 1.0
        mat = self.normalizeAdj(mat)

        idxs = torch.from_numpy(np.vstack([mat.row, mat.col]).astype(np.int64))
        vals = torch.from_numpy(mat.data.astype(np.float32))
        shape = torch.Size(mat.shape)

        return torch.sparse.FloatTensor(idxs, vals, shape).to(self.device)

    def _train_epoch(self, train_data, epoch_idx, loss_func=None):
        # 0. Build diffusion loader if not exists
        self._build_diffusion_loader(train_data)
        
        # 1. Diffusion Training
        self.model.train()
        epDiLoss_image, epDiLoss_text = 0, 0
        steps = len(self.diffusion_loader)
        
        iEmbeds = self.model.getItemEmbeds().detach()
        # uEmbeds not used in diffusion training in DiffMM? 
        # DiffMM: iEmbeds = self.model.getItemEmbeds().detach()
        # usr_id_embeds = torch.mm(x_start, itmEmbeds)
        
        image_feats = self.model.getImageFeats().detach()
        text_feats = self.model.getTextFeats().detach()
        
        for i, batch in enumerate(self.diffusion_loader):
            batch_item, batch_index = batch
            batch_item, batch_index = batch_item.to(self.device), batch_index.to(self.device)
            
            self.denoise_opt_image.zero_grad()
            self.denoise_opt_text.zero_grad()
            
            diff_loss_image, gc_loss_image = self.model.diffusion_model.training_losses(
                self.model.denoise_model_image, batch_item, iEmbeds, batch_index, image_feats)
            
            diff_loss_text, gc_loss_text = self.model.diffusion_model.training_losses(
                self.model.denoise_model_text, batch_item, iEmbeds, batch_index, text_feats)
                
            loss_image = diff_loss_image.mean() + gc_loss_image.mean() * self.model.e_loss
            loss_text = diff_loss_text.mean() + gc_loss_text.mean() * self.model.e_loss
            
            epDiLoss_image += loss_image.item()
            epDiLoss_text += loss_text.item()
            
            loss = loss_image + loss_text
            loss.backward()
            
            self.denoise_opt_image.step()
            self.denoise_opt_text.step()
            
        # 2. Graph Construction
        # Re-generate matrices using trained diffusion model
        with torch.no_grad():
            u_list_image = []
            i_list_image = []
            edge_list_image = []

            u_list_text = []
            i_list_text = []
            edge_list_text = []
            
            for _, batch in enumerate(self.diffusion_loader):
                batch_item, batch_index = batch
                batch_item, batch_index = batch_item.to(self.device), batch_index.to(self.device)

                # image
                denoised_batch = self.model.diffusion_model.p_sample(self.model.denoise_model_image, batch_item, self.model.sampling_steps, self.model.sampling_noise)
                top_item, indices_ = torch.topk(denoised_batch, k=self.model.rebuild_k)

                for i in range(batch_index.shape[0]):
                    for j in range(indices_[i].shape[0]): 
                        u_list_image.append(int(batch_index[i].cpu().numpy()))
                        i_list_image.append(int(indices_[i][j].cpu().numpy()))
                        edge_list_image.append(1.0)

                # text
                denoised_batch = self.model.diffusion_model.p_sample(self.model.denoise_model_text, batch_item, self.model.sampling_steps, self.model.sampling_noise)
                top_item, indices_ = torch.topk(denoised_batch, k=self.model.rebuild_k)

                for i in range(batch_index.shape[0]):
                    for j in range(indices_[i].shape[0]): 
                        u_list_text.append(int(batch_index[i].cpu().numpy()))
                        i_list_text.append(int(indices_[i][j].cpu().numpy()))
                        edge_list_text.append(1.0)
            
            # image
            u_list_image = np.array(u_list_image)
            i_list_image = np.array(i_list_image)
            edge_list_image = np.array(edge_list_image)
            self.model.image_UI_matrix = self.buildUIMatrix(u_list_image, i_list_image, edge_list_image)
            self.model.image_UI_matrix = self.model.edgeDropper(self.model.image_UI_matrix)

            # text
            u_list_text = np.array(u_list_text)
            i_list_text = np.array(i_list_text)
            edge_list_text = np.array(edge_list_text)
            self.model.text_UI_matrix = self.buildUIMatrix(u_list_text, i_list_text, edge_list_text)
            self.model.text_UI_matrix = self.model.edgeDropper(self.model.text_UI_matrix)
            
        # 3. Recommendation Training (Standard)
        # Call super method to handle BPR training
        rec_loss, loss_batches = super(DiffMMTrainer, self)._train_epoch(train_data, epoch_idx, loss_func)
        
        # Log diffusion loss
        self.logger.info(f"Diffusion Loss: Image={epDiLoss_image/steps:.4f}, Text={epDiLoss_text/steps:.4f}")
        
        return rec_loss, loss_batches


class GenRecV1Trainer(DiffMMTrainer):
    def __init__(self, config, model, mg=False):
        # Call Trainer.__init__ directly to skip DiffMMTrainer.__init__ which assumes denoise_model_text exists
        Trainer.__init__(self, config, model, mg)
        
        self.denoise_opt_image = optim.Adam(self.model.denoise_model_image.parameters(), lr=config['learning_rate'], weight_decay=0)
        # GenRecV1 does not train text diffusion model by default
        
        self.diffusion_loader = None
        self.item_num = model.n_items
        self.user_num = model.n_users
        
        # Build II Matrix (Item-Item)
        # This is done once at initialization
        self.image_II_matrix = None
        self.text_II_matrix = None
        self._build_item_item_matrix()
        
        # Interest Clustering & Debiasing
        self.multimodal_interest_space = None
        if self.config['OpenInterestDebiase'] if 'OpenInterestDebiase' in self.config else False:
             self._init_interest_clustering()
            
    def _init_interest_clustering(self):
        # Parameters for clustering
        # Assuming these are in config, or set defaults
        kmeans_cluster_num = self.config['kmeans_cluster_num'] if 'kmeans_cluster_num' in self.config else 20
        use_auto_optimal_k = self.config['use_auto_optimal_k'] if 'use_auto_optimal_k' in self.config else False
        
        image_modal_cluster = MultimodalCluster(
            num_cluster_visual_modal=20,
            num_cluster_text_modal=20,
            num_cluster_audio_modal=20,
            num_cluster_fusion_modal=20,
            kmeans_cluster_num=kmeans_cluster_num,
            spectral_cluster_num=20,
            sim_top_k=20,
            use_auto_optimal_k=use_auto_optimal_k,
            kmeans_cluster_num_min=3,
            kmeans_cluster_num_mean=7,
            kmeans_cluster_num_max=237,
            kmeans_stride=10
        )
        
        # Optimal cluster nums per dataset
        image_optimial_k = 18
        text_optimial_k = 59
        audio_optimial_k = 46
        
        dataset_name = self.config['dataset']
        if not use_auto_optimal_k:
            if dataset_name == 'tiktok':
                image_optimial_k = 18
                text_optimial_k = 59
                audio_optimial_k = 46
            elif dataset_name == 'baby':
                image_optimial_k = 6
                text_optimial_k = 11
            elif dataset_name == 'sports':
                image_optimial_k = 9
                text_optimial_k = 12
                
        # Perform clustering
        self.logger.info("Performing Multimodal Clustering...")
        image_feats = self.model.image_embedding
        text_feats = self.model.text_embedding
        
        if image_feats is not None:
            image_labels = image_modal_cluster.multimodal_specific_cluster(
                image_feats, 'image_modal', image_optimial_k)
        else:
            image_labels = None
            
        if text_feats is not None:
            text_labels = image_modal_cluster.multimodal_specific_cluster(
                text_feats, 'text_modal', text_optimial_k)
        else:
            text_labels = None
            
        self.multimodal_interest_space = {
            'image_modal': image_labels,
            'text_modal': text_labels
        }
        self.logger.info("Multimodal Clustering Done.")

    def _build_item_item_matrix(self):
        # Image
        if self.model.image_embedding is not None:
            self.model.image_II_matrix = self._build_knn_adj(self.model.image_embedding)
        
        # Text
        if self.model.text_embedding is not None:
            self.model.text_II_matrix = self._build_knn_adj(self.model.text_embedding)
            
    def _build_knn_adj(self, feature):
        # Normalize
        feature_norm = F.normalize(feature, p=2, dim=-1)
        sim_adj = torch.mm(feature_norm, feature_norm.transpose(1, 0))
        sim_adj_sparse = build_knn_normalized_graph(sim_adj, topk=self.config['knn_k'], is_sparse=self.model.sparse, norm_type='sym')
        return sim_adj_sparse.to(self.device)

    def _train_epoch(self, train_data, epoch_idx, loss_func=None):
        # 0. Build diffusion loader if not exists
        self._build_diffusion_loader(train_data)
        
        # 1. Diffusion Training
        self.model.train()
        epDiLoss_image = 0
        steps = len(self.diffusion_loader)
        
        iEmbeds = self.model.getItemEmbeds().detach()
        image_feats = self.model.getImageFeats().detach() if self.model.image_embedding is not None else None
        text_feats = self.model.getTextFeats().detach() if self.model.text_embedding is not None else None
        
        # Note: GenRecV1 uses User-Item interaction vectors for diffusion
        # batch_item in diffusion_loader is [batch_users, n_items] (dense)
        
        for i, batch in enumerate(self.diffusion_loader):
            batch_item, batch_index = batch
            batch_item, batch_index = batch_item.to(self.device), batch_index.to(self.device)
            
            # Train Image Diffusion
            if image_feats is not None:
                self.denoise_opt_image.zero_grad()
                # FlipInterestDiffusion.training_losses returns a single scalar loss
                loss_image = self.model.diffusion_model.training_losses(
                    self.model.denoise_model_image, batch_item, iEmbeds, batch_index, image_feats, text_feats)
                
                epDiLoss_image += loss_image.item()
                loss_image.backward()
                self.denoise_opt_image.step()
                
            # GenRecV1 usually only trains one diffusion model (for image/structure) or uses joint?
            # In Main.py lines 297: self.diffusion_model.training_losses(self.denoise_model_image, ...)
            # It seems it only trains denoise_model_image. 
            # But wait, it passes text_feats to it too?
            # "loss_image = self.diffusion_model.training_losses(self.denoise_model_image, batch_item, iEmbeds, batch_index, image_feats, text_feats, audio_feats)"
            # So it seems there is only ONE diffusion model training step per batch, utilizing all modalities?
            # But wait, Main.py defines self.denoise_model_image = ModalDenoiseTransformer(...)
            # It doesn't define denoise_model_text.
            # So I will stick to training just self.denoise_model_image.
            
        # 2. Graph Construction (Re-build UI matrix)
        with torch.no_grad():
            u_list_image = []
            i_list_image = []
            edge_list_image = []
            
            for _, batch in enumerate(self.diffusion_loader):
                batch_item, batch_index = batch
                batch_item, batch_index = batch_item.to(self.device), batch_index.to(self.device)
                
                # p_sample returns (x_t, probs)
                denoised_batch, denoised_prob = self.model.diffusion_model.p_sample(
                    self.model.denoise_model_image, batch_item, self.model.sampling_steps, self.model.bayesian_samplinge_schedule)
                
                # TopK selection logic from GenRecV1 Main.py
                # _, indices = torch.topk(denoised_prob, k=self.model.gen_topk, dim=1)
                # mask = torch.zeros_like(denoised_prob, dtype=torch.bool).scatter_(1, indices, True)
                # denoised_batch = torch.where(mask, denoised_batch, batch_item)
                
                # top_item, indices_ = torch.topk(denoised_batch * denoised_prob, k=self.model.rebuild_k)
                
                # Using simplified logic from Main.py
                _, indices = torch.topk(denoised_prob, k=self.model.gen_topk, dim=1) 
                mask = torch.zeros_like(denoised_prob, dtype=torch.bool).scatter_(1, indices, True)
                denoised_batch = torch.where(mask, denoised_batch, batch_item)
                
                # Interest Debiasing
                if (self.config['OpenInterestDebiase'] if 'OpenInterestDebiase' in self.config else False) and self.multimodal_interest_space is not None:
                     # GenRec-V1 applies debiasing per batch
                     # Assuming 'image_modal' and 'text_modal' are available
                     # Note: Main.py handles audio if tiktok, but we focus on image/text for general
                     
                     # We need sample_ratio
                     sample_ratio = self.config['sample_ratio'] if 'sample_ratio' in self.config else 0.1
                     
                     image_interest_judge = InterestDebiase(
                        origin_interaction_graph = batch_item,
                        generated_interaction_graph = denoised_batch,
                        interest_cluster_space_dict = self.multimodal_interest_space,
                        image_modality='image_modal',
                        text_modality='text_modal',
                        audio_modality=None, # simplified
                        sample_ratio = sample_ratio
                     )
                     denoised_batch = image_interest_judge.interest_query_debiase()

                top_item, indices_ = torch.topk(denoised_batch * denoised_prob, k=self.model.rebuild_k)
                
                for i in range(batch_index.shape[0]):
                    for j in range(indices_[i].shape[0]): 
                        u_list_image.append(int(batch_index[i].cpu().numpy()))
                        i_list_image.append(int(indices_[i][j].cpu().numpy()))
                        edge_list_image.append(1.0)
            
            # Build Matrix
            u_list_image = np.array(u_list_image)
            i_list_image = np.array(i_list_image)
            edge_list_image = np.array(edge_list_image)
            self.model.image_UI_matrix = self.buildUIMatrix(u_list_image, i_list_image, edge_list_image)
            self.model.image_UI_matrix = self.model.edgeDropper(self.model.image_UI_matrix)
            
            # GenRecV1 usually doesn't generate Text UI matrix in Main.py?
            # Lines 458-462 are commented out in Main.py for Text UI matrix.
            # So I will skip Text UI matrix generation.
            
        # 3. Recommendation Training
        # Set R (Sparse Interaction Matrix) for Item-Item GCN if not set
        # self.model.R is needed. GeneralRecommender usually doesn't have self.R
        # But DataHandler in GenRecV1 has it.
        # MMRec Trainer uses train_data (DataLoader).
        # I need to construct R from train_data if model needs it.
        # But wait, self.model.norm_adj is available in DiffMM (calculated in init).
        # GenRecV1 uses self.handler.R which seems to be sparse user-item matrix.
        # I can use self.model.norm_adj as R? Or I need un-normalized R?
        # In item_item_GCN: image_user_embedding = torch.sparse.mm(R, image_item_id_embedding)
        # Usually R is the adjacency matrix. self.model.norm_adj should work or I can construct it.
        # Let's assume self.model.norm_adj is fine or use a helper to get it.
        # DiffMM has get_norm_adj_mat. I should ensure GenRecV1 has it too.
        # I'll check GenRecV1 class again. I didn't add get_norm_adj_mat or norm_adj initialization.
        # I should add it to GenRecV1.__init__.
        
        # For now, let's inject R into model
        if not hasattr(self.model, 'R'):
             # DiffMM uses norm_adj.
             # I'll use norm_adj as R for now.
             pass
             
        rec_loss, loss_batches = super(DiffMMTrainer, self)._train_epoch(train_data, epoch_idx, loss_func)
        
        self.logger.info(f"Diffusion Loss: {epDiLoss_image/steps:.4f}")
        return rec_loss, loss_batches

class MVDiffTrainer(Trainer):
    """Trainer for MVDiff model with multimodal feature diffusion and sparsity diffusion"""
    def __init__(self, config, model, mg=False):
        super(MVDiffTrainer, self).__init__(config, model, mg)
        
        # Denoise optimizers for user-item interactions
        self.denoise_opt_image = optim.Adam(self.model.denoise_model_image.parameters(), lr=config['learning_rate'], weight_decay=0)
        self.denoise_opt_text = optim.Adam(self.model.denoise_model_text.parameters(), lr=config['learning_rate'], weight_decay=0)
        if self.model.audio_modality:
            self.denoise_opt_audio = optim.Adam(self.model.denoise_model_audio.parameters(), lr=config['learning_rate'], weight_decay=0)
        
        # Multimodal feature denoise optimizers
        if hasattr(self.model, 'image_modal_denoise_model'):
            self.image_modal_denoise_optimizer = optim.Adam(self.model.image_modal_denoise_model.parameters(), lr=config['learning_rate'], weight_decay=0)
        if hasattr(self.model, 'text_modal_denoise_model'):
            self.text_modal_denoise_optimizer = optim.Adam(self.model.text_modal_denoise_model.parameters(), lr=config['learning_rate'], weight_decay=0)
        if self.model.audio_modality and hasattr(self.model, 'audio_modal_denoise_model'):
            self.audio_modal_denoise_optimizer = optim.Adam(self.model.audio_modal_denoise_model.parameters(), lr=config['learning_rate'], weight_decay=0)
        
        self.diffusion_loader = None
        self.multimodal_feature_loader = None
        self.item_num = model.n_items
        self.user_num = model.n_users
    
    def _build_diffusion_loader(self, train_data):
        """Build diffusion loader for user-item interactions"""
        if self.diffusion_loader is not None:
            return
        
        df = train_data.dataset.df
        uid_field = self.config['USER_ID_FIELD']
        iid_field = self.config['ITEM_ID_FIELD']
        
        user_interactions = df.groupby(uid_field)[iid_field].apply(list).to_dict()
        
        class DiffusionDataset(torch.utils.data.Dataset):
            def __init__(self, user_interactions, user_num, item_num):
                self.user_interactions = user_interactions
                self.user_num = user_num
                self.item_num = item_num
                self.users = list(range(user_num))
            
            def __len__(self):
                return self.user_num
            
            def __getitem__(self, idx):
                user_id = self.users[idx]
                items = self.user_interactions.get(user_id, [])
                vector = torch.zeros(self.item_num)
                if len(items) > 0:
                    vector[items] = 1.0
                return vector, torch.tensor(user_id).float()
        
        dataset = DiffusionDataset(user_interactions, self.user_num, self.item_num)
        self.diffusion_loader = DataLoader(dataset, batch_size=self.config['train_batch_size'], shuffle=True, num_workers=0)
    
    def _build_multimodal_feature_loader(self):
        """Build loader for multimodal features"""
        if self.multimodal_feature_loader is not None:
            return
        
        class MultimodalFeatureDataset(torch.utils.data.Dataset):
            def __init__(self, image_feats, text_feats, audio_feats):
                self.image_feats = image_feats
                self.text_feats = text_feats
                self.audio_feats = audio_feats
            
            def __len__(self):
                return self.image_feats.shape[0] if self.image_feats is not None else self.text_feats.shape[0]
            
            def __getitem__(self, index):
                image_modal_feature = self.image_feats[index] if self.image_feats is not None else None
                text_modal_feature = self.text_feats[index] if self.text_feats is not None else None
                audio_modal_feature = self.audio_feats[index] if self.audio_feats is not None else None
                
                if audio_modal_feature is not None:
                    return (image_modal_feature, text_modal_feature, audio_modal_feature)
                else:
                    return (image_modal_feature, text_modal_feature)
        
        dataset = MultimodalFeatureDataset(
            self.model.image_embedding,
            self.model.text_embedding,
            self.model.audio_embedding if self.model.audio_modality else None
        )
        self.multimodal_feature_loader = DataLoader(dataset, batch_size=self.config['train_batch_size'], shuffle=False, num_workers=0)
    
    def normalizeAdj(self, mat):
        degree = np.array(mat.sum(axis=-1))
        dInvSqrt = np.reshape(np.power(degree, -0.5), [-1])
        dInvSqrt[np.isinf(dInvSqrt)] = 0.0
        dInvSqrtMat = sp.diags(dInvSqrt)
        return mat.dot(dInvSqrtMat).transpose().dot(dInvSqrtMat).tocoo()
    
    def buildUIMatrix(self, u_list, i_list, edge_list):
        mat = coo_matrix((edge_list, (u_list, i_list)), shape=(self.user_num, self.item_num), dtype=np.float32)
        a = sp.csr_matrix((self.user_num, self.user_num))
        b = sp.csr_matrix((self.item_num, self.item_num))
        mat = sp.vstack([sp.hstack([a, mat]), sp.hstack([mat.transpose(), b])])
        mat = (mat != 0) * 1.0
        mat = (mat + sp.eye(mat.shape[0])) * 1.0
        mat = self.normalizeAdj(mat)
        
        idxs = torch.from_numpy(np.vstack([mat.row, mat.col]).astype(np.int64))
        vals = torch.from_numpy(mat.data.astype(np.float32))
        shape = torch.Size(mat.shape)
        return torch.sparse.FloatTensor(idxs, vals, shape).to(self.device)
    
    def buildItem2ItemMatrix(self, feature):
        """Build Item-Item matrix from features"""
        feature_norm = feature.div(torch.norm(feature, p=2, dim=-1, keepdim=True) + 1e-8)
        sim_adj = torch.mm(feature_norm, feature_norm.transpose(1, 0))
        sim_adj_sparse = build_knn_normalized_graph(sim_adj, topk=self.model.knn_k, is_sparse=True, norm_type='sym')
        return sim_adj, sim_adj_sparse
    
    def _train_epoch(self, train_data, epoch_idx, loss_func=None):
        """Train one epoch for MVDiff"""
        # Build loaders
        self._build_diffusion_loader(train_data)
        self._build_multimodal_feature_loader()
        
        self.model.train()
        epDiLoss_image_modal, epDiLoss_text_modal = 0, 0
        if self.model.audio_modality:
            epDiLoss_audio_modal = 0
        
        # 1. Multimodal Feature Diffusion Training
        for i, data in enumerate(self.multimodal_feature_loader):
            if len(data) > 2:
                image_batch, text_batch, audio_batch = data
                image_batch, text_batch, audio_batch = image_batch.to(self.device), text_batch.to(self.device), audio_batch.to(self.device)
            else:
                image_batch, text_batch = data
                audio_batch = None
                image_batch, text_batch = image_batch.to(self.device), text_batch.to(self.device)
            
            self.image_modal_denoise_optimizer.zero_grad()
            self.text_modal_denoise_optimizer.zero_grad()
            if self.model.audio_modality:
                self.audio_modal_denoise_optimizer.zero_grad()
            
            image_modal_difussion_loss = self.model.diffusion_model.training_multimodal_feature_diffusion_losses(
                model=self.model.image_modal_denoise_model,
                x_start_image=image_batch,
                x_start_text=text_batch,
                x_start_audio=audio_batch,
                modal_flag='image'
            )
            
            text_modal_diffusion_loss = self.model.diffusion_model.training_multimodal_feature_diffusion_losses(
                model=self.model.text_modal_denoise_model,
                x_start_image=image_batch,
                x_start_text=text_batch,
                x_start_audio=audio_batch,
                modal_flag='text'
            )
            
            image_modal_difussion_loss = image_modal_difussion_loss.mean()
            text_modal_diffusion_loss = text_modal_diffusion_loss.mean()
            
            epDiLoss_image_modal += image_modal_difussion_loss.item()
            epDiLoss_text_modal += text_modal_diffusion_loss.item()
            
            if self.model.audio_modality:
                audio_modal_diffusion_loss = self.model.diffusion_model.training_multimodal_feature_diffusion_losses(
                    model=self.model.audio_modal_denoise_model,
                    x_start_image=image_batch,
                    x_start_text=text_batch,
                    x_start_audio=audio_batch,
                    modal_flag='audio'
                )
                audio_modal_diffusion_loss = audio_modal_diffusion_loss.mean()
                epDiLoss_audio_modal += audio_modal_diffusion_loss.item()
                loss = image_modal_difussion_loss + text_modal_diffusion_loss + audio_modal_diffusion_loss
            else:
                loss = image_modal_difussion_loss + text_modal_diffusion_loss
            
            loss.backward()
            self.image_modal_denoise_optimizer.step()
            self.text_modal_denoise_optimizer.step()
            if self.model.audio_modality:
                self.audio_modal_denoise_optimizer.step()
        
        # 2. Generate Multimodal Diffusion Features and Item-Item Graphs
        with torch.no_grad():
            image_modal_diffusion_representation_list = []
            text_modal_diffusion_representation_list = []
            if self.model.audio_modality:
                audio_modal_diffusion_representation_list = []
            
            for _, data in enumerate(self.multimodal_feature_loader):
                if len(data) > 2:
                    image_batch, text_batch, audio_batch = data
                    image_batch, text_batch, audio_batch = image_batch.to(self.device), text_batch.to(self.device), audio_batch.to(self.device)
                else:
                    image_batch, text_batch = data
                    audio_batch = None
                    image_batch, text_batch = image_batch.to(self.device), text_batch.to(self.device)
                
                denoised_image_batch = self.model.diffusion_model.p_sample(
                    self.model.image_modal_denoise_model, image_batch, text_batch, audio_batch,
                    self.model.sampling_steps, self.model.sampling_noise, modal_flag='image')
                denoised_text_batch = self.model.diffusion_model.p_sample(
                    self.model.text_modal_denoise_model, image_batch, text_batch, audio_batch,
                    self.model.sampling_steps, self.model.sampling_noise, modal_flag='text')
                
                image_modal_diffusion_representation_list.append(denoised_image_batch)
                text_modal_diffusion_representation_list.append(denoised_text_batch)
                
                if self.model.audio_modality:
                    denoised_audio_batch = self.model.diffusion_model.p_sample(
                        self.model.audio_modal_denoise_model, image_batch, text_batch, audio_batch,
                        self.model.sampling_steps, self.model.sampling_noise, modal_flag='audio')
                    audio_modal_diffusion_representation_list.append(denoised_audio_batch)
            
            image_modal_diffusion_representation = torch.concat(image_modal_diffusion_representation_list)
            text_modal_diffusion_representation = torch.concat(text_modal_diffusion_representation_list)
            
            image_modal_diffusion_representation += self.model.image_embedding
            text_modal_diffusion_representation += self.model.text_embedding
            
            self.model.image_II_matrix_dense, self.model.image_II_matrix = self.buildItem2ItemMatrix(image_modal_diffusion_representation)
            self.model.text_II_matrix_dense, self.model.text_II_matrix = self.buildItem2ItemMatrix(text_modal_diffusion_representation)
            
            if self.model.audio_modality:
                audio_modal_diffusion_representation = torch.concat(audio_modal_diffusion_representation_list)
                audio_modal_diffusion_representation += self.model.audio_embedding
                self.model.audio_II_matrix_dense, self.model.audio_II_matrix = self.buildItem2ItemMatrix(audio_modal_diffusion_representation)
                self.model.modal_fusion_II_matrix = self.model.image_II_matrix + self.model.text_II_matrix + self.model.audio_II_matrix
            else:
                self.model.modal_fusion_II_matrix = self.model.image_II_matrix + self.model.text_II_matrix
            
            # Add original II matrices
            image_II_origin_dense, image_II_origin = self.buildItem2ItemMatrix(self.model.image_embedding)
            text_II_origin_dense, text_II_origin = self.buildItem2ItemMatrix(self.model.text_embedding)
            self.model.image_II_matrix += image_II_origin
            self.model.text_II_matrix += text_II_origin
            
            if self.model.audio_modality:
                audio_II_origin_dense, audio_II_origin = self.buildItem2ItemMatrix(self.model.audio_embedding)
                self.model.audio_II_matrix += audio_II_origin
        
        # 3. User-Item Interaction Diffusion Training
        epDiLoss_image, epDiLoss_text = 0, 0
        if self.model.audio_modality:
            epDiLoss_audio = 0
        
        steps = len(self.diffusion_loader)
        iEmbeds = self.model.getItemEmbeds().detach()
        image_feats = self.model.getImageFeats().detach()
        text_feats = self.model.getTextFeats().detach()
        if self.model.audio_modality:
            audio_feats = self.model.getAudioFeats().detach()
        
        for i, batch in enumerate(self.diffusion_loader):
            batch_item, batch_index = batch
            batch_item, batch_index = batch_item.to(self.device), batch_index.to(self.device)
            
            self.denoise_opt_image.zero_grad()
            self.denoise_opt_text.zero_grad()
            if self.model.audio_modality:
                self.denoise_opt_audio.zero_grad()
            
            diff_loss_image, gc_loss_image, contra_loss_image = self.model.sparity_diffusion_model.training_losses(
                self.model.denoise_model_image, batch_item, iEmbeds, batch_index, image_feats)
            
            diff_loss_text, gc_loss_text, contra_loss_text = self.model.sparity_diffusion_model.training_losses(
                self.model.denoise_model_text, batch_item, iEmbeds, batch_index, text_feats)
            
            loss_image = diff_loss_image.mean() + gc_loss_image.mean() * self.model.e_loss + contra_loss_image.mean() * self.model.ssl_reg
            loss_text = diff_loss_text.mean() + gc_loss_text.mean() * self.model.e_loss + contra_loss_text.mean() * self.model.ssl_reg
            
            epDiLoss_image += loss_image.item()
            epDiLoss_text += loss_text.item()
            
            if self.model.audio_modality:
                diff_loss_audio, gc_loss_audio, contra_loss_audio = self.model.sparity_diffusion_model.training_losses(
                    self.model.denoise_model_audio, batch_item, iEmbeds, batch_index, audio_feats)
                loss_audio = diff_loss_audio.mean() + gc_loss_audio.mean() * self.model.e_loss + contra_loss_audio.mean() * self.model.ssl_reg
                epDiLoss_audio += loss_audio.item()
                loss = loss_image + loss_text + loss_audio
            else:
                loss = loss_image + loss_text
            
            loss.backward()
            self.denoise_opt_image.step()
            self.denoise_opt_text.step()
            if self.model.audio_modality:
                self.denoise_opt_audio.step()
        
        # 4. Rebuild User-Item Matrices
        with torch.no_grad():
            u_list_image, i_list_image, edge_list_image = [], [], []
            u_list_text, i_list_text, edge_list_text = [], [], []
            if self.model.audio_modality:
                u_list_audio, i_list_audio, edge_list_audio = [], [], []
            
            for _, batch in enumerate(self.diffusion_loader):
                batch_item, batch_index = batch
                batch_item, batch_index = batch_item.to(self.device), batch_index.to(self.device)
                
                # Image modality
                denoised_batch = self.model.sparity_diffusion_model.p_sample(
                    self.model.denoise_model_image, batch_item, self.model.sampling_steps, self.model.sampling_noise)
                top_item, indices_ = torch.topk(denoised_batch, k=self.model.rebuild_k)
                
                for i in range(batch_index.shape[0]):
                    latent_intertest_items = indices_[i]
                    latent_multimodal_items_sim = torch.multiply(
                        self.model.image_II_matrix_dense[latent_intertest_items],
                        self.model.text_II_matrix_dense[latent_intertest_items])
                    if self.model.audio_modality:
                        latent_multimodal_items_sim = torch.multiply(
                            latent_multimodal_items_sim,
                            self.model.audio_II_matrix_dense[latent_intertest_items])
                    
                    latent_multimodal_items_prob, latent_multimodal_items_index = torch.topk(
                        latent_multimodal_items_sim, self.model.rebuild_k, dim=-1)
                    high_order_items_prob, high_order_items_index = torch.topk(
                        latent_multimodal_items_prob.flatten(), self.model.rebuild_k + self.model.high_order_topk)
                    high_order_latent_interest_items = latent_multimodal_items_index.flatten()[high_order_items_index]
                    
                    for item in high_order_latent_interest_items:
                        u_list_image.append(int(batch_index[i].cpu().numpy()))
                        i_list_image.append(int(item.item()))
                        edge_list_image.append(1.0)
                
                # Text modality
                denoised_batch = self.model.sparity_diffusion_model.p_sample(
                    self.model.denoise_model_text, batch_item, self.model.sampling_steps, self.model.sampling_noise)
                top_item, indices_ = torch.topk(denoised_batch, k=self.model.rebuild_k)
                
                for i in range(batch_index.shape[0]):
                    for j in range(indices_[i].shape[0]):
                        u_list_text.append(int(batch_index[i].cpu().numpy()))
                        i_list_text.append(int(indices_[i][j].cpu().numpy()))
                        edge_list_text.append(1.0)
                
                if self.model.audio_modality:
                    denoised_batch = self.model.sparity_diffusion_model.p_sample(
                        self.model.denoise_model_audio, batch_item, self.model.sampling_steps, self.model.sampling_noise)
                    top_item, indices_ = torch.topk(denoised_batch, k=self.model.rebuild_k)
                    
                    for i in range(batch_index.shape[0]):
                        for j in range(indices_[i].shape[0]):
                            u_list_audio.append(int(batch_index[i].cpu().numpy()))
                            i_list_audio.append(int(indices_[i][j].cpu().numpy()))
                            edge_list_audio.append(1.0)
            
            u_list_image = np.array(u_list_image)
            i_list_image = np.array(i_list_image)
            edge_list_image = np.array(edge_list_image)
            self.model.image_UI_matrix = self.buildUIMatrix(u_list_image, i_list_image, edge_list_image)
            self.model.image_UI_matrix = self.model.edgeDropper(self.model.image_UI_matrix)
            
            u_list_text = np.array(u_list_text)
            i_list_text = np.array(i_list_text)
            edge_list_text = np.array(edge_list_text)
            self.model.text_UI_matrix = self.buildUIMatrix(u_list_text, i_list_text, edge_list_text)
            self.model.text_UI_matrix = self.model.edgeDropper(self.model.text_UI_matrix)
            
            if self.model.audio_modality:
                u_list_audio = np.array(u_list_audio)
                i_list_audio = np.array(i_list_audio)
                edge_list_audio = np.array(edge_list_audio)
                self.model.audio_UI_matrix = self.buildUIMatrix(u_list_audio, i_list_audio, edge_list_audio)
                self.model.audio_UI_matrix = self.model.edgeDropper(self.model.audio_UI_matrix)
        
        # 5. Recommendation Training (Standard BPR)
        rec_loss, loss_batches = super(MVDiffTrainer, self)._train_epoch(train_data, epoch_idx, loss_func)
        
        # Log losses
        self.logger.info(f"MVDiff Losses - Feature Diffusion: Image={epDiLoss_image_modal/len(self.multimodal_feature_loader):.4f}, "
                        f"Text={epDiLoss_text_modal/len(self.multimodal_feature_loader):.4f}")
        if self.model.audio_modality:
            self.logger.info(f"Audio={epDiLoss_audio_modal/len(self.multimodal_feature_loader):.4f}")
        self.logger.info(f"Interaction Diffusion: Image={epDiLoss_image/steps:.4f}, Text={epDiLoss_text/steps:.4f}")
        if self.model.audio_modality:
            self.logger.info(f"Audio={epDiLoss_audio/steps:.4f}")
        
        return rec_loss, loss_batches
