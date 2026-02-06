# coding: utf-8
r"""
GenBm3: 将 GenRecv2 的 forward / calculate_loss 可插拔到 BM3，用于实验。
- use_genrecv2_style=False: 保留原始 BM3 逻辑（forward 无参返回 u_g,i_g，calculate_loss 为 BM3 的对比+reg）
- use_genrecv2_style=True: GenRecv2 风格（BPR + RF + gen_cl_loss + ps_loss，多模态图 + condition）
"""

import os
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F

from models.bm3 import BM3
from models.rf_modules import RFEmbeddingGenerator, CausalDenoiser
from models.mm_graph_builder import (
    build_image_text_knn_adjs,
    get_ii_adj_intersection_attention,
    get_adj_mat_with_ii,
    sparse_mx_to_torch_sparse_tensor,
)


def _info_nce(view1, view2, temperature, chunk_size=4096):
    """分块 InfoNCE，避免显存溢出。view1/view2: [N, D]。"""
    view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
    n_samples = view1.size(0)
    if n_samples <= chunk_size:
        pos_score = (view1 * view2).sum(dim=-1)
        pos_score = torch.exp(pos_score / temperature)
        ttl_score = torch.matmul(view1, view2.transpose(0, 1))
        ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)
        cl_loss = -torch.log(pos_score / ttl_score + 1e-8)
        return torch.mean(cl_loss)
    cl_loss_sum = 0.0
    for i in range(0, n_samples, chunk_size):
        end_i = min(i + chunk_size, n_samples)
        view1_chunk = view1[i:end_i]
        pos_score = (view1_chunk * view2[i:end_i]).sum(dim=-1)
        pos_score = torch.exp(pos_score / temperature)
        ttl_score = torch.zeros(end_i - i, device=view1.device)
        for j in range(0, n_samples, chunk_size):
            end_j = min(j + chunk_size, n_samples)
            sim_chunk = torch.matmul(view1_chunk, view2[j:end_j].transpose(0, 1))
            ttl_score += torch.exp(sim_chunk / temperature).sum(dim=1)
        cl_loss_sum += (-torch.log(pos_score / ttl_score + 1e-8)).sum()
    return cl_loss_sum / n_samples


class GenBm3(BM3):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        # 可插拔开关：False=纯 BM3，True=GenRecv2 风格（BPR + RF + gen_cl_loss + ps_loss）
        self.use_genrecv2_style = config["use_genrecv2_style"] if "use_genrecv2_style" in config else True

        if not self.use_genrecv2_style:
            # 保留原始 BM3：不覆盖 norm_adj，不建 R/RF/denoise
            return

        # ---------- GenRecv2 插拔：多模态图 + R ----------
        self.dataset_path = os.path.abspath(config["data_path"] + config["dataset"])
        self.knn_k = config["knn_k"] if "knn_k" in config else 10
        self.sparse = True
        self.mm_image_weight = config["mm_image_weight"] if "mm_image_weight" in config else 0.1
        self.interaction_matrix = dataset.inter_matrix(form="coo").astype(np.float32)

        image_adj, text_adj = None, None
        if self.v_feat is not None or self.t_feat is not None:
            image_adj, text_adj = build_image_text_knn_adjs(
                self.dataset_path,
                self.knn_k,
                self.sparse,
                self.device,
                self.image_embedding.weight if self.v_feat is not None else None,
                self.text_embedding.weight if self.t_feat is not None else None,
            )
            if image_adj is not None:
                self.image_original_adj = image_adj
            if text_adj is not None:
                self.text_original_adj = text_adj

        if image_adj is not None and text_adj is not None:
            ii_adj = get_ii_adj_intersection_attention(
                image_adj, text_adj, self.mm_image_weight, self.n_items
            )
        else:
            ii_adj = sp.coo_matrix((self.n_items, self.n_items), dtype=np.float32)

        norm_adj_scipy, R_scipy = get_adj_mat_with_ii(
            self.interaction_matrix, ii_adj, self.n_users, self.n_items
        )
        self.norm_adj = sparse_mx_to_torch_sparse_tensor(norm_adj_scipy).float().to(self.device)
        self.R = sparse_mx_to_torch_sparse_tensor(R_scipy).float().to(self.device)

        # ---------- GenRecv2 插拔：RF + 协同信号增强 ----------
        self.gen_cl_loss = config["gen_cl_loss"] if "gen_cl_loss" in config else 0.01
        self.bm_temp = config["bm_temp"] if "bm_temp" in config else 0.2

        self.use_rf = config["use_rf"] if "use_rf" in config else True
        if self.use_rf:
            self.rf_generator = RFEmbeddingGenerator(
                embedding_dim=self.embedding_dim,
                hidden_dim=config["rf_hidden_dim"] if "rf_hidden_dim" in config else 128,
                n_layers=config["rf_n_layers"] if "rf_n_layers" in config else 2,
                dropout=config["rf_dropout"] if "rf_dropout" in config else 0.1,
                learning_rate=config["rf_learning_rate"] if "rf_learning_rate" in config else 0.0001,
                sampling_steps=config["rf_sampling_steps"] if "rf_sampling_steps" in config else 10,
                warmup_epochs=config["rf_warmup_epochs"] if "rf_warmup_epochs" in config else 5,
                train_mix_ratio=config["rf_mix_ratio"] if "rf_mix_ratio" in config else 0.1,
                inference_mix_ratio=config["rf_inference_mix_ratio"] if "rf_inference_mix_ratio" in config else 0.2,
                contrast_temp=config["rf_contrast_temp"] if "rf_contrast_temp" in config else 0.2,
                contrast_weight=config["rf_loss_weight"] if "rf_loss_weight" in config else 1.0,
                n_users=self.n_users,
                n_items=self.n_items,
                use_2rf=config["use_2rf"] if "use_2rf" in config else False,
                rf_2rf_transition_epoch=config["rf_2rf_transition_epoch"] if "rf_2rf_transition_epoch" in config else None,
                use_gradient_checkpointing=config["use_gradient_checkpointing"] if "use_gradient_checkpointing" in config else True,
            )
            self._rf_logged_this_epoch = False
            self._current_batch_users = None
            self._current_batch_items = None
            self._training_epoch = -1

        self.use_denoise = config["use_denoise"] if "use_denoise" in config else False
        if self.use_denoise:
            self.ps_loss_weight = config["ps_loss_weight"] if "ps_loss_weight" in config else 0.1
            self.causal_denoiser = CausalDenoiser(
                embedding_dim=self.embedding_dim,
                n_users=self.n_users,
                n_items=self.n_items,
                n_layers=config["denoise_layers"] if "denoise_layers" in config else 2,
                clean_rating_threshold=config["clean_rating_threshold"] if "clean_rating_threshold" in config else 5.0,
                device=self.device,
            )
            self.causal_denoiser.load_treatment_labels(dataset)

    def pre_epoch_processing(self):
        super().pre_epoch_processing()
        if self.use_genrecv2_style and self.use_rf:
            self._training_epoch += 1
            self.rf_generator.set_epoch(self._training_epoch)
            self._rf_logged_this_epoch = False

    def _get_base_embeddings(self):
        """BM3 GCN 输出 (u_g, i_g)，i_g 已含 item residual。"""
        u_g, i_g = super().forward()
        return u_g, i_g

    def _build_conditions_and_prior(self, all_rep):
        """多模态 condition 与 user/item prior（与 RFBM3/GenRecv2 一致）。"""
        t_feat = self.text_trs(self.text_embedding.weight) if self.t_feat is not None else None
        v_feat = self.image_trs(self.image_embedding.weight) if self.v_feat is not None else None
        full_conditions = []
        if v_feat is not None:
            user_v = torch.sparse.mm(self.R, v_feat) if hasattr(self, "R") else torch.zeros(self.n_users, v_feat.shape[1], device=v_feat.device)
            full_conditions.append(torch.cat([user_v, v_feat], dim=0))
        if t_feat is not None:
            user_t = torch.sparse.mm(self.R, t_feat) if hasattr(self, "R") else torch.zeros(self.n_users, t_feat.shape[1], device=t_feat.device)
            full_conditions.append(torch.cat([user_t, t_feat], dim=0))
        if len(full_conditions) == 0:
            return full_conditions, None
        Z_u = torch.zeros(self.n_users, self.embedding_dim, device=all_rep.device)
        if v_feat is not None and hasattr(self, "R"):
            Z_u = Z_u + torch.sparse.mm(self.R, v_feat)
        if t_feat is not None and hasattr(self, "R"):
            Z_u = Z_u + torch.sparse.mm(self.R, t_feat)
        Z_hat_u = Z_u.mean(dim=0, keepdim=True)
        user_prior = Z_u - Z_hat_u
        Z_i = torch.zeros(self.n_items, self.embedding_dim, device=all_rep.device)
        if v_feat is not None:
            Z_i = Z_i + v_feat
        if t_feat is not None:
            Z_i = Z_i + t_feat
        Z_hat_i = Z_i.mean(dim=0, keepdim=True)
        item_prior = Z_i - Z_hat_i
        full_prior = torch.cat([user_prior, item_prior], dim=0)
        return full_conditions, full_prior

    def forward(self, interaction=None, train=False):
        """
        可插拔接口：
        - use_genrecv2_style=False: 与 BM3 一致，忽略 interaction/train，返回 (u_g, i_g)。
        - use_genrecv2_style=True: GenRecv2 风格，需要 interaction，返回 pos_scores, neg_scores [, rf_outputs]（train=True 时含 integration_embeds / extended_id_embeds_aug 用于 gen_cl_loss）。
        """
        if not self.use_genrecv2_style:
            return super().forward()

        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]
        pos_shifted = pos_items + self.n_users
        neg_shifted = neg_items + self.n_users

        u_g, i_g = self._get_base_embeddings()
        all_rep = torch.cat([u_g, i_g], dim=0)
        integration_embeds = all_rep  # 用于 gen_cl_loss 的“主表示”
        user_rep, item_rep = u_g, i_g
        rf_outputs = None
        ps_loss = 0.0

        if self.use_rf and self.training and train:
            if self.use_denoise:
                ego = torch.cat((self.user_embedding.weight, self.item_id_embedding.weight), dim=0)
                denoised_emb, ps_loss = self.causal_denoiser(ego)
                rf_target = denoised_emb.detach() if denoised_emb is not None else all_rep.detach()
            else:
                rf_target = all_rep.detach()

            full_conditions, full_prior = self._build_conditions_and_prior(all_rep)
            if len(full_conditions) > 0:
                self.rf_generator.compute_loss_and_step(
                    target_embeds=rf_target,
                    conditions=[c.detach() for c in full_conditions],
                    user_prior=full_prior.detach() if full_prior is not None else None,
                    epoch=self.rf_generator.current_epoch,
                    batch_users=self._current_batch_users,
                    batch_pos_items=self._current_batch_items,
                )
                if not self._rf_logged_this_epoch:
                    print(
                        f"  [GenBm3 RF] epoch={self.rf_generator.current_epoch}, "
                        f"rf_loss=..., cl_loss=..."
                    )
                    self._rf_logged_this_epoch = True
                rf_embeds = self.rf_generator.generate(full_conditions)
                all_rep = self.rf_generator.mix_embeddings(
                    all_rep, rf_embeds.detach(), training=True, epoch=self.rf_generator.current_epoch
                )
                user_rep, item_rep = torch.split(all_rep, [self.n_users, self.n_items], dim=0)
            # 协同信号增强：aug = dropout(原始 GCN)，用于 gen_cl_loss（与 GenRecv2 一致）
            extended_id_embeds_aug = F.dropout(integration_embeds, p=self.dropout)
            rf_outputs = {
                "ps_loss": ps_loss,
                "integration_embeds": integration_embeds,
                "extended_id_embeds_aug": extended_id_embeds_aug,
            }

        elif self.use_rf and not self.training:
            with torch.no_grad():
                full_conditions, _ = self._build_conditions_and_prior(all_rep)
                if len(full_conditions) > 0:
                    rf_embeds = self.rf_generator.generate(full_conditions)
                    all_rep = self.rf_generator.mix_embeddings(
                        all_rep, rf_embeds, training=False, epoch=self.rf_generator.current_epoch
                    )
                    user_rep, item_rep = torch.split(all_rep, [self.n_users, self.n_items], dim=0)

        # use_rf=False 时也提供 integration/aug，用于 gen_cl_loss
        if self.training and train and rf_outputs is None:
            extended_id_embeds_aug = F.dropout(integration_embeds, p=self.dropout)
            rf_outputs = {
                "ps_loss": 0.0,
                "integration_embeds": integration_embeds,
                "extended_id_embeds_aug": extended_id_embeds_aug,
            }

        self.result_embed = torch.cat([user_rep, item_rep], dim=0)
        u_emb = self.result_embed[users]
        pos_emb = self.result_embed[pos_shifted]
        neg_emb = self.result_embed[neg_shifted]
        pos_scores = torch.sum(u_emb * pos_emb, dim=1)
        neg_scores = torch.sum(u_emb * neg_emb, dim=1)

        if train and rf_outputs is not None:
            return pos_scores, neg_scores, rf_outputs
        return pos_scores, neg_scores

    def calculate_loss(self, interaction):
        """
        可插拔：
        - use_genrecv2_style=False: 原始 BM3 的 calculate_loss(interactions)。
        - use_genrecv2_style=True: BPR + reg + ps_loss + gen_cl_loss（InfoNCE 协同信号增强）。
        """
        if not self.use_genrecv2_style:
            return super().calculate_loss(interaction)

        self._current_batch_users = interaction[0]
        self._current_batch_items = interaction[1]

        out = self.forward(interaction, train=True)
        if isinstance(out, tuple) and len(out) == 3:
            pos_scores, neg_scores, rf_outputs = out
        else:
            pos_scores, neg_scores = out
            rf_outputs = None

        loss_value = -torch.mean(torch.log2(torch.sigmoid(pos_scores - neg_scores)))
        user_rep, item_rep = torch.split(self.result_embed, [self.n_users, self.n_items], dim=0)
        reg_loss = self.reg_weight * self.reg_loss(user_rep, item_rep)
        total_loss = loss_value + reg_loss

        if self.use_denoise and rf_outputs is not None and "ps_loss" in rf_outputs:
            total_loss = total_loss + self.ps_loss_weight * rf_outputs["ps_loss"]

        # 协同信号增强对比学习损失（GenRecv2 风格）
        if rf_outputs is not None and "integration_embeds" in rf_outputs and "extended_id_embeds_aug" in rf_outputs:
            users, pos_items = interaction[0], interaction[1]
            int_user, int_item = torch.split(rf_outputs["integration_embeds"], [self.n_users, self.n_items], dim=0)
            aug_user, aug_item = torch.split(rf_outputs["extended_id_embeds_aug"], [self.n_users, self.n_items], dim=0)
            cl_loss_aug = (
                _info_nce(int_user[users], aug_user[users], self.bm_temp)
                + _info_nce(int_item[pos_items], aug_item[pos_items], self.bm_temp)
            )
            total_loss = total_loss + self.gen_cl_loss * cl_loss_aug

        return total_loss

    def full_sort_predict(self, interaction):
        if not self.use_genrecv2_style:
            return super().full_sort_predict(interaction)
        _ = self.forward(interaction, train=False)
        user = interaction[0]
        user_emb = self.result_embed[: self.n_users]
        item_emb = self.result_embed[self.n_users :]
        return torch.matmul(user_emb[user], item_emb.t())
