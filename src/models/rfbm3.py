# coding: utf-8
# @email: enoche.chow@gmail.com
r"""
RFBM3: RF-Enhanced BM3
Integrates Rectified Flow module to enhance collaborative filtering embeddings
With optional causal denoising using Inverse Propensity Weighting (IPW)
"""

import os
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from torch.nn.functional import cosine_similarity

from models.bm3 import BM3
from models.rf_modules import RFEmbeddingGenerator, CausalDenoiser
from models.mm_graph_builder import (
    build_image_text_knn_adjs,
    get_ii_adj_intersection_attention,
    get_adj_mat_with_ii,
    sparse_mx_to_torch_sparse_tensor,
)


class RFBM3(BM3):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        # 模块一: 与 backbone 一致的图像/文本 KNN 图构建 + 边集取交边权 -> 增强 norm_adj 与 R
        self.dataset_path = os.path.abspath(config["data_path"] + config["dataset"])
        self.knn_k = config['knn_k'] if 'knn_k' in config else 10
        self.sparse = True
        self.mm_image_weight = config['mm_image_weight'] if 'mm_image_weight' in config else 0.1
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

        self.use_rf = config["use_rf"] if "use_rf" in config else True

        if self.use_rf:
            # Initialize RF generator with consistent parameters
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
                # 2-RF parameters
                use_2rf=config["use_2rf"] if "use_2rf" in config else True,
                rf_2rf_transition_epoch=config["rf_2rf_transition_epoch"] if "rf_2rf_transition_epoch" in config else None,
                # Memory optimization
                use_gradient_checkpointing=config["use_gradient_checkpointing"] if "use_gradient_checkpointing" in config else True,
            )
            self._rf_logged_this_epoch = False

            # Store batch indices for RF contrastive loss
            self._current_batch_users = None
            self._current_batch_items = None

            # Track training epoch (starts at -1, will be incremented to 0 in first pre_epoch_processing)
            self._training_epoch = -1

        # ===== Denoising Module =====
        self.use_denoise = config["use_denoise"] if "use_denoise" in config else False

        if self.use_denoise:
            self.ps_loss_weight = config["ps_loss_weight"] if "ps_loss_weight" in config else 0.1

            # Initialize CausalDenoiser
            self.causal_denoiser = CausalDenoiser(
                embedding_dim=self.embedding_dim,
                n_users=self.n_users,
                n_items=self.n_items,
                n_layers=config["denoise_layers"] if "denoise_layers" in config else self.n_layers,
                clean_rating_threshold=config["clean_rating_threshold"] if "clean_rating_threshold" in config else 5.0,
                device=self.device,
            )
            # Load treatment labels from dataset
            self.causal_denoiser.load_treatment_labels(dataset)


    def pre_epoch_processing(self):
        """Called by trainer at the beginning of each epoch."""
        super().pre_epoch_processing()
        # Increment epoch counter and update RF generator
        if self.use_rf:
            self._training_epoch += 1
            self.rf_generator.set_epoch(self._training_epoch)
            self._rf_logged_this_epoch = False

    def forward(self):
        h = self.item_id_embedding.weight

        # Original BM3 GCN
        ego_embeddings = torch.cat((self.user_embedding.weight,
                                    self.item_id_embedding.weight), dim=0)
        all_embeddings = [ego_embeddings]

        for i in range(self.n_layers):
            ego_embeddings = torch.sparse.mm(self.norm_adj, ego_embeddings)
            all_embeddings += [ego_embeddings]

        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = all_embeddings.mean(dim=1, keepdim=False)

        # all_embeddings: (n_users + n_items, embedding_dim)
        all_embeddings_ori = all_embeddings.clone()

        u_g_embeddings, i_g_embeddings_ori = torch.split(
            all_embeddings, [self.n_users, self.n_items], dim=0
        )

        # ===== RF Enhancement for both users and items =====
        rf_outputs = None

        if self.use_rf:
            # Prepare multimodal conditions for items
            t_feat_online = self.text_trs(self.text_embedding.weight) if self.t_feat is not None else None
            v_feat_online = self.image_trs(self.image_embedding.weight) if self.v_feat is not None else None

            # Extend conditions to user+item space
            # For users, we use the item features aggregated via interaction matrix
            item_conditions = []
            full_conditions = []
            if v_feat_online is not None:
                item_conditions.append(v_feat_online)
                # Aggregate item visual features to users via R matrix
                user_v_feat = torch.sparse.mm(self.R, v_feat_online) if hasattr(self, 'R') else torch.zeros(self.n_users, v_feat_online.shape[1]).to(v_feat_online.device)
                full_v_feat = torch.cat([user_v_feat, v_feat_online], dim=0)
                full_conditions.append(full_v_feat)
            if t_feat_online is not None:
                item_conditions.append(t_feat_online)
                # Aggregate item text features to users via R matrix
                user_t_feat = torch.sparse.mm(self.R, t_feat_online) if hasattr(self, 'R') else torch.zeros(self.n_users, t_feat_online.shape[1]).to(t_feat_online.device)
                full_t_feat = torch.cat([user_t_feat, t_feat_online], dim=0)
                full_conditions.append(full_t_feat)

            if len(full_conditions) > 0 and self.training:
                # print("[RFBM3] Forward in TRAINING mode")
                # ===== Denoising: compute denoised embeddings as RF target =====
                ps_loss = 0.0
                if self.use_denoise:
                    # Get initial ego embeddings for denoising
                    ego_emb_for_denoise = torch.cat((self.user_embedding.weight,
                                                     self.item_id_embedding.weight), dim=0)
                    denoised_emb, ps_loss = self.causal_denoiser(ego_emb_for_denoise)
                    if denoised_emb is not None:
                        # Use denoised embeddings as RF generation target
                        rf_target = denoised_emb.detach()
                    else:
                        rf_target = all_embeddings_ori.detach()
                else:
                    rf_target = all_embeddings_ori.detach()

                # 计算用户先验（用于RF指导）
                # Z_u: 用户特定的多模态兴趣表示
                Z_u = torch.zeros(self.n_users, self.embedding_dim).to(all_embeddings_ori.device)
                if v_feat_online is not None and hasattr(self, 'R'):
                    Z_u = Z_u + user_v_feat
                if t_feat_online is not None and hasattr(self, 'R'):
                    Z_u = Z_u + user_t_feat

                # Z_hat_u: 通用用户兴趣表示（所有用户的平均值）
                Z_hat_u = Z_u.mean(dim=0, keepdim=True)

                # 用户先验: 独特的用户兴趣
                user_prior = Z_u - Z_hat_u  # shape: (n_users, embedding_dim)

                # 计算物品先验（用于RF指导）
                # Z_i: 物品特定的多模态特征表示
                Z_i = torch.zeros(self.n_items, self.embedding_dim).to(all_embeddings_ori.device)
                if v_feat_online is not None:
                    Z_i = Z_i + v_feat_online
                if t_feat_online is not None:
                    Z_i = Z_i + t_feat_online

                # Z_hat_i: 通用物品特征表示（所有物品的平均值）
                Z_hat_i = Z_i.mean(dim=0, keepdim=True)

                # 物品先验: 独特的物品特征
                item_prior = Z_i - Z_hat_i  # shape: (n_items, embedding_dim)

                # 合并用户和物品先验
                full_prior = torch.cat([user_prior, item_prior], dim=0)

                # RF training with denoised embeddings as target
                loss_dict = self.rf_generator.compute_loss_and_step(
                    target_embeds=rf_target,
                    conditions=[c.detach() for c in full_conditions],
                    user_prior=full_prior.detach(),
                    epoch=self.rf_generator.current_epoch,
                    # Pass batch interaction indices for interaction-based contrastive loss
                    batch_users=self._current_batch_users,
                    batch_pos_items=self._current_batch_items,
                )

                if not self._rf_logged_this_epoch:
                    mode = "2-RF" if loss_dict.get("is_2rf", False) else "1-RF"
                    log_msg = f"  [{mode} Train] epoch={self.rf_generator.current_epoch}, "
                    log_msg += f"rf_loss={loss_dict['rf_loss']:.6f}, cl_loss={loss_dict['cl_loss']:.6f}"
                    if self.use_denoise:
                        log_msg += f", ps_loss={ps_loss:.6f}"
                    print(log_msg)
                    self._rf_logged_this_epoch = True

                # Generate RF embeddings for full user+item space
                rf_embeds = self.rf_generator.generate(full_conditions)

                # Mix embeddings (using original embeddings, not denoised)
                all_embeddings_mixed = self.rf_generator.mix_embeddings(
                    all_embeddings_ori, rf_embeds.detach(), training=True
                )

                u_g_embeddings, i_g_embeddings = torch.split(
                    all_embeddings_mixed, [self.n_users, self.n_items], dim=0
                )

                # Store rf_outputs for cl_loss in calculate_loss
                rf_outputs = {
                    "rf_embeds": rf_embeds,
                    "target_embeds": rf_target,
                    "ps_loss": ps_loss,
                }

                # Residual connection for items only
                i_g_embeddings = i_g_embeddings + h
                return u_g_embeddings, i_g_embeddings, rf_outputs

            elif len(full_conditions) > 0 and not self.training:
                # Inference mode
                # print("[RFBM3] Forward in INFERENCE mode")
                with torch.no_grad():
                    rf_embeds = self.rf_generator.generate(full_conditions)
                    all_embeddings_mixed = self.rf_generator.mix_embeddings(
                        all_embeddings_ori, rf_embeds, training=False
                    )
                    u_g_embeddings, i_g_embeddings = torch.split(
                        all_embeddings_mixed, [self.n_users, self.n_items], dim=0
                    )

        # Residual connection for items
        i_g_embeddings = i_g_embeddings_ori + h

        return u_g_embeddings, i_g_embeddings, None

    def calculate_loss(self, interactions):
        # Store batch indices for RF contrastive loss
        self._current_batch_users = interactions[0]
        self._current_batch_items = interactions[1]

        # online network (RFBM3 forward returns 3 values)
        u_online_ori, i_online_ori, rf_outputs = self.forward()
        t_feat_online, v_feat_online = None, None
        if self.t_feat is not None:
            t_feat_online = self.text_trs(self.text_embedding.weight)
        if self.v_feat is not None:
            v_feat_online = self.image_trs(self.image_embedding.weight)

        with torch.no_grad():
            u_target, i_target = u_online_ori.clone(), i_online_ori.clone()
            u_target.detach()
            i_target.detach()
            u_target = F.dropout(u_target, self.dropout)
            i_target = F.dropout(i_target, self.dropout)

            if self.t_feat is not None:
                t_feat_target = t_feat_online.clone()
                t_feat_target = F.dropout(t_feat_target, self.dropout)

            if self.v_feat is not None:
                v_feat_target = v_feat_online.clone()
                v_feat_target = F.dropout(v_feat_target, self.dropout)

        u_online, i_online = self.predictor(u_online_ori), self.predictor(i_online_ori)

        users, items = interactions[0], interactions[1]
        u_online = u_online[users, :]
        i_online = i_online[items, :]
        u_target = u_target[users, :]
        i_target = i_target[items, :]

        loss_t, loss_v, loss_tv, loss_vt = 0.0, 0.0, 0.0, 0.0
        if self.t_feat is not None:
            t_feat_online = self.predictor(t_feat_online)
            t_feat_online = t_feat_online[items, :]
            t_feat_target = t_feat_target[items, :]
            loss_t = 1 - cosine_similarity(t_feat_online, i_target.detach(), dim=-1).mean()
            loss_tv = 1 - cosine_similarity(t_feat_online, t_feat_target.detach(), dim=-1).mean()
        if self.v_feat is not None:
            v_feat_online = self.predictor(v_feat_online)
            v_feat_online = v_feat_online[items, :]
            v_feat_target = v_feat_target[items, :]
            loss_v = 1 - cosine_similarity(v_feat_online, i_target.detach(), dim=-1).mean()
            loss_vt = 1 - cosine_similarity(v_feat_online, v_feat_target.detach(), dim=-1).mean()

        loss_ui = 1 - cosine_similarity(u_online, i_target.detach(), dim=-1).mean()
        loss_iu = 1 - cosine_similarity(i_online, u_target.detach(), dim=-1).mean()

        total_loss = (loss_ui + loss_iu).mean() + self.reg_weight * self.reg_loss(u_online_ori, i_online_ori) + \
                     self.cl_weight * (loss_t + loss_v + loss_tv + loss_vt).mean()

        # Add propensity score loss if denoising is enabled
        if self.use_denoise and rf_outputs is not None and "ps_loss" in rf_outputs:
            total_loss = total_loss + self.ps_loss_weight * rf_outputs["ps_loss"]

        # Note: cl_loss is now always computed in rf_modules.py via compute_loss_and_step()

        return total_loss

    def full_sort_predict(self, interaction):
        user = interaction[0]
        u_online, i_online, _ = self.forward()
        u_online, i_online = self.predictor(u_online), self.predictor(i_online)
        score_mat_ui = torch.matmul(u_online[user], i_online.transpose(0, 1))
        return score_mat_ui