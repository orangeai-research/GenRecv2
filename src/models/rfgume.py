# Rectified Flow for GUME: RF-GUME
# Integrating Rectified Flow into GUME for generating extended_id_embeds

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.gume import GUME
from models.rf_modules import RFEmbeddingGenerator, CausalDenoiser


from models.module.rectified_flow import RectifiedFlow_Cross_Modality_Alignment
from models.module.rectified_flow import RectifiedFlowAligner

class RFGUME(GUME):
    """
    GUME with Rectified Flow

    This is a refactored version that uses the pluggable RF module.
    The RF components are now fully decoupled and can be reused in other models.
    """

    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        # 1. 跨模态对齐模型 (RF Cross-Modality Alignment)
        self.rf_weight = config['rf_weight'] if 'rf_weight' in config else 0.1
        self.rf_intra_model = RectifiedFlowAligner(self.embedding_dim).to(self.device)
        # 获取输入维度
        # 原始特征维度 (从 embedding layer 获取)
        input_dim1 = self.image_embedding.weight.shape[1]
        input_dim2 = self.text_embedding.weight.shape[1]
        
        # 初始化带投影的 RF 对齐模型
        self.rf_cross_modality_alignment_model = RectifiedFlow_Cross_Modality_Alignment(
            dim=self.embedding_dim,
            input_dim1=input_dim1,
            input_dim2=input_dim2
        ).to(self.device)

        

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
                # User guidance parameters
                user_guidance_scale=config["user_guidance_scale"] if "user_guidance_scale" in config else 0.2,
                guidance_decay_power=config["guidance_decay_power"] if "guidance_decay_power" in config else 2.0,
                cosine_guidance_scale=config["cosine_guidance_scale"] if "cosine_guidance_scale" in config else 0.1,
                cosine_decay_power=config["cosine_decay_power"] if "cosine_decay_power" in config else 2.0,
                # 2-RF parameters
                use_2rf=config["use_2rf"] if "use_2rf" in config else False,
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
                n_layers=config["denoise_layers"] if "denoise_layers" in config else 2,
                clean_rating_threshold=config["clean_rating_threshold"] if "clean_rating_threshold" in config else 5.0,
                device=self.device,
            )
            # Load treatment labels from dataset
            self.causal_denoiser.load_treatment_labels(dataset)

    def pre_epoch_processing(self):
        """Called by trainer at the beginning of each epoch."""
        if self.use_rf:
            self._training_epoch += 1
            self.rf_generator.set_epoch(self._training_epoch)
            self._rf_logged_this_epoch = False

    def forward(self, adj, train=False):
        """
        使用RF生成extended_id_embeds的前向传播

        Args:
            adj: 邻接矩阵
            train: 是否训练模式

        Returns:
            all_embeds: 最终的嵌入
            rf_outputs: RF相关的输出（用于计算损失）
            other_outputs: 其他输出
        """
        # ===== 原始GUME的多模态编码 =====
        image_item_embeds = torch.multiply(
            self.item_id_embedding.weight,
            self.image_space_trans(self.image_embedding.weight),
        )
        text_item_embeds = torch.multiply(
            self.item_id_embedding.weight,
            self.text_space_trans(self.text_embedding.weight),
        )

        item_embeds = self.item_id_embedding.weight
        user_embeds = self.user_embedding.weight

        # 原始的extended_id_embeds（用作RF的目标）
        extended_id_embeds_target = self.conv_ui(adj, user_embeds, item_embeds)

        # ===== 显式的多模态特征（作为RF的条件） =====
        # Image modality
        explicit_image_item = self.conv_ii(self.image_original_adj, image_item_embeds)
        explicit_image_user = torch.sparse.mm(self.R, explicit_image_item)
        explicit_image_embeds = torch.cat(
            [explicit_image_user, explicit_image_item], dim=0
        )

        # Text modality
        explicit_text_item = self.conv_ii(self.text_original_adj, text_item_embeds)
        explicit_text_user = torch.sparse.mm(self.R, explicit_text_item)
        explicit_text_embeds = torch.cat(
            [explicit_text_user, explicit_text_item], dim=0
        )

        # ===== 使用RF生成extended_id_embeds =====
        rf_outputs = None
        
        if self.use_rf and train:
            # ===== 训练模式：RF独立训练 =====

            # ===== Denoising: compute denoised embeddings as RF target =====
            ps_loss = 0.0
            if self.use_denoise:
                # Get initial ego embeddings for denoising
                ego_emb_for_denoise = torch.cat((user_embeds, item_embeds), dim=0)
                denoised_emb, ps_loss = self.causal_denoiser(ego_emb_for_denoise)
                if denoised_emb is not None:
                    # Use denoised embeddings as RF generation target
                    rf_target = denoised_emb.detach() # TODO: Update0
                    # rf_target = (denoised_emb.detach() + extended_id_embeds_target.detach()) / 2 # TODO: Update1
                else:
                    rf_target = extended_id_embeds_target.detach()
            else:
                rf_target = extended_id_embeds_target.detach()

            # 计算用户先验（用于RF指导）
            # Z_u: 用户特定的多模态兴趣表示
            Z_u = explicit_image_embeds[:self.n_users] + explicit_text_embeds[:self.n_users] # TODO: 模态的偏好可能并行比较好
            # Z_hat: 通用兴趣表示（所有用户的平均值）
            Z_hat = Z_u.mean(dim=0, keepdim=True)
            # 用户先验: 独特的用户兴趣
            user_prior = Z_u - Z_hat  # shape: (n_users, embedding_dim)
            # 对于物品，不使用个性化指导（零指导）
            item_prior = torch.zeros(self.n_items, self.embedding_dim).to(Z_u.device) 
            # ###############UPdate 2 Start###############
            # # # 计算物品先验（用于RF指导）
            # # Z_i = explicit_image_embeds[self.n_users:] + explicit_text_embeds[self.n_users:]
            # # Z_i_hat = Z_i.mean(dim=0, keepdim=True)
            # # item_prior = Z_i - Z_i_hat
            # # 合并用户和物品先验
            ###############UPdate 2 END###############
            full_prior = torch.cat([user_prior, item_prior], dim=0)

            # 使用RF生成器计算损失并更新
            loss_dict = self.rf_generator.compute_loss_and_step(
                target_embeds=rf_target,
                conditions=[explicit_image_embeds.detach(), explicit_text_embeds.detach()],
                user_prior=full_prior.detach(),
                epoch=self.rf_generator.current_epoch,
                # Pass batch interaction indices for interaction-based contrastive loss
                batch_users=self._current_batch_users,
                batch_pos_items=self._current_batch_items,
            )

            # 打印RF训练信息（每个epoch只打印一次）
            if not self._rf_logged_this_epoch:
                print(
                    f"  [RF Train] epoch={self.rf_generator.current_epoch}, "
                    f"rf_loss={loss_dict['rf_loss']:.6f}, "
                    f"cl_loss={loss_dict['cl_loss']:.6f}"
                )
                self._rf_logged_this_epoch = True

            # 生成RF embeddings
            rf_embeds = self.rf_generator.generate(
                [explicit_image_embeds, explicit_text_embeds]
            )

            # 混合原始和RF生成的embeddings
            extended_id_embeds = self.rf_generator.mix_embeddings(
                extended_id_embeds_target,
                rf_embeds.detach(),
                training=True,
                epoch=self.rf_generator.current_epoch,
            )
            
            # Store RF outputs for loss computation
            rf_outputs = {"ps_loss": ps_loss}

        elif self.use_rf and not train:
            # ===== 推理模式：使用RF生成并混合 =====
            with torch.no_grad():
                rf_embeds = self.rf_generator.generate(
                    [explicit_image_embeds, explicit_text_embeds]
                )
                extended_id_embeds = self.rf_generator.mix_embeddings(
                    extended_id_embeds_target,
                    rf_embeds,
                    training=False,
                    epoch=self.rf_generator.current_epoch,
                )
        else:
            # 不使用RF，保持原始GUME行为
            extended_id_embeds = extended_id_embeds_target

        # ===== 继续GUME的其余部分 =====
        extended_image_embeds = self.conv_ui(
            adj, self.extended_image_user.weight, explicit_image_item
        )
        extended_text_embeds = self.conv_ui(
            adj, self.extended_text_user.weight, explicit_text_item
        )
        extended_it_embeds = (extended_image_embeds + extended_text_embeds) / 2

        # Attributes Separation for Better Integration
        image_weights, text_weights = torch.split(
            self.softmax(
                torch.cat(
                    [
                        self.separate_coarse(explicit_image_embeds),
                        self.separate_coarse(explicit_text_embeds),
                    ],
                    dim=-1,
                )
            ),
            1,
            dim=-1,
        )
        coarse_grained_embeds = (
            image_weights * explicit_image_embeds + text_weights * explicit_text_embeds
        )

        fine_grained_image = torch.multiply(
            self.image_behavior(extended_id_embeds),
            (explicit_image_embeds - coarse_grained_embeds),
        )
        fine_grained_text = torch.multiply(
            self.text_behavior(extended_id_embeds),
            (explicit_text_embeds - coarse_grained_embeds),
        )
        integration_embeds = (
            fine_grained_image + fine_grained_text + coarse_grained_embeds
        ) / 3

        all_embeds = extended_id_embeds + integration_embeds

        if train and self.use_rf:
            other_outputs = {
                "integration_embeds": integration_embeds,
                "extended_id_embeds": extended_id_embeds,
                "extended_it_embeds": extended_it_embeds,
                "explicit_image_embeds": explicit_image_embeds,
                "explicit_text_embeds": explicit_text_embeds,
            }
            # Merge rf_outputs if available
            if rf_outputs is not None:
                other_outputs.update(rf_outputs)

            return all_embeds, other_outputs
        elif train and not self.use_rf:
            return (
                all_embeds,
                (integration_embeds, extended_id_embeds, extended_it_embeds),
                (explicit_image_embeds, explicit_text_embeds),
            )

        return all_embeds

    def calculate_loss(self, interaction):
        """
        计算总损失（RF损失已在forward中独立计算和反向传播）

        Args:
            interaction: 交互数据

        Returns:
            total_loss: 总损失
        """
        # Store batch indices for RF contrastive loss
        self._current_batch_users = interaction[0]
        self._current_batch_items = interaction[1]
        
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]

        # 前向传播（RF损失已在forward中独立计算和反向传播）
        if self.use_rf:
            embeds_1, other_outputs = self.forward(self.norm_adj, train=True)

            integration_embeds = other_outputs["integration_embeds"]
            extended_id_embeds = other_outputs["extended_id_embeds"]
            extended_it_embeds = other_outputs["extended_it_embeds"]
            explicit_image_embeds = other_outputs["explicit_image_embeds"]
            explicit_text_embeds = other_outputs["explicit_text_embeds"]
        else:
            embeds_1, embeds_2, embeds_3 = self.forward(self.norm_adj, train=True)
            integration_embeds, extended_id_embeds, extended_it_embeds = embeds_2
            explicit_image_embeds, explicit_text_embeds = embeds_3

        users_embeddings, items_embeddings = torch.split(
            embeds_1, [self.n_users, self.n_items], dim=0
        )

        # ===== 原始GUME损失 =====
        u_g_embeddings = users_embeddings[users]
        pos_i_g_embeddings = items_embeddings[pos_items]
        neg_i_g_embeddings = items_embeddings[neg_items]

        vt_loss = self.vt_loss * self.align_vt(
            explicit_image_embeds, explicit_text_embeds
        )

        integration_users, integration_items = torch.split(
            integration_embeds, [self.n_users, self.n_items], dim=0
        )
        extended_id_user, extended_id_items = torch.split(
            extended_id_embeds, [self.n_users, self.n_items], dim=0
        )
        bpr_loss, reg_loss_1 = self.bpr_loss(
            u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings
        )

        bm_loss = self.bm_loss * (
            self.InfoNCE(
                integration_users[users], extended_id_user[users], self.bm_temp
            )
            + self.InfoNCE(
                integration_items[pos_items], extended_id_items[pos_items], self.bm_temp
            )
        )

        al_loss = vt_loss + bm_loss

        extended_it_user, extended_it_items = torch.split(
            extended_it_embeds, [self.n_users, self.n_items], dim=0
        )

        c_loss = self.InfoNCE(
            extended_it_user[users], integration_users[users], self.um_temp
        )
        noise_loss_1 = self.cal_noise_loss(users, integration_users, self.um_temp)
        noise_loss_2 = self.cal_noise_loss(users, extended_it_user, self.um_temp)
        um_loss = self.um_loss * (c_loss + noise_loss_1 + noise_loss_2)

        reg_loss_2 = (
            self.reg_weight_2
            * self.sq_sum(extended_it_items[pos_items])
            / self.batch_size
        )
        reg_loss = reg_loss_1 + reg_loss_2

        total_loss = bpr_loss + al_loss + um_loss + reg_loss
        
        # Add propensity score loss if denoising is enabled
        if self.use_denoise and self.use_rf and "ps_loss" in other_outputs:
            total_loss = total_loss + self.ps_loss_weight * other_outputs["ps_loss"]

        # ===================================================== 
        #  RF for Cross-Modality Alignment (Task A)
        #  Image -> Text (或双向)
        # ===================================================== 
        unique_items = torch.unique(torch.cat([pos_items, neg_items])) 
        # 获取原始特征 (4096/384 维)
        # batch_img = self.image_embedding.weight[unique_items]
        # batch_txt = self.text_embedding.weight[unique_items]

        batch_img = explicit_image_embeds[unique_items] 
        batch_txt = explicit_text_embeds[unique_items]
        
        # 计算 Image -> Text 的流损失 
        # RF 模型内部会自动处理投影降维
        # loss_rf_cross_modality_align = self.rf_weight * self.rf_cross_modality_alignment_model.compute_loss(batch_img, batch_txt) 
        loss_rf_cross_modality_align = self.rf_weight * self.rf_intra_model.compute_loss(batch_img, batch_txt)
        # Note: cl_loss is now always computed in rf_modules.py via compute_loss_and_step()

        return total_loss + loss_rf_cross_modality_align

    def full_sort_predict(self, interaction):
        """
        预测（推理模式）
        """
        user = interaction[0]

        if self.use_rf:
            all_embeds = self.forward(self.norm_adj, train=False)
        else:
            all_embeds = self.forward(self.norm_adj)

        restore_user_e, restore_item_e = torch.split(
            all_embeds, [self.n_users, self.n_items], dim=0
        )
        u_embeddings = restore_user_e[user]

        scores = torch.matmul(u_embeddings, restore_item_e.transpose(0, 1))
        return scores