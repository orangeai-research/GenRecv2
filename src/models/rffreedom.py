# coding: utf-8
# @email: enoche.chow@gmail.com
r"""
RFFREEDOM: RF-Enhanced FREEDOM
Integrates Rectified Flow module to enhance collaborative filtering embeddings
With optional causal denoising using Inverse Propensity Weighting (IPW)
"""

import torch
import torch.nn.functional as F

from models.freedom import FREEDOM
from models.rf_modules import RFEmbeddingGenerator, CausalDenoiser


class RFFREEDOM(FREEDOM):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)

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
                # 2-RF parameters
                use_2rf=config["use_2rf"] if "use_2rf" in config else True,
                rf_2rf_transition_epoch=config["rf_2rf_transition_epoch"] if "rf_2rf_transition_epoch" in config else None,
                # Memory optimization
                use_gradient_checkpointing=config["use_gradient_checkpointing"] if "use_gradient_checkpointing" in config else True,
            )
            self._rf_logged_this_epoch = False

            # Store batch indices for RF contrastive loss
            self._current_batch_users = None
            self._current_batch_pos_items = None

            # Track training epoch (starts at -1, will be incremented to 0 in first pre_epoch_processing)
            self._training_epoch = -1

        # ===== Denoising Module =====
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
        """Called by trainer at the beginning of each epoch."""
        super().pre_epoch_processing()
        # Increment epoch counter and update RF generator
        if self.use_rf:
            self._training_epoch += 1
            self.rf_generator.set_epoch(self._training_epoch)
            self._rf_logged_this_epoch = False

    def forward(self, adj):
        # 1. Multimodal feature aggregation (through mm_adj)
        h = self.item_id_embedding.weight
        for i in range(self.n_layers):
            h = torch.sparse.mm(self.mm_adj, h)

        # 2. Collaborative graph convolution
        ego_embeddings = torch.cat((self.user_embedding.weight,
                                    self.item_id_embedding.weight), dim=0)
        all_embeddings = [ego_embeddings]
        for i in range(self.n_ui_layers):
            side_embeddings = torch.sparse.mm(adj, ego_embeddings)
            ego_embeddings = side_embeddings
            all_embeddings += [ego_embeddings]

        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
        all_embeddings_ori = all_embeddings.clone()

        u_g_embeddings, i_g_embeddings = torch.split(all_embeddings,
                                                      [self.n_users, self.n_items], dim=0)
        i_g_embeddings_ori = i_g_embeddings.clone()

        # ===== RF Enhancement for both users and items =====
        rf_outputs = None

        if self.use_rf and self.training:
            # Prepare conditions for items
            image_feats = self.image_trs(self.image_embedding.weight) if self.v_feat is not None else None
            text_feats = self.text_trs(self.text_embedding.weight) if self.t_feat is not None else None

            # Extend conditions to user+item space
            full_conditions = []
            if image_feats is not None:
                # Aggregate item visual features to users
                user_image_feats = torch.sparse.mm(self.R, image_feats) if hasattr(self, 'R') else torch.zeros(self.n_users, image_feats.shape[1]).to(image_feats.device)
                full_image_feats = torch.cat([user_image_feats, image_feats], dim=0)
                full_conditions.append(full_image_feats)
            if text_feats is not None:
                # Aggregate item text features to users
                user_text_feats = torch.sparse.mm(self.R, text_feats) if hasattr(self, 'R') else torch.zeros(self.n_users, text_feats.shape[1]).to(text_feats.device)
                full_text_feats = torch.cat([user_text_feats, text_feats], dim=0)
                full_conditions.append(full_text_feats)

            if len(full_conditions) > 0:
                # ===== Denoising: compute denoised embeddings as RF target =====
                ps_loss = 0.0
                if self.use_denoise:
                    ego_emb_for_denoise = torch.cat((self.user_embedding.weight,
                                                     self.item_id_embedding.weight), dim=0)
                    denoised_emb, ps_loss = self.causal_denoiser(ego_emb_for_denoise)
                    if denoised_emb is not None:
                        rf_target = denoised_emb.detach()
                    else:
                        rf_target = all_embeddings_ori.detach()
                else:
                    rf_target = all_embeddings_ori.detach()

                # RF training with denoised embeddings as target
                loss_dict = self.rf_generator.compute_loss_and_step(
                    target_embeds=rf_target,
                    conditions=[c.detach() for c in full_conditions],
                    epoch=self.rf_generator.current_epoch,
                    # Pass batch interaction indices for interaction-based contrastive loss
                    batch_users=self._current_batch_users,
                    batch_pos_items=self._current_batch_pos_items,
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

                # Mix embeddings
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

        elif self.use_rf and not self.training:
            # Inference mode
            with torch.no_grad():
                image_feats = self.image_trs(self.image_embedding.weight) if self.v_feat is not None else None
                text_feats = self.text_trs(self.text_embedding.weight) if self.t_feat is not None else None

                full_conditions = []
                if image_feats is not None:
                    user_image_feats = torch.sparse.mm(self.R, image_feats) if hasattr(self, 'R') else torch.zeros(self.n_users, image_feats.shape[1]).to(image_feats.device)
                    full_image_feats = torch.cat([user_image_feats, image_feats], dim=0)
                    full_conditions.append(full_image_feats)
                if text_feats is not None:
                    user_text_feats = torch.sparse.mm(self.R, text_feats) if hasattr(self, 'R') else torch.zeros(self.n_users, text_feats.shape[1]).to(text_feats.device)
                    full_text_feats = torch.cat([user_text_feats, text_feats], dim=0)
                    full_conditions.append(full_text_feats)

                if len(full_conditions) > 0:
                    rf_embeds = self.rf_generator.generate(full_conditions)
                    all_embeddings_mixed = self.rf_generator.mix_embeddings(
                        all_embeddings_ori, rf_embeds, training=False
                    )
                    u_g_embeddings, i_g_embeddings = torch.split(
                        all_embeddings_mixed, [self.n_users, self.n_items], dim=0
                    )

        # Fuse multimodal features
        return u_g_embeddings, i_g_embeddings + h, rf_outputs

    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]

        # Store batch indices for RF contrastive loss
        self._current_batch_users = users
        self._current_batch_pos_items = pos_items

        ua_embeddings, ia_embeddings, rf_outputs = self.forward(self.masked_adj)
        self.build_item_graph = False

        u_g_embeddings = ua_embeddings[users]
        pos_i_g_embeddings = ia_embeddings[pos_items]
        neg_i_g_embeddings = ia_embeddings[neg_items]

        batch_mf_loss = self.bpr_loss(u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings)
        mf_v_loss, mf_t_loss = 0.0, 0.0
        if self.t_feat is not None:
            text_feats = self.text_trs(self.text_embedding.weight)
            mf_t_loss = self.bpr_loss(ua_embeddings[users], text_feats[pos_items], text_feats[neg_items])
        if self.v_feat is not None:
            image_feats = self.image_trs(self.image_embedding.weight)
            mf_v_loss = self.bpr_loss(ua_embeddings[users], image_feats[pos_items], image_feats[neg_items])

        total_loss = batch_mf_loss + self.reg_weight * (mf_t_loss + mf_v_loss)

        # Add propensity score loss if denoising is enabled
        if self.use_denoise and rf_outputs is not None and "ps_loss" in rf_outputs:
            total_loss = total_loss + self.ps_loss_weight * rf_outputs["ps_loss"]

        # Note: cl_loss is now always computed in rf_modules.py via compute_loss_and_step()

        return total_loss

    def full_sort_predict(self, interaction):
        user = interaction[0]

        restore_user_e, restore_item_e, _ = self.forward(self.norm_adj)
        u_embeddings = restore_user_e[user]

        # dot with all item embedding to accelerate
        scores = torch.matmul(u_embeddings, restore_item_e.transpose(0, 1))
        return scores
