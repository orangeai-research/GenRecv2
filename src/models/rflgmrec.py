# coding: utf-8
# @email: georgeguo.gzq.cn@gmail.com
r"""
RFLGMRec: RF-Enhanced LGMRec
Integrates Rectified Flow module to enhance collaborative graph embeddings
With optional causal denoising using Inverse Propensity Weighting (IPW)
"""

import torch
import torch.nn.functional as F

from models.lgmrec import LGMRec
from models.rf_modules import RFEmbeddingGenerator, CausalDenoiser
from models.module.rectified_flow import RectifiedFlow_Cross_Modality_Alignment

class RFLGMRec(LGMRec):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        # 1. 跨模态对齐模型 (RF Cross-Modality Alignment)
        self.rf_weight = config['rf_weight'] if 'rf_weight' in config else 0.1
        
        # 获取输入维度
        input_dim1 = self.v_feat.shape[1] if self.v_feat is not None else None
        input_dim2 = self.t_feat.shape[1] if self.t_feat is not None else None
        
        # 初始化带投影的 RF 对齐模型
        self.rf_cross_modality_alignment_model = RectifiedFlow_Cross_Modality_Alignment(
            dim=self.embedding_dim,
            input_dim1=input_dim1,
            input_dim2=input_dim2
        ).to(self.device)

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
            print(f"RFLGMRec: Starting epoch {self._training_epoch}")
            self._rf_logged_this_epoch = False

    def forward(self):
        # hyperedge dependencies constructing
        if self.v_feat is not None:
            iv_hyper = torch.mm(self.image_embedding.weight, self.v_hyper)
            uv_hyper = torch.mm(self.adj, iv_hyper)
            iv_hyper = F.gumbel_softmax(iv_hyper, self.tau, dim=1, hard=False)
            uv_hyper = F.gumbel_softmax(uv_hyper, self.tau, dim=1, hard=False)
        if self.t_feat is not None:
            it_hyper = torch.mm(self.text_embedding.weight, self.t_hyper)
            ut_hyper = torch.mm(self.adj, it_hyper)
            it_hyper = F.gumbel_softmax(it_hyper, self.tau, dim=1, hard=False)
            ut_hyper = F.gumbel_softmax(ut_hyper, self.tau, dim=1, hard=False)

        # CGE: collaborative graph embedding
        cge_embs = self.cge()
        cge_embs_ori = cge_embs.clone()

        # ===== RF Enhancement for full user+item CGE =====
        rf_outputs = None

        if self.use_rf:
            # Get modality features (already user+item)
            full_conditions = []
            if self.v_feat is not None:
                v_feats = self.mge('v')  # Already aggregated on user-item graph
                full_conditions.append(v_feats)

            if self.t_feat is not None:
                t_feats = self.mge('t')  # Already aggregated on user-item graph
                full_conditions.append(t_feats)

            if len(full_conditions) > 0 and self.training:
                # ===== Denoising: compute denoised embeddings as RF target =====
                ps_loss = 0.0
                if self.use_denoise:
                    ego_emb_for_denoise = torch.cat((self.user_embedding.weight,
                                                     self.item_id_embedding.weight), dim=0)
                    denoised_emb, ps_loss = self.causal_denoiser(ego_emb_for_denoise)
                    if denoised_emb is not None:
                        rf_target = denoised_emb.detach()
                    else:
                        rf_target = cge_embs_ori.detach()
                else:
                    rf_target = cge_embs_ori.detach()

                # 计算用户先验（用于RF指导）
                Z_u = torch.zeros(self.n_users, self.embedding_dim).to(cge_embs_ori.device)
                if self.v_feat is not None:
                    Z_u = Z_u + v_feats[:self.n_users]
                if self.t_feat is not None:
                    Z_u = Z_u + t_feats[:self.n_users]

                Z_hat_u = Z_u.mean(dim=0, keepdim=True)
                user_prior = Z_u - Z_hat_u

                Z_i = torch.zeros(self.n_items, self.embedding_dim).to(cge_embs_ori.device)
                if self.v_feat is not None:
                    Z_i = Z_i + v_feats[self.n_users:]
                if self.t_feat is not None:
                    Z_i = Z_i + t_feats[self.n_users:]

                Z_hat_i = Z_i.mean(dim=0, keepdim=True)
                item_prior = Z_i - Z_hat_i

                full_prior = torch.cat([user_prior, item_prior], dim=0)

                # RF training with denoised embeddings as target
                loss_dict = self.rf_generator.compute_loss_and_step(
                    target_embeds=rf_target,
                    conditions=[c.detach() for c in full_conditions],
                    user_prior=full_prior.detach(),
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

                rf_embeds = self.rf_generator.generate(full_conditions, inference_only=True)

                cge_embs = self.rf_generator.mix_embeddings(
                    cge_embs_ori, rf_embeds.detach(), training=True
                )

                # Always store rf_outputs (cl_loss_in_main only controls loss computation)
                rf_outputs = {
                    "rf_embeds": rf_embeds,
                    "target_embeds": rf_target,
                    "ps_loss": ps_loss,
                }

            elif len(full_conditions) > 0 and not self.training:
                # Inference mode
                n_steps = 10 if self.rf_generator.is_2rf_active else None

                with torch.no_grad():
                    rf_embeds = self.rf_generator.generate(
                        full_conditions, 
                        n_steps=n_steps,
                        inference_only=True
                    )
                    cge_embs = self.rf_generator.mix_embeddings(
                        cge_embs_ori, rf_embeds, training=False
                    )

        # Continue with original LGMRec logic
        if self.v_feat is not None and self.t_feat is not None:
            # MGE: modal graph embedding
            v_feats = self.mge('v')
            t_feats = self.mge('t')
            # local embeddings = collaborative-related embedding + modality-related embedding
            mge_embs = F.normalize(v_feats) + F.normalize(t_feats)
            lge_embs = cge_embs + mge_embs
            # GHE: global hypergraph embedding
            uv_hyper_embs, iv_hyper_embs = self.hgnnLayer(self.drop(iv_hyper), self.drop(uv_hyper), cge_embs[self.n_users:])
            ut_hyper_embs, it_hyper_embs = self.hgnnLayer(self.drop(it_hyper), self.drop(ut_hyper), cge_embs[self.n_users:])
            av_hyper_embs = torch.concat([uv_hyper_embs, iv_hyper_embs], dim=0)
            at_hyper_embs = torch.concat([ut_hyper_embs, it_hyper_embs], dim=0)
            ghe_embs = av_hyper_embs + at_hyper_embs
            # local embeddings + alpha * global embeddings
            all_embs = lge_embs + self.alpha * F.normalize(ghe_embs)
        else:
            all_embs = cge_embs
            uv_hyper_embs, iv_hyper_embs, ut_hyper_embs, it_hyper_embs = None, None, None, None

        u_embs, i_embs = torch.split(all_embs, [self.n_users, self.n_items], dim=0)

        return u_embs, i_embs, [uv_hyper_embs, iv_hyper_embs, ut_hyper_embs, it_hyper_embs], rf_outputs

    def calculate_loss(self, interaction):
        # Store batch indices for RF contrastive loss
        self._current_batch_users = interaction[0]
        self._current_batch_pos_items = interaction[1]

        ua_embeddings, ia_embeddings, hyper_embeddings, rf_outputs = self.forward()

        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]
        u_g_embeddings = ua_embeddings[users]
        pos_i_g_embeddings = ia_embeddings[pos_items]
        neg_i_g_embeddings = ia_embeddings[neg_items]

        batch_bpr_loss = self.bpr_loss(u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings)

        batch_hcl_loss = 0.0
        if hyper_embeddings[0] is not None:
            [uv_embs, iv_embs, ut_embs, it_embs] = hyper_embeddings
            batch_hcl_loss = self.ssl_triple_loss(uv_embs[users], ut_embs[users], ut_embs) + \
                             self.ssl_triple_loss(iv_embs[pos_items], it_embs[pos_items], it_embs)

        batch_reg_loss = self.reg_loss(u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings)

        loss = batch_bpr_loss + self.cl_weight * batch_hcl_loss + self.reg_weight * batch_reg_loss

        # Add propensity score loss if denoising is enabled
        if self.use_denoise and rf_outputs is not None and "ps_loss" in rf_outputs:
            loss = loss + self.ps_loss_weight * rf_outputs["ps_loss"]

        # ===================================================== 
        #  RF for Cross-Modality Alignment (Task A)
        #  Image -> Text (或双向)
        # ===================================================== 
        unique_items = torch.unique(torch.cat([pos_items, neg_items])) 
        # 获取显式特征 (LGMRec 中的 mge 会返回聚合后的特征，或者直接用 Embedding)
        # 注意：LGMRec 只有 image_embedding 和 text_embedding，没有显式保存 explicit_*_embeds
        # 我们需要直接使用原始 Embedding
        batch_img = self.image_embedding.weight[unique_items]
        batch_txt = self.text_embedding.weight[unique_items]
        
        # 计算 Image -> Text 的流损失 
        # RF 模型内部会自动处理投影降维
        loss_rf_cross_modality_align = self.rf_weight * self.rf_cross_modality_alignment_model.compute_loss(batch_img, batch_txt) 

        # Note: cl_loss is now always computed in rf_modules.py via compute_loss_and_step()

        return loss + loss_rf_cross_modality_align

    def full_sort_predict(self, interaction):
        user = interaction[0]
        user_embs, item_embs, _, _ = self.forward()
        scores = torch.matmul(user_embs[user], item_embs.T)
        return scores