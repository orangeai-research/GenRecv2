
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbone import BACKBONE
from models.rf_modules import RFEmbeddingGenerator, CausalDenoiser
from models.module.rectified_flow import RectifiedFlowAligner


def sliced_wasserstein_distance(X, Y, n_proj=32, chunk_size=4096):
    """
    Sliced Wasserstein-2 distance between two batches of embeddings (可微).
    X, Y: (N, D), 同一 N。对 n_proj 个随机方向做 1D 投影，排序后算 L2，再平均。
    chunk_size: 当 N 很大时对方向分块以省显存。
    """
    N, D = X.shape
    if N != Y.shape[0] or N == 0:
        return torch.tensor(0.0, device=X.device, dtype=X.dtype)
    device, dtype = X.device, X.dtype
    # 随机单位方向 (n_proj, D)
    theta = torch.randn(n_proj, D, device=device, dtype=dtype)
    theta = F.normalize(theta, dim=1)
    # 分块投影以省显存
    if N <= chunk_size:
        X_proj = X @ theta.t()   # (N, n_proj)
        Y_proj = Y @ theta.t()
        X_proj_sorted, _ = torch.sort(X_proj, dim=0)
        Y_proj_sorted, _ = torch.sort(Y_proj, dim=0)
        w2_per_slice = ((X_proj_sorted - Y_proj_sorted) ** 2).mean(dim=0)
        return w2_per_slice.mean()
    w2_sum = 0.0
    for i in range(0, n_proj, chunk_size):
        end_i = min(i + chunk_size, n_proj)
        th = theta[i:end_i]  # (chunk, D)
        xp = X @ th.t()   # (N, chunk)
        yp = Y @ th.t()
        xp, _ = torch.sort(xp, dim=0)
        yp, _ = torch.sort(yp, dim=0)
        w2_sum += ((xp - yp) ** 2).mean(dim=0).sum()
    return w2_sum / n_proj


def fid_embedding(X, Y, eps=1e-6):
    """
    Fréchet distance between two (N, D) embedding distributions (embedding-space FID).
    FID = ||mu_x - mu_y||^2 + Tr(Sigma_x) + Tr(Sigma_y) - 2*Tr(sqrt(Sigma_x @ Sigma_y))
    X, Y: (N, D), same N. Returns a scalar (lower is better).
    """
    N, D = X.shape
    if N != Y.shape[0] or N < 2:
        return torch.tensor(0.0, device=X.device, dtype=X.dtype)
    mu_x = X.mean(dim=0)
    mu_y = Y.mean(dim=0)
    X_centered = X - mu_x
    Y_centered = Y - mu_y
    # (D, D) cov, use 1/(N-1) for unbiased
    Sigma_x = (X_centered.t() @ X_centered) / (N - 1)
    Sigma_y = (Y_centered.t() @ Y_centered) / (N - 1)
    Sigma_x = Sigma_x + eps * torch.eye(D, device=X.device, dtype=X.dtype)
    Sigma_y = Sigma_y + eps * torch.eye(D, device=Y.device, dtype=Y.dtype)
    diff = mu_x - mu_y
    tr_sx = Sigma_x.diagonal().sum()
    tr_sy = Sigma_y.diagonal().sum()
    # sqrt(Sigma_x @ Sigma_y); matrix_sqrt requires PSD
    prod = Sigma_x @ Sigma_y
    # sqrt of product via eigendecomposition for numerical stability
    try:
        eigvals = torch.linalg.eigvalsh(prod)
        eigvals = torch.clamp(eigvals, min=0.0)
        tr_sqrt = (eigvals ** 0.5).sum()
    except Exception:
        tr_sqrt = torch.tensor(0.0, device=X.device, dtype=X.dtype)
    fid = (diff ** 2).sum() + tr_sx + tr_sy - 2.0 * tr_sqrt
    return fid.clamp(min=0.0)


def trajectory_linearity(trajectory, eps=1e-8):
    """
    轨迹线性度：相邻步方向与整体方向 (x_T - x_0) 的余弦相似度平均。
    trajectory: (T+1, N, D)，T 为步数。
    公式: (1/T) * sum_{t=0}^{T-1} mean_over_N( CosSim(x_{t+1}-x_t, x_T-x_0) )。
    直线轨迹接近 1.0，弯曲轨迹更低。
    """
    T1, N, D = trajectory.shape
    if T1 < 2:
        return torch.tensor(1.0, device=trajectory.device, dtype=trajectory.dtype)
    T = T1 - 1
    traj = trajectory.float()
    global_dir = traj[-1] - traj[0]  # (N, D)
    norm_global = global_dir.norm(dim=1, keepdim=True).clamp(min=eps)
    global_dir = global_dir / norm_global
    cos_sum = 0.0
    for t in range(T):
        delta = traj[t + 1] - traj[t]  # (N, D)
        norm_delta = delta.norm(dim=1, keepdim=True).clamp(min=eps)
        delta = delta / norm_delta
        cos_t = (delta * global_dir).sum(dim=1).mean()  # scalar
        cos_sum = cos_sum + cos_t
    return cos_sum / T



class GenRecv2(BACKBONE):
    """
    GUME with Rectified Flow

    This is a refactored version that uses the pluggable RF module.
    The RF components are now fully decoupled and can be reused in other models.
    """

    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        self.use_rf = config["use_rf"] if "use_rf" in config else True
        self.rf_weight = config["rf_weight"] if "rf_weight" in config else 0.3
        self.gen_cl_loss = config["gen_cl_loss"] if "gen_cl_loss" in config else 0.01
        #self.rf_cross_modality_alignment_model = RectifiedFlowAligner(self.embedding_dim).to(self.device)
        # Wasserstein 模态对齐（可选）- 权重不宜过大，建议配合 warmup 使用
        self.use_wasserstein_vt = config["use_wasserstein_vt"] if "use_wasserstein_vt" in config else False
        self.wasserstein_vt_weight = config["wasserstein_vt_weight"] if "wasserstein_vt_weight" in config else 0.01
        self.wasserstein_n_slices = config["wasserstein_n_slices"] if "wasserstein_n_slices" in config else 32
        self.wasserstein_warmup_epochs = config["wasserstein_warmup_epochs"] if "wasserstein_warmup_epochs" in config else 40
        # Wasserstein 生成-原始对齐（可选）：增强表征 与 原始兴趣表征 的分布对齐；建议小权重 + detach 原始
        self.use_wasserstein_gen_align = config["use_wasserstein_gen_align"] if "use_wasserstein_gen_align" in config else False
        self.wasserstein_gen_weight = config["wasserstein_gen_weight"] if "wasserstein_gen_weight" in config else 0.01
        self.wasserstein_gen_detach_origin = config["wasserstein_gen_detach_origin"] if "wasserstein_gen_detach_origin" in config else True

        if self.use_rf:
            # Initialize RF generator with consistent parameters
            self.rf_generator = RFEmbeddingGenerator(
                embedding_dim=self.embedding_dim,
                hidden_dim=config["rf_hidden_dim"] if "rf_hidden_dim" in config else 128,
                n_layers=config["rf_n_layers"] if "rf_n_layers" in config else 2,
                dropout=config["rf_dropout"] if "rf_dropout" in config else 0.1,
                learning_rate=config["rf_learning_rate"] if "rf_learning_rate" in config else 0.0001,
                sampling_steps=config["rf_sampling_steps"] if "rf_sampling_steps" in config else 10,
                warmup_epochs=config["rf_warmup_epochs"] if "rf_warmup_epochs" in config else 20,
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
        self._current_epoch = getattr(self, "_current_epoch", -1) + 1
        if self.use_rf:
            self._training_epoch += 1
            self._current_epoch = self._training_epoch
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
        # ===== 通用的多模态编码 =====
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
        extended_id_embeds = self.conv_ui(adj, user_embeds, item_embeds) # 用户兴趣ID表征向量 H1

        extended_id_embeds_origin = self.conv_ui(self.norm_adj_origin, user_embeds, item_embeds) # 用户兴趣ID表征向量 
        extended_id_embeds_target = extended_id_embeds # 用户兴趣ID表征向量 H1

        extended_id_embeds_aug = extended_id_embeds

        # ===== 显式的多模态特征（作为RF的条件） =====
        # Image modality # explicit_image_embeds 用户对图像模态的兴趣表征
        explicit_image_item = self.conv_ii(self.image_original_adj, image_item_embeds)
        explicit_image_user = torch.sparse.mm(self.R, explicit_image_item)
        explicit_image_embeds = torch.cat(
            [explicit_image_user, explicit_image_item], dim=0
        )

        # Text modality #explicit_text_embeds  用户对文本模态的兴趣表征
        explicit_text_item = self.conv_ii(self.text_original_adj, text_item_embeds)
        explicit_text_user = torch.sparse.mm(self.R, explicit_text_item)
        explicit_text_embeds = torch.cat(
            [explicit_text_user, explicit_text_item], dim=0
        )

        # ===== 使用RF生成extended_id_embeds =====
        rf_start = extended_id_embeds_origin
        rf_outputs = None
        
        if self.use_rf and self.training:
            # ===== 训练模式：RF独立训练 =====
            # print("[RFGUME] Forward in TRAINING mode")
            # ===== Denoising: compute denoised embeddings as RF target =====
            ps_loss = 0.0
            if self.use_denoise:
                # Get initial ego embeddings for denoising
                ego_emb_for_denoise = torch.cat((user_embeds, item_embeds), dim=0)
                denoised_emb, ps_loss = self.causal_denoiser(ego_emb_for_denoise)
                if denoised_emb is not None:
                    # Use denoised embeddings as RF generation target
                    rf_target = denoised_emb.detach()
                else:
                    rf_target = extended_id_embeds_target.detach()
            else:
                rf_target = extended_id_embeds_target.detach()

            # 计算用户先验（用于RF指导）
            # Z_u: 用户特定的多模态兴趣表示
            Z_u = explicit_image_embeds[:self.n_users] + explicit_text_embeds[:self.n_users]

            # Z_hat: 通用兴趣表示（所有用户的平均值）
            Z_hat = Z_u.mean(dim=0, keepdim=True)

            # 用户先验: 独特的用户兴趣
            user_prior = Z_u - Z_hat  # shape: (n_users, embedding_dim)

            # 对于物品，不使用个性化指导（零指导）
            item_prior = torch.zeros(self.n_items, self.embedding_dim).to(Z_u.device)

            # 合并用户和物品先验
            full_prior = torch.cat([user_prior, item_prior], dim=0)

            # 使用RF生成器计算损失并更新
            loss_dict = self.rf_generator.compute_loss_and_step(
                start_embeds=rf_start,
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
            # extended_id_embeds = self.rf_generator.mix_embeddings(
            #     extended_id_embeds_target,
            #     rf_embeds.detach(),
            #     training=True,
            #     epoch=self.rf_generator.current_epoch,
            # )
            
            # Store RF outputs for loss computation
            rf_outputs = {"ps_loss": ps_loss}

            # print("rf_embeds:", rf_embeds)
            # 协同信号增强
            origin_weights, gen_weights = torch.split(self.softmax(torch.cat([self.behavior_adaptive_aware(extended_id_embeds), self.behavior_adaptive_aware(rf_embeds),],dim=-1,)),1,dim=-1,)
            extended_id_embeds_aug =  (origin_weights * extended_id_embeds + gen_weights * rf_embeds)
            #print("extended_id_embeds_aug:", extended_id_embeds_aug)

        elif self.use_rf and not train:
            # ===== 推理模式：使用RF生成并混合 =====
            # print("[GenRecv2] Forward in INFERENCE mode")
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
                # origin_weights, gen_weights = torch.split(self.softmax(torch.cat([self.behavior_adaptive_aware(extended_id_embeds), self.behavior_adaptive_aware(rf_embeds),],dim=-1,)),1,dim=-1,)
                # extended_id_embeds =  (origin_weights * extended_id_embeds + gen_weights * rf_embeds)

        else:
            # 不使用RF
            extended_id_embeds = extended_id_embeds_target
            extended_id_embeds_aug = extended_id_embeds

        # ===== 继续其余部分 =====
        # extended_image_embeds = self.conv_ui(
        #     adj, self.extended_image_user.weight, explicit_image_item
        # )
        # extended_text_embeds = self.conv_ui(
        #     adj, self.extended_text_user.weight, explicit_text_item
        # )
        # extended_it_embeds = (extended_image_embeds + extended_text_embeds) / 2
    

        # 模态信号增强
        image_weights, text_weights = torch.split(self.softmax(torch.cat([self.separate_coarse(explicit_image_embeds), self.separate_coarse(explicit_text_embeds),],dim=-1,)),1,dim=-1,)
        coarse_grained_embeds = (image_weights * explicit_image_embeds + text_weights * explicit_text_embeds)



        # extended_it_embeds = coarse_grained_embeds

        # fine_grained_image = torch.multiply(
        #     self.image_behavior(extended_id_embeds),
        #     (explicit_image_embeds - coarse_grained_embeds),
        # )
        # fine_grained_text = torch.multiply(
        #     self.text_behavior(extended_id_embeds),
        #     (explicit_text_embeds - coarse_grained_embeds),
        # )
        # integration_embeds = (
        #     fine_grained_image + fine_grained_text + coarse_grained_embeds
        # ) / 3
        integration_embeds = coarse_grained_embeds

        # all_embeds = extended_id_embeds_aug + integration_embeds
        all_embeds = extended_id_embeds + integration_embeds


        if train and self.use_rf:
            other_outputs = {
                "integration_embeds": integration_embeds,
                "extended_id_embeds": extended_id_embeds,
                "extended_id_embeds_aug": extended_id_embeds_aug,
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
                (integration_embeds, extended_id_embeds, extended_id_embeds_aug),
                (explicit_image_embeds, explicit_text_embeds),
            )

        return all_embeds

    @torch.no_grad()
    def _get_origin_and_generated_embeds(self):
        """获取原始 GCN 兴趣表征与 RF 生成表征，用于验证时计算生成质量指标。"""
        if not self.use_rf:
            return None, None
        adj = self.norm_adj
        # 与 forward 一致的 GCN 与多模态部分
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
        extended_id_embeds = self.conv_ui(adj, user_embeds, item_embeds)
        explicit_image_item = self.conv_ii(self.image_original_adj, image_item_embeds)
        explicit_image_user = torch.sparse.mm(self.R, explicit_image_item)
        explicit_image_embeds = torch.cat([explicit_image_user, explicit_image_item], dim=0)
        explicit_text_item = self.conv_ii(self.text_original_adj, text_item_embeds)
        explicit_text_user = torch.sparse.mm(self.R, explicit_text_item)
        explicit_text_embeds = torch.cat([explicit_text_user, explicit_text_item], dim=0)
        rf_embeds = self.rf_generator.generate(
            [explicit_image_embeds, explicit_text_embeds]
        )
        return extended_id_embeds, rf_embeds

    def compute_gen_quality_metrics(self, n_proj=32, fid_eps=1e-6):
        """
        计算生成兴趣表征质量指标：SWD、平均余弦相似度、MSE、FID。
        原始表征为 GCN 的 extended_id_embeds，生成表征为 rf_embeds。
        Returns:
            dict: gen_swd, gen_cosine, gen_mse, gen_fid（标量，可写入 valid_result）
        """
        origin, gen = self._get_origin_and_generated_embeds()
        if origin is None or gen is None:
            return {}
        origin = origin.detach().float()
        gen = gen.detach().float()
        N, D = origin.shape
        # Sliced Wasserstein Distance（与现有实现一致）
        swd = sliced_wasserstein_distance(origin, gen, n_proj=n_proj, chunk_size=4096)
        swd_val = float(swd.cpu().item())
        # 平均余弦相似度（逐样本）
        o_n = F.normalize(origin, dim=1)
        g_n = F.normalize(gen, dim=1)
        cos_per_sample = (o_n * g_n).sum(dim=1)
        cosine_val = float(cos_per_sample.mean().cpu().item())
        # MSE
        mse_val = float(F.mse_loss(gen, origin).cpu().item())
        # FID（嵌入版）
        fid_val = float(fid_embedding(origin, gen, eps=fid_eps).cpu().item())
        return {
            "gen_swd": swd_val,
            "gen_cosine": cosine_val,
            "gen_mse": mse_val,
            "gen_fid": fid_val,
        }

    @torch.no_grad()
    def get_origin_embedding_for_cross_model(self):
        """
        返回 GenRecv2 的原始兴趣编码（GCN(norm_adj) 的 UI 表征），用作跨模型可比时的共同参考。
        shape: (n_users + n_items, embedding_dim)。
        """
        user_embeds = self.user_embedding.weight
        item_embeds = self.item_id_embedding.weight
        extended_id_embeds = self.conv_ui(self.norm_adj, user_embeds, item_embeds)
        return extended_id_embeds.detach().float()

    @torch.no_grad()
    def compute_trajectory_linearity(self, n_steps=None, return_trajectory_for_viz=False, start_noise=None):
        """
        实验：轨迹线性度 (Trajectory Linearity)。
        公式: (1/T) * sum_t CosSim(x_{t+1}-x_t, x_1-x_0)。RefFlow 应接近 1.0（直线）。start_noise 可指定共用 z_0。
        Returns:
            dict: trajectory_linearity (标量); 若 return_trajectory_for_viz=True 则含 trajectory (T+1,N,D) 供 PCA/t-SNE 可视化。
        """
        if not self.use_rf:
            return {}
        adj = self.norm_adj
        image_item_embeds = torch.multiply(
            self.item_id_embedding.weight,
            self.image_space_trans(self.image_embedding.weight),
        )
        text_item_embeds = torch.multiply(
            self.item_id_embedding.weight,
            self.text_space_trans(self.text_embedding.weight),
        )
        explicit_image_item = self.conv_ii(self.image_original_adj, image_item_embeds)
        explicit_image_user = torch.sparse.mm(self.R, explicit_image_item)
        explicit_image_embeds = torch.cat([explicit_image_user, explicit_image_item], dim=0)
        explicit_text_item = self.conv_ii(self.text_original_adj, text_item_embeds)
        explicit_text_user = torch.sparse.mm(self.R, explicit_text_item)
        explicit_text_embeds = torch.cat([explicit_text_user, explicit_text_item], dim=0)
        out = self.rf_generator.generate(
            [explicit_image_embeds, explicit_text_embeds],
            n_steps=n_steps,
            return_trajectory=True,
            start_noise=start_noise,
            inference_only=True,
        )
        if not isinstance(out, tuple) or len(out) != 2:
            return {}
        z_final, trajectory = out  # (T+1, N, D)
        linearity = trajectory_linearity(trajectory)
        result = {"trajectory_linearity": float(linearity.cpu().item())}
        if return_trajectory_for_viz:
            result["trajectory"] = trajectory.cpu()  # (T+1, N, D) for PCA/t-SNE
        return result

    def compute_gen_quality_metrics_cross(self, origin_ref, n_proj=32, fid_eps=1e-6):
        """
        跨模型可比：本模型「生成」表征 vs 共同参考 origin_ref（通常为 GenRecv2 的原始兴趣编码）。
        origin_ref: (N, D) 与当前数据集 n_users+n_items 一致，D 需与 generated 的维度一致。
        Returns:
            dict: gen_cross_swd, gen_cross_cosine, gen_cross_mse, gen_cross_fid；维度不一致时返回 {}。
        """
        _, gen = self._get_origin_and_generated_embeds()
        if gen is None:
            return {}
        gen = gen.detach().float()
        ref = origin_ref.to(gen.device).float()
        if ref.shape[0] != gen.shape[0] or ref.shape[1] != gen.shape[1]:
            return {}
        o_n = F.normalize(ref, dim=1)
        g_n = F.normalize(gen, dim=1)
        cos_per_sample = (o_n * g_n).sum(dim=1)
        swd_val = float(sliced_wasserstein_distance(ref, gen, n_proj=n_proj, chunk_size=4096).cpu().item())
        cosine_val = float(cos_per_sample.mean().cpu().item())
        mse_val = float(F.mse_loss(gen, ref).cpu().item())
        fid_val = float(fid_embedding(ref, gen, eps=fid_eps).cpu().item())
        return {
            "gen_cross_swd": swd_val,
            "gen_cross_cosine": cosine_val,
            "gen_cross_mse": mse_val,
            "gen_cross_fid": fid_val,
        }

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

            integration_embeds = other_outputs["integration_embeds"] # 模态信号增强
            extended_id_embeds = other_outputs["extended_id_embeds"] # 用户兴趣ID表征向量 H1
            extended_id_embeds_aug = other_outputs["extended_id_embeds_aug"] # 协同信号增强
            explicit_image_embeds = other_outputs["explicit_image_embeds"] # 用户对图像模态的兴趣表征
            explicit_text_embeds = other_outputs["explicit_text_embeds"] # 用户对文本模态的兴趣表征
        else:
            embeds_1, embeds_2, embeds_3 = self.forward(self.norm_adj, train=True)
            integration_embeds, extended_id_embeds, extended_id_embeds_aug = embeds_2 # 模态信号增强, 用户兴趣ID表征向量 H1, 协同信号增强
            explicit_image_embeds, explicit_text_embeds = embeds_3

        users_embeddings, items_embeddings = torch.split(
            embeds_1, [self.n_users, self.n_items], dim=0
        )


        u_g_embeddings = users_embeddings[users]
        pos_i_g_embeddings = items_embeddings[pos_items]
        neg_i_g_embeddings = items_embeddings[neg_items]

        # ===== 通用BPR损失 =====
        bpr_loss, reg_loss_1 = self.bpr_loss(u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings) # 通用BPR损失 必选
        # ===== 模态对齐损失 =====
        vt_loss =  self.align_vt(explicit_image_embeds, explicit_text_embeds) # 模态对齐损失 必选
        # 可选：Sliced Wasserstein（图-文分布对齐），warmup 后再加、权重宜小
        if self.use_wasserstein_vt and getattr(self, "_current_epoch", 0) >= self.wasserstein_warmup_epochs:
            swd_vt = sliced_wasserstein_distance(
                explicit_image_embeds, explicit_text_embeds,
                n_proj=self.wasserstein_n_slices, chunk_size=4096
            )
            vt_loss = vt_loss + self.wasserstein_vt_weight * swd_vt
        #align_gen_loss = self.align_vt(extended_id_embeds, extended_id_embeds_aug)  # 生成与原始对齐

        # ===== 模态信号增强损失 =====
        integration_users, integration_items = torch.split(integration_embeds, [self.n_users, self.n_items], dim=0)
        extended_id_user, extended_id_items = torch.split(extended_id_embeds, [self.n_users, self.n_items], dim=0)
        # bm_loss = self.bm_loss * (self.InfoNCE(integration_users[users], extended_id_user[users], self.bm_temp)
        # + self.InfoNCE(integration_items[pos_items], extended_id_items[pos_items], self.bm_temp))

        # ===== 协同信号增强损失 =====
        extended_id_user_aug, extended_id_items_aug = torch.split(extended_id_embeds_aug, [self.n_users, self.n_items], dim=0)
        cl_loss_aug =  (self.InfoNCE(integration_users[users], extended_id_user_aug[users], self.bm_temp)
        + self.InfoNCE(integration_items[pos_items], extended_id_items_aug[pos_items], self.bm_temp))

        align_loss = self.vt_loss * vt_loss   + cl_loss_aug * self.gen_cl_loss

        # 可选：Sliced Wasserstein 对齐「增强表征」与「原始兴趣表征」；warmup 后加、小权重，可选 detach 原始避免拖累主任务
        if self.use_wasserstein_gen_align and getattr(self, "_current_epoch", 0) >= self.wasserstein_warmup_epochs:
            orig = extended_id_embeds.detach() if self.wasserstein_gen_detach_origin else extended_id_embeds
            swd_gen = sliced_wasserstein_distance(
                orig, extended_id_embeds_aug,
                n_proj=self.wasserstein_n_slices, chunk_size=4096
            )
            align_loss = align_loss + self.wasserstein_gen_weight * swd_gen

        # extended_it_user, extended_it_items = torch.split(extended_it_embeds, [self.n_users, self.n_items], dim=0)
        # c_loss = self.InfoNCE(extended_it_user[users], integration_users[users], self.um_temp)
        # noise_loss_1 = self.cal_noise_loss(users, integration_users, self.um_temp)
        # noise_loss_2 = self.cal_noise_loss(users, extended_it_user, self.um_temp)
        # um_loss = self.um_loss * (c_loss + noise_loss_1 + noise_loss_2)

        # reg_loss_2 = (self.reg_weight_2 * self.sq_sum(extended_it_items[pos_items]) / self.batch_size)
        # reg_loss = reg_loss_1 + reg_loss_2

        # total_loss = bpr_loss + al_loss + um_loss + reg_loss
        total_loss = bpr_loss + align_loss + reg_loss_1 
        
        # Add propensity score loss if denoising is enabled
        if self.use_denoise and self.use_rf and "ps_loss" in other_outputs:
            total_loss = total_loss + self.ps_loss_weight * other_outputs["ps_loss"]

        # ===================================================== 
        #  RF for Cross-Modality Alignment (Task A)
        #  Image -> Text (或双向)
        # ===================================================== 
        # unique_items = torch.unique(torch.cat([pos_items, neg_items])) 
        # # 获取原始特征 (4096/384 维)
        # batch_img = self.image_embedding.weight[unique_items]
        # batch_txt = self.text_embedding.weight[unique_items]
        
        # 计算 Image -> Text 的流损失 
        # RF 模型内部会自动处理投影降维
        # rf_cross_modality_align_loss = self.rf_weight * self.rf_cross_modality_alignment_model.compute_loss(batch_img, batch_txt) 

        # Note: cl_loss is now always computed in rf_modules.py via compute_loss_and_step()

        return total_loss 

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
