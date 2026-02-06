# Rectified Flow for GUME: RF-GUME
# Integrating Rectified Flow into GUME for generating extended_id_embeds

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.gume import GUME


class SimpleVelocityNet(nn.Module):
    """
    简单的速度场网络，用于Rectified Flow

    Args:
        embedding_dim: 嵌入维度
        hidden_dim: 隐藏层维度
        n_layers: 网络层数
        dropout: dropout率
        condition_dim: 条件维度（image + text）
    """

    def __init__(self, embedding_dim, hidden_dim, n_layers, dropout, condition_dim):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        # 时间嵌入层（使用正弦位置编码）
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbedding(64),
            nn.Linear(64, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

        # 条件编码器
        self.condition_encoder = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

        # 输入投影
        self.input_proj = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

        # 残差块
        self.res_blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim, hidden_dim, dropout) for _ in range(n_layers)]
        )

        # 输出投影
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, x, t, conditions):
        """
        Args:
            x: 当前状态 X_t, shape (batch, embedding_dim)
            t: 时间步, shape (batch, 1)
            conditions: 条件特征（image + text）, shape (batch, condition_dim)

        Returns:
            velocity: 速度 v(x, t, c), shape (batch, embedding_dim)
        """
        # 时间嵌入
        t_emb = self.time_embed(t)  # (batch, hidden_dim)

        # 条件嵌入
        cond_emb = self.condition_encoder(conditions)  # (batch, hidden_dim)

        # 输入投影
        h = self.input_proj(x)  # (batch, hidden_dim)

        # 融合时间和条件
        h = h + t_emb + cond_emb

        # 通过残差块
        for res_block in self.res_blocks:
            h = res_block(h)

        # 输出速度
        v = self.output_proj(h)

        return v


class ResidualBlock(nn.Module):
    """简单的残差块"""

    def __init__(self, in_dim, out_dim, dropout):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
            nn.LayerNorm(out_dim),
        )

        self.skip = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x):
        return F.silu(self.net(x) + self.skip(x))


class SinusoidalPositionEmbedding(nn.Module):
    """正弦位置编码（用于时间步）"""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        """
        Args:
            t: 时间步, shape (batch, 1)
        Returns:
            embeddings: shape (batch, dim)
        """
        device = t.device
        half_dim = self.dim // 2

        embeddings = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t * embeddings[None, :]

        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)

        return embeddings


class RFExtendedIdGenerator(nn.Module):
    """
    使用Rectified Flow生成extended_id_embeds

    这个模块独立于GUME的其他部分，便于进行消融实验
    """

    def __init__(self, embedding_dim, hidden_dim, n_layers, dropout):
        super().__init__()

        self.embedding_dim = embedding_dim

        # 速度场网络（条件维度 = embedding_dim * 2，因为有image和text两个条件）
        self.velocity_net = SimpleVelocityNet(
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            dropout=dropout,
            condition_dim=embedding_dim * 2,  # image + text conditions
        )

    def rectified_flow_loss(self, target_embeds, image_cond, text_cond):
        """
        计算Rectified Flow损失

        Loss = E_t [||v(X_t, t, c) - (X1 - X0)||^2]
        其中 X_t = t*X1 + (1-t)*X0

        Args:
            target_embeds: 目标的extended_id_embeds (X1), shape (batch, embedding_dim)
            image_cond: explicit_image_item 条件, shape (batch, embedding_dim)
            text_cond: explicit_text_item 条件, shape (batch, embedding_dim)

        Returns:
            rf_loss: Rectified Flow损失
        """
        batch_size = target_embeds.shape[0]

        # X1 = 目标嵌入
        X1 = target_embeds

        # X0 = 随机噪声
        X0 = torch.randn_like(X1)

        # 随机采样时间步 t ~ Uniform[0, 1]
        t = torch.rand(batch_size, 1).to(X1.device)

        # 线性插值: X_t = t*X1 + (1-t)*X0
        X_t = t * X1 + (1 - t) * X0

        # 拼接条件（image + text）
        conditions = torch.cat([image_cond, text_cond], dim=-1)

        # 预测速度: v(X_t, t, conditions)
        v_pred = self.velocity_net(X_t, t, conditions)

        # 目标速度: X1 - X0（直线方向）
        v_target = X1 - X0

        # Rectified Flow损失（MSE）
        rf_loss = F.mse_loss(v_pred, v_target)

        return rf_loss

    def sample_ode(self, z_0, image_cond, text_cond, n_steps=10):
        """
        使用Euler方法求解ODE生成extended_id_embeds

        dZ_t = v(Z_t, t, conditions) dt

        Args:
            z_0: 初始噪声, shape (batch, embedding_dim)
            image_cond: explicit_image_item 条件
            text_cond: explicit_text_item 条件
            n_steps: ODE求解步数

        Returns:
            z_1: 生成的extended_id_embeds
        """
        z_t = z_0
        dt = 1.0 / n_steps

        # 拼接条件
        conditions = torch.cat([image_cond, text_cond], dim=-1)

        # Euler方法求解ODE
        for i in range(n_steps):
            t = torch.full((z_0.shape[0], 1), i * dt).to(z_0.device)
            v = self.velocity_net(z_t, t, conditions)
            z_t = z_t + v * dt

        return z_t

    def forward(self, image_cond, text_cond, n_steps=10):
        """
        前向传播：生成extended_id_embeds

        Args:
            image_cond: explicit_image_item 条件
            text_cond: explicit_text_item 条件
            n_steps: ODE求解步数

        Returns:
            generated_embeds: 生成的extended_id_embeds
        """
        batch_size = image_cond.shape[0]

        # 从标准高斯噪声开始
        z_0 = torch.randn(batch_size, self.embedding_dim).to(image_cond.device)

        # 求解ODE生成嵌入
        generated_embeds = self.sample_ode(z_0, image_cond, text_cond, n_steps)

        return generated_embeds


class RFGUME(GUME):
    """
    GUME with Rectified Flow

    在GUME的基础上，使用Rectified Flow生成extended_id_embeds
    """

    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        # RF相关参数
        self.use_rf = config["use_rf"] if "use_rf" in config else True
        self.rf_hidden_dim = (
            config["rf_hidden_dim"] if "rf_hidden_dim" in config else 256
        )
        self.rf_n_layers = config["rf_n_layers"] if "rf_n_layers" in config else 2
        self.rf_dropout = config["rf_dropout"] if "rf_dropout" in config else 0.1
        self.rf_loss_weight = (
            config["rf_loss_weight"] if "rf_loss_weight" in config else 1.0
        )
        self.contrast_loss_weight = (
            config["contrast_loss_weight"] if "contrast_loss_weight" in config else 0.1
        )
        self.rf_sampling_steps = (
            config["rf_sampling_steps"] if "rf_sampling_steps" in config else 10
        )

        if self.use_rf:
            # 为user和item分别创建RF生成器（低耦合设计）
            self.rf_user_generator = RFExtendedIdGenerator(
                embedding_dim=self.embedding_dim,
                hidden_dim=self.rf_hidden_dim,
                n_layers=self.rf_n_layers,
                dropout=self.rf_dropout,
            )

            self.rf_item_generator = RFExtendedIdGenerator(
                embedding_dim=self.embedding_dim,
                hidden_dim=self.rf_hidden_dim,
                n_layers=self.rf_n_layers,
                dropout=self.rf_dropout,
            )

            print(
                f"RF-GUME initialized with RF generators: hidden_dim={self.rf_hidden_dim}, "
                f"n_layers={self.rf_n_layers}, sampling_steps={self.rf_sampling_steps}"
            )

    def forward_with_rf(self, adj, train=False):
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
        if self.use_rf and train:
            # 训练模式：使用RF生成embeds，这样RF生成器才能通过梯度学习
            # 生成user embeds
            rf_user_embeds = self.rf_user_generator(
                explicit_image_user, explicit_text_user, n_steps=self.rf_sampling_steps
            )

            # 生成item embeds
            rf_item_embeds = self.rf_item_generator(
                explicit_image_item, explicit_text_item, n_steps=self.rf_sampling_steps
            )

            # 使用RF生成的embeds替换原始的
            # 这样梯度可以回传到RF生成器
            extended_id_embeds = torch.cat([rf_user_embeds, rf_item_embeds], dim=0)

            # 保存生成的embeds用于计算对比损失
            rf_generated_user = rf_user_embeds
            rf_generated_item = rf_item_embeds

        elif self.use_rf and not train:
            # 推理模式：使用RF生成
            with torch.no_grad():
                # 生成user embeds
                rf_user_embeds = self.rf_user_generator(
                    explicit_image_user,
                    explicit_text_user,
                    n_steps=self.rf_sampling_steps,
                )

                # 生成item embeds
                rf_item_embeds = self.rf_item_generator(
                    explicit_image_item,
                    explicit_text_item,
                    n_steps=self.rf_sampling_steps,
                )

                extended_id_embeds = torch.cat([rf_user_embeds, rf_item_embeds], dim=0)
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

        if train:
            # 返回RF相关的输出
            rf_outputs = {
                "extended_id_embeds_target": extended_id_embeds_target,
                "explicit_image_user": explicit_image_user,
                "explicit_image_item": explicit_image_item,
                "explicit_text_user": explicit_text_user,
                "explicit_text_item": explicit_text_item,
            }

            # 如果使用RF，添加生成的embeds
            if self.use_rf:
                rf_outputs["rf_user_embeds"] = rf_generated_user
                rf_outputs["rf_item_embeds"] = rf_generated_item

            other_outputs = {
                "integration_embeds": integration_embeds,
                "extended_id_embeds": extended_id_embeds,
                "extended_it_embeds": extended_it_embeds,
                "explicit_image_embeds": explicit_image_embeds,
                "explicit_text_embeds": explicit_text_embeds,
            }

            return all_embeds, rf_outputs, other_outputs

        return all_embeds

    def calculate_rf_loss(self, rf_outputs):
        """
        计算RF相关的损失

        Args:
            rf_outputs: RF相关的输出

        Returns:
            rf_loss: Rectified Flow损失
            contrast_loss: 生成的和原始的extended_id_embeds的对比损失
        """
        # 分离user和item的目标embeds
        extended_id_embeds_target = rf_outputs["extended_id_embeds_target"]
        user_target, item_target = torch.split(
            extended_id_embeds_target, [self.n_users, self.n_items], dim=0
        )

        # User的RF损失 - 这是核心损失，学习速度场
        user_rf_loss = self.rf_user_generator.rectified_flow_loss(
            user_target,
            rf_outputs["explicit_image_user"],
            rf_outputs["explicit_text_user"],
        )

        # Item的RF损失
        item_rf_loss = self.rf_item_generator.rectified_flow_loss(
            item_target,
            rf_outputs["explicit_image_item"],
            rf_outputs["explicit_text_item"],
        )

        # 总的RF损失
        rf_loss = (user_rf_loss + item_rf_loss) / 2

        # 对比损失：生成的embeds（已经在forward中生成）应该接近目标embeds
        # 这个损失作为额外的约束，确保RF生成的embeds质量
        rf_user_embeds = rf_outputs["rf_user_embeds"]
        rf_item_embeds = rf_outputs["rf_item_embeds"]

        # InfoNCE对比损失
        # 使用与GUME相同的temperature参数
        user_contrast_loss = self.InfoNCE(rf_user_embeds, user_target, self.bm_temp)
        item_contrast_loss = self.InfoNCE(rf_item_embeds, item_target, self.bm_temp)
        contrast_loss = (user_contrast_loss + item_contrast_loss) / 2

        return rf_loss, contrast_loss

    def calculate_loss(self, interaction):
        """
        计算总损失（包括原始GUME损失和RF损失）

        Args:
            interaction: 交互数据

        Returns:
            total_loss: 总损失
        """
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]

        # 前向传播
        if self.use_rf:
            embeds_1, rf_outputs, other_outputs = self.forward_with_rf(
                self.norm_adj, train=True
            )

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

        # 原始GUME的总损失
        gume_loss = bpr_loss + al_loss + um_loss + reg_loss

        # ===== RF损失（如果使用RF）=====
        if self.use_rf:
            rf_loss, contrast_loss = self.calculate_rf_loss(rf_outputs)

            # 总损失 = GUME损失 + RF损失 + 对比损失
            total_loss = (
                gume_loss
                + self.rf_loss_weight * rf_loss
                + self.contrast_loss_weight * contrast_loss
            )

            return total_loss
        else:
            return gume_loss

    def full_sort_predict(self, interaction):
        """
        预测（推理模式）
        """
        user = interaction[0]

        if self.use_rf:
            all_embeds = self.forward_with_rf(self.norm_adj, train=False)
        else:
            all_embeds = self.forward(self.norm_adj)

        restore_user_e, restore_item_e = torch.split(
            all_embeds, [self.n_users, self.n_items], dim=0
        )
        u_embeddings = restore_user_e[user]

        scores = torch.matmul(u_embeddings, restore_item_e.transpose(0, 1))
        return scores
