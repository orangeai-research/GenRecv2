"""
Rectified Flow Pluggable Module

This module provides a standalone, reusable implementation of Rectified Flow (RF)
that can be easily integrated into any recommendation network to enhance embeddings.

Key Components:
- SinusoidalPositionEmbedding: Time step encoding
- ResidualBlock: Residual block for network depth
- SimpleVelocityNet: Velocity field network for RF
- cosine_similarity_gradient: Cosine similarity gradient computation
- RFEmbeddingGenerator: Main pluggable RF generator class

Usage Example:
    ```python
    from models.rf_modules import RFEmbeddingGenerator

    # Initialize RF generator
    rf_gen = RFEmbeddingGenerator(
        embedding_dim=64,
        hidden_dim=256,
        n_layers=2,
        learning_rate=0.001,
    )

    # Training: compute loss and update
    loss_dict = rf_gen.compute_loss_and_step(
        target_embeds=target,
        conditions=[image_cond, text_cond],
        user_prior=prior,
        epoch=current_epoch,
    )

    # Generate embeddings
    rf_embeds = rf_gen.generate([image_cond, text_cond])

    # Mix with original embeddings
    final_embeds = rf_gen.mix_embeddings(
        original_embeds, rf_embeds, training=True
    )
    ```
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict


def cosine_similarity_gradient(x_t: torch.Tensor, x_1: torch.Tensor) -> torch.Tensor:
    """
    Compute the gradient of cosine similarity cos_sim(x_t, x_1) with respect to x_t.

    Gradient formula: ∇_{x_t} cos_sim(x_t, x_1) = (x_1 / ||x_1||) / ||x_t|| - (x_t / ||x_t||) * cos_sim / ||x_t||

    Args:
        x_t: Current state, shape (batch, embedding_dim)
        x_1: Target state, shape (batch, embedding_dim)

    Returns:
        grad: Cosine similarity gradient, shape (batch, embedding_dim)
    """
    # Compute cosine similarity
    cos_sim = F.cosine_similarity(x_t, x_1, dim=-1)  # (batch,)
    cos_sim = cos_sim.unsqueeze(-1)  # (batch, 1)

    # Compute norms
    x_t_norm = x_t.norm(dim=-1, keepdim=True)  # (batch, 1)
    x_t_norm = torch.clamp(x_t_norm, min=1e-8)  # Avoid division by zero

    # Normalized vectors
    x_1_normalized = F.normalize(x_1, dim=-1)  # (batch, embedding_dim)
    x_t_normalized = F.normalize(x_t, dim=-1)  # (batch, embedding_dim)

    # Compute gradient
    grad = x_1_normalized / x_t_norm - x_t_normalized * cos_sim / x_t_norm

    return grad


class PropensityScoreEstimator(nn.Module):
    """
    Propensity Score Estimator for causal denoising.

    Estimates the probability that an interaction is clean:
    e_{u,i} = P(T_{u,i}=1 | S_{u,i}) = sigmoid(alpha * S_{u,i} + beta)

    where:
    - T_{u,i} = 1 means clean interaction (rating >= threshold)
    - T_{u,i} = 0 means noisy interaction (rating < threshold)
    - S_{u,i} is the user-item similarity score
    """

    def __init__(self):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(0.0))

    def forward(self, similarity_scores: torch.Tensor) -> torch.Tensor:
        """
        Compute propensity scores.

        Args:
            similarity_scores: User-item similarity scores S_{u,i}, shape (n_interactions,)

        Returns:
            e_scores: Propensity scores e_{u,i}, shape (n_interactions,)
        """
        return torch.sigmoid(self.alpha * similarity_scores + self.beta)

    def compute_loss(self, e_scores: torch.Tensor, treatment_labels: torch.Tensor) -> torch.Tensor:
        """
        Compute cross-entropy loss for propensity score training.

        L_PS = -sum[T * log(e) + (1-T) * log(1-e)]

        Args:
            e_scores: Predicted propensity scores, shape (n_interactions,)
            treatment_labels: True treatment labels T (0 or 1), shape (n_interactions,)

        Returns:
            loss: Binary cross-entropy loss
        """
        return F.binary_cross_entropy(e_scores, treatment_labels)


class CausalDenoiser(nn.Module):
    """
    Causal Denoiser using Inverse Propensity Weighting (IPW).

    Implements the causal denoising formula:
    h_u^{(l+1)} = ReLU(Σ_{i∈N(u)} (T_{u,i}/e_{u,i}) * W^{(l)} * h_i^{(l)} + b^{(l)})

    where:
    - T_{u,i} = 1 for clean interactions (rating >= threshold), 0 otherwise
    - e_{u,i} = P(T=1|S_{u,i}) is the propensity score
    - T/e is the Inverse Propensity Weight (IPW)
    """

    def __init__(
        self,
        embedding_dim: int,
        n_users: int,
        n_items: int,
        n_layers: int = 2,
        clean_rating_threshold: float = 5.0,
        device: torch.device = None,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.n_users = n_users
        self.n_items = n_items
        self.n_layers = n_layers
        self.clean_rating_threshold = clean_rating_threshold
        self.device = device if device is not None else torch.device('cpu')

        # Propensity score estimator
        self.propensity_estimator = PropensityScoreEstimator()

        # Learnable GCN weights for denoising
        self.denoise_W = nn.ModuleList([
            nn.Linear(embedding_dim, embedding_dim, bias=True)
            for _ in range(n_layers)
        ])
        for layer in self.denoise_W:
            nn.init.xavier_normal_(layer.weight)

        # Will be initialized when load_treatment_labels is called
        self.treatment_matrix = None
        self.denoise_user_ids = None
        self.denoise_item_ids = None
        self.denoise_treatments = None

    def load_treatment_labels(self, dataset):
        """
        Load rating-based treatment labels T_{u,i} from dataset.
        T_{u,i} = 1 if rating >= threshold (clean), else 0 (noisy)

        Args:
            dataset: Dataset object containing df with ratings
        """
        import numpy as np
        import scipy.sparse as sp

        if not hasattr(dataset, 'df') or dataset.df is None:
            self.treatment_matrix = None
            return

        inter_df = dataset.df
        rating_field = dataset.rating_field if hasattr(dataset, 'rating_field') else None

        if rating_field and rating_field in inter_df.columns:
            ratings = inter_df[rating_field].values
            treatments = (ratings >= self.clean_rating_threshold).astype(np.float32)

            user_ids = inter_df[dataset.uid_field].values
            item_ids = inter_df[dataset.iid_field].values

            # Store as sparse CSR matrix for efficient lookup
            self.treatment_matrix = sp.coo_matrix(
                (treatments, (user_ids, item_ids)),
                shape=(self.n_users, self.n_items)
            ).tocsr()

            # Store interaction indices for IPW computation
            self.denoise_user_ids = torch.LongTensor(user_ids).to(self.device)
            self.denoise_item_ids = torch.LongTensor(item_ids).to(self.device)
            self.denoise_treatments = torch.FloatTensor(treatments).to(self.device)
        else:
            self.treatment_matrix = None

    def forward(self, ego_embeddings: torch.Tensor) -> tuple:
        """
        IPW-weighted GCN aggregation for denoising.

        Args:
            ego_embeddings: Original ego embeddings [n_users + n_items, D]

        Returns:
            denoised_emb: Denoised embeddings [n_users + n_items, D]
            ps_loss: Propensity score cross-entropy loss
        """
        if self.treatment_matrix is None:
            return None, 0.0

        u_emb, i_emb = torch.split(ego_embeddings, [self.n_users, self.n_items], dim=0)

        # Step 2: Compute propensity scores
        # S_{u,i} = cosine_similarity(u, i) for observed interactions
        u_norm = F.normalize(u_emb, dim=1)
        i_norm = F.normalize(i_emb, dim=1)

        # Compute similarity for observed interaction pairs
        sim_scores = (u_norm[self.denoise_user_ids] * i_norm[self.denoise_item_ids]).sum(dim=1)

        # Propensity score estimation: e_{u,i} = sigmoid(alpha * S + beta)
        e_scores = self.propensity_estimator(sim_scores)

        # Propensity score loss: cross-entropy
        ps_loss = self.propensity_estimator.compute_loss(e_scores, self.denoise_treatments)

        # Step 3: IPW weights - T_{u,i} / e_{u,i}
        # For T=0 (noisy): weight = 0 (ignored)
        # For T=1 (clean): weight = 1/e (upweighted)
        ipw_weights = self.denoise_treatments / (e_scores.detach() + 1e-8)

        # Build weighted adjacency matrix for user-item bipartite graph
        n_nodes = self.n_users + self.n_items

        # User -> Item edges (user rows, item cols + n_users)
        row_u2i = self.denoise_user_ids
        col_u2i = self.denoise_item_ids + self.n_users

        # Item -> User edges (item rows + n_users, user cols)
        row_i2u = self.denoise_item_ids + self.n_users
        col_i2u = self.denoise_user_ids

        # Combine edges
        row_indices = torch.cat([row_u2i, row_i2u])
        col_indices = torch.cat([col_u2i, col_i2u])
        ipw_values = torch.cat([ipw_weights, ipw_weights])

        # Create sparse weighted adjacency matrix
        indices = torch.stack([row_indices, col_indices], dim=0)
        weighted_adj = torch.sparse_coo_tensor(
            indices, ipw_values, size=(n_nodes, n_nodes)
        ).coalesce()

        # Degree normalization: D^{-0.5} A D^{-0.5}
        degree = torch.sparse.sum(weighted_adj, dim=1).to_dense() + 1e-8
        d_inv_sqrt = torch.pow(degree, -0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0

        # Apply weighted GCN layers with learnable W and b
        current_emb = ego_embeddings
        all_embeddings = [current_emb]

        for l in range(self.n_layers):
            # Message passing: A * h
            msg = torch.sparse.mm(weighted_adj, current_emb)

            # Apply symmetric normalization
            msg = d_inv_sqrt.unsqueeze(1) * msg

            # Apply learnable transformation: W * h + b, then ReLU
            current_emb = self.denoise_W[l](msg)
            current_emb = F.relu(current_emb)
            all_embeddings.append(current_emb)

        # Mean pooling across layers
        denoised_emb = torch.stack(all_embeddings, dim=1).mean(dim=1)

        return denoised_emb, ps_loss


class SinusoidalPositionEmbedding(nn.Module):
    """Sinusoidal position encoding for time steps."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: Time step, shape (batch, 1)
        Returns:
            embeddings: shape (batch, dim)
        """
        device = t.device
        half_dim = self.dim // 2

        # 确保所有 tensor 都在同一设备上
        embeddings = torch.log(torch.tensor(10000.0, device=device)) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t * embeddings[None, :]

        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)

        return embeddings


class ResidualBlock(nn.Module):
    """Simple residual block with LayerNorm and SiLU activation."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(self.net(x) + self.skip(x))


class SimpleVelocityNet(nn.Module):
    """
    Simple velocity field network for Rectified Flow.

    Args:
        embedding_dim: Embedding dimension
        hidden_dim: Hidden layer dimension
        n_layers: Number of layers
        dropout: Dropout rate
        condition_dim: Condition dimension (sum of all condition embeddings)
        user_guidance_scale: User prior guidance scaling factor
        guidance_decay_power: User prior decay power
        cosine_guidance_scale: Cosine gradient guidance scaling factor
        cosine_decay_power: Cosine gradient decay power
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        n_layers: int,
        dropout: float,
        condition_dim: int,
        user_guidance_scale: float = 0.2,
        guidance_decay_power: float = 2.0,
        cosine_guidance_scale: float = 0.1,
        cosine_decay_power: float = 2.0,
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.user_guidance_scale = user_guidance_scale
        self.guidance_decay_power = guidance_decay_power
        self.cosine_guidance_scale = cosine_guidance_scale
        self.cosine_decay_power = cosine_decay_power

        # Time embedding layer (using sinusoidal position encoding)
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbedding(64),
            nn.Linear(64, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

        # Condition encoder
        self.condition_encoder = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

        # Residual blocks
        self.res_blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim, hidden_dim, dropout) for _ in range(n_layers)]
        )

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        conditions: torch.Tensor,
        user_prior: Optional[torch.Tensor] = None,
        x_1: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Current state X_t, shape (batch, embedding_dim)
            t: Time step, shape (batch, 1)
            conditions: Condition features (concatenated), shape (batch, condition_dim)
            user_prior: User-specific interest guidance, shape (batch, embedding_dim)
            x_1: Target state X_1, shape (batch, embedding_dim) - for cosine gradient

        Returns:
            velocity: Velocity v(x, t, c), shape (batch, embedding_dim)
        """
        # Time embedding
        t_emb = self.time_embed(t)  # (batch, hidden_dim)

        # Condition embedding
        cond_emb = self.condition_encoder(conditions)  # (batch, hidden_dim)

        # Input projection
        h = self.input_proj(x)  # (batch, hidden_dim)

        # Fuse time and condition
        h = h + t_emb + cond_emb

        # Pass through residual blocks
        for res_block in self.res_blocks:
            h = res_block(h)

        # Output velocity
        v = self.output_proj(h)

        # Add prior knowledge guidance (training mode only)
        if self.training:
            # First term: User interest prior guidance
            if user_prior is not None:
                # Time decay weight: lambda_1(t) = (1-t)^power
                lambda_1 = (1 - t) ** self.guidance_decay_power  # shape: (batch, 1)

                # Add user prior guidance term
                v = v + lambda_1 * self.user_guidance_scale * user_prior

            # Second term: Cosine similarity gradient guidance
            if x_1 is not None:
                # Time decay weight: lambda_2(t) = (1-t)^power
                lambda_2 = (1 - t) ** self.cosine_decay_power  # shape: (batch, 1)

                # Compute cosine similarity gradient
                cos_grad = cosine_similarity_gradient(x, x_1)

                # Add cosine gradient guidance term
                v = v + lambda_2 * self.cosine_guidance_scale * cos_grad

        return v


class RFEmbeddingGenerator(nn.Module):
    """
    Pluggable Rectified Flow Embedding Generator with 2-RF support.

    This is the main class for integrating RF into recommendation networks.
    It provides a clean API for training and inference, with support for
    2-Rectified Flow (Reflow) and gradient checkpointing for memory efficiency.

    Args:
        embedding_dim: Embedding dimension
        hidden_dim: Hidden layer dimension (default: 256)
        n_layers: Number of network layers (default: 2)
        dropout: Dropout rate (default: 0.1)
        learning_rate: Independent learning rate for RF (default: 0.001)
        sampling_steps: ODE sampling steps (default: 10)
        user_guidance_scale: User prior scaling factor (default: 0.2)
        guidance_decay_power: User prior decay power (default: 2.0)
        cosine_guidance_scale: Cosine gradient scaling factor (default: 0.1)
        cosine_decay_power: Cosine gradient decay power (default: 2.0)
        warmup_epochs: Warmup epochs (default: 0)
        train_mix_ratio: Training mix ratio (default: 0.5)
        inference_mix_ratio: Inference mix ratio (default: 0.5)
        contrast_temp: Contrastive loss temperature (default: 0.2)
        contrast_weight: Contrastive loss weight (default: 1.0)
        n_users: Number of users (required for interaction-based contrastive loss)
        n_items: Number of items (required for interaction-based contrastive loss)
        infonce_negative_samples: Number of negative samples for InfoNCE (default: 1024)
        infonce_batch_size: Batch size for chunked InfoNCE processing (default: 4096)
        use_2rf: Enable 2-Rectified Flow (Reflow) training (default: True)
        rf_2rf_transition_epoch: Epoch to transition from 1-RF to 2-RF (default: warmup_epochs + 5)
        use_gradient_checkpointing: Enable gradient checkpointing to save memory (default: True)
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int = 256,
        n_layers: int = 2,
        dropout: float = 0.1,
        learning_rate: float = 0.001,
        sampling_steps: int = 10,
        user_guidance_scale: float = 0.2,
        guidance_decay_power: float = 2.0,
        cosine_guidance_scale: float = 0.1,
        cosine_decay_power: float = 2.0,
        warmup_epochs: int = 0,
        train_mix_ratio: float = 0.5,
        inference_mix_ratio: float = 0.5,
        contrast_temp: float = 0.2,
        contrast_weight: float = 1.0,
        n_users: int = 0,
        n_items: int = 0,
        # InfoNCE negative sampling parameters
        infonce_negative_samples: int = 1024,  # Number of negative samples
        infonce_batch_size: int = 4096,        # Batch size for chunked processing
        # 2-RF parameters
        use_2rf: bool = True,
        rf_2rf_transition_epoch: Optional[int] = None,
        # Memory optimization
        use_gradient_checkpointing: bool = True,
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.sampling_steps = sampling_steps
        self.user_guidance_scale = user_guidance_scale
        self.guidance_decay_power = guidance_decay_power
        self.cosine_guidance_scale = cosine_guidance_scale
        self.cosine_decay_power = cosine_decay_power
        self.warmup_epochs = warmup_epochs
        self.train_mix_ratio = train_mix_ratio
        self.inference_mix_ratio = inference_mix_ratio
        self.contrast_temp = contrast_temp
        self.contrast_weight = contrast_weight
        self.n_users = n_users
        self.n_items = n_items

        # InfoNCE negative sampling parameters
        self.infonce_negative_samples = infonce_negative_samples
        self.infonce_batch_size = infonce_batch_size

        # 2-RF parameters
        self.use_2rf = use_2rf
        self.rf_2rf_transition_epoch = rf_2rf_transition_epoch or (warmup_epochs + 5)
        self.is_2rf_active = False

        # Debug: Print 2-RF configuration
        print(f"[RF Init] use_2rf={self.use_2rf}, transition_epoch={self.rf_2rf_transition_epoch}, warmup={warmup_epochs}")

        # Cache for 2-RF reflow dataset
        self.reflow_z0 = None
        self.reflow_z1 = None
        self.reflow_conditions = None

        # Memory optimization
        self.use_gradient_checkpointing = use_gradient_checkpointing

        # Will be set when we know the condition dimension
        self.velocity_net = None
        self.optimizer = None

        # Current epoch (updated externally)
        self.current_epoch = 0

    def _init_velocity_net(self, condition_dim: int, device: torch.device):
        """Initialize velocity network (called on first forward pass)."""
        if self.velocity_net is None:
            self.velocity_net = SimpleVelocityNet(
                embedding_dim=self.embedding_dim,
                hidden_dim=self.hidden_dim,
                n_layers=self.n_layers,
                dropout=self.dropout,
                condition_dim=condition_dim,
                user_guidance_scale=self.user_guidance_scale,
                guidance_decay_power=self.guidance_decay_power,
                cosine_guidance_scale=self.cosine_guidance_scale,
                cosine_decay_power=self.cosine_decay_power,
            )

            # Move velocity_net to the correct device
            self.velocity_net = self.velocity_net.to(device)

            # Initialize optimizer
            self.optimizer = torch.optim.AdamW(
                self.velocity_net.parameters(),
                lr=self.learning_rate
            )

    def set_epoch(self, epoch: int):
        """
        Update current epoch and handle 1-RF → 2-RF transition.

        Args:
            epoch: Current training epoch
        """
        self.current_epoch = epoch

        # Automatic transition to 2-RF after warmup + stabilization
        if self.use_2rf and epoch >= self.rf_2rf_transition_epoch and not self.is_2rf_active:
            self.is_2rf_active = True
            print(f"\n{'='*60}")
            print(f"[2-RF] Transitioning to 2-Rectified Flow at epoch {epoch}")
            print(f"[2-RF] Will use 1-RF outputs as new starting distribution")
            print(f"{'='*60}\n")

    def _rectified_flow_loss(
        self,
        target_embeds: torch.Tensor,
        conditions: torch.Tensor,
        user_prior: Optional[torch.Tensor] = None,
        fixed_noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute Rectified Flow loss with optional user interest prior guidance.

        Loss = E_t [||v(X_t, t, c) - (X1 - X0)||^2]
        where X_t = t*X1 + (1-t)*X0

        For 2-RF (Reflow) training, pass fixed_noise to enforce strict coupling (Z0, Z1).

        Args:
            target_embeds: Target embeddings (X1), shape (batch, embedding_dim)
            conditions: Concatenated condition embeddings, shape (batch, condition_dim)
            user_prior: User interest prior guidance, shape (batch, embedding_dim)
            fixed_noise: Fixed noise for 2-RF training. If None, use random noise (1-RF).
                         Shape (batch, embedding_dim)

        Returns:
            rf_loss: Rectified Flow loss
        """
        batch_size = target_embeds.shape[0]

        # X1 = target embeddings
        X1 = target_embeds

        # X0 = fixed noise (2-RF) or random noise (1-RF)
        if fixed_noise is not None:
            X0 = fixed_noise
        else:
            X0 = torch.randn_like(X1)

        # Sample time step t ~ Uniform[0, 1]
        t = torch.rand(batch_size, 1).to(X1.device)

        # Linear interpolation: X_t = t*X1 + (1-t)*X0
        X_t = t * X1 + (1 - t) * X0

        # Predict velocity: v(X_t, t, conditions, user_prior, X1)
        v_pred = self.velocity_net(X_t, t, conditions, user_prior=user_prior, x_1=X1)

        # Target velocity: X1 - X0 (straight line direction)
        v_target = X1 - X0

        # Rectified Flow loss (MSE)
        rf_loss = F.mse_loss(v_pred, v_target)

        return rf_loss

    def _infonce_loss(
        self,
        view1: torch.Tensor,
        view2: torch.Tensor,
        temperature: float,
    ) -> torch.Tensor:
        """
        Compute InfoNCE contrastive loss (matching mgcn.py implementation).

        Args:
            view1: First view embeddings
            view2: Second view embeddings
            temperature: Temperature parameter

        Returns:
            loss: InfoNCE loss
        """
        # Normalize embeddings
        view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)

        # Positive score: element-wise product and sum
        pos_score = (view1 * view2).sum(dim=-1)
        pos_score = torch.exp(pos_score / temperature)

        # Total score: matrix multiplication
        ttl_score = torch.matmul(view1, view2.transpose(0, 1))
        ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)

        # InfoNCE loss
        cl_loss = -torch.log(pos_score / ttl_score)

        return torch.mean(cl_loss)


    def _infonce_loss_interaction_based(
        self,
        rf_embeds: torch.Tensor,
        target_embeds: torch.Tensor,
        pos_indices: torch.Tensor,
        temperature: float,
        n_negatives: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Compute InfoNCE loss between RF-generated embeddings and target embeddings,
        using interaction-based positive samples and randomly sampled negatives.

        Args:
            rf_embeds: RF-generated embeddings [N, D]
            target_embeds: Target embeddings [N, D]
            pos_indices: Indices of positive samples from interactions [batch]
            temperature: Temperature parameter
            n_negatives: Number of negative samples per positive (default: self.infonce_negative_samples)

        Returns:
            loss: InfoNCE loss value
        """
        n_negatives = n_negatives if n_negatives is not None else self.infonce_negative_samples

        N = target_embeds.size(0)
        device = rf_embeds.device
        batch_size = pos_indices.size(0)

        # Get positive pairs: (rf_embeds[pos_ids], target_embeds[pos_ids])
        rf_pos = F.normalize(rf_embeds[pos_indices], dim=1)  # [batch, D]
        target_pos = F.normalize(target_embeds[pos_indices], dim=1)  # [batch, D]

        # Positive scores: element-wise product and sum
        pos_score = (rf_pos * target_pos).sum(dim=-1)  # [batch]
        pos_score = torch.exp(pos_score / temperature)

        # Sample negative indices from target_embeds
        neg_indices = torch.randint(0, N, (batch_size, n_negatives), device=device)

        # Ensure negatives don't include positive indices
        pos_indices_expanded = pos_indices.unsqueeze(1)  # [batch, 1]
        mask = (neg_indices == pos_indices_expanded)
        neg_indices = torch.where(mask, (neg_indices + 1) % N, neg_indices)

        # Get negative embeddings from target_embeds
        target_neg = F.normalize(target_embeds[neg_indices], dim=1)  # [batch, n_neg, D]

        # Compute negative scores: [batch, n_neg]
        neg_scores = torch.bmm(
            rf_pos.unsqueeze(1),  # [batch, 1, D]
            target_neg.transpose(1, 2)  # [batch, D, n_neg]
        ).squeeze(1)  # [batch, n_neg]
        neg_scores = torch.exp(neg_scores / temperature)

        # Total score: positive + sum of negatives
        ttl_score = pos_score + neg_scores.sum(dim=1)

        # InfoNCE loss
        cl_loss = -torch.log(pos_score / ttl_score)

        return torch.mean(cl_loss)

    def compute_loss_and_step(
        self,
        start_embeds: torch.Tensor,
        target_embeds: torch.Tensor,
        conditions: List[torch.Tensor],
        user_prior: Optional[torch.Tensor] = None,
        epoch: Optional[int] = None,
        batch_users: Optional[torch.Tensor] = None,
        batch_pos_items: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """
        Compute RF loss and contrastive loss, then optimize.
                Changes:
        1. Always compute cl_loss here (removed cl_loss_in_main logic)
        2. Use _infonce_loss_interaction_based with batch indices
        3. Support 2-RF via automatic reflow dataset preparation

        Args:
            target_embeds: Target embedding (RF learning target), shape (batch, embedding_dim)
            conditions: List of condition embeddings (will be concatenated)
            user_prior: Optional user prior guidance, shape (batch, embedding_dim)
            epoch: Current epoch (for warmup control)
            batch_users: Batch user indices for interaction-based InfoNCE, shape (batch_size,)
            batch_pos_items: Batch positive item indices for interaction-based InfoNCE, shape (batch_size,)

        Returns:
            loss_dict: {"rf_loss": float, "cl_loss": float, "total_loss": float, "is_2rf": bool}
        """
        # Update epoch if provided
        if epoch is not None:
            self.current_epoch = epoch

        # Concatenate conditions
        conditions_cat = torch.cat(conditions, dim=-1)

        # Initialize velocity network if needed
        self._init_velocity_net(conditions_cat.shape[-1], target_embeds.device)

        # Ensure velocity_net is in training mode
        self.velocity_net.train()

        # === 2-RF: Prepare reflow dataset if transitioning ===
        fixed_noise = None
        rf_target = target_embeds.detach() # Target 不传梯度

        if self.is_2rf_active:
            # Generate paired (Z0, Z1) dataset using current 1-RF model
            # 每个 epoch 都更新，确保使用最新的模型输出
            if self.reflow_z0 is None or True:  # Always update
                # 显式使用 no_grad 和 inference_only=True，防止梯度泄露
                with torch.no_grad():
                    self.reflow_z0, self.reflow_z1, self.reflow_conditions = \
                        self.prepare_reflow_dataset(conditions, target_embeds.device)

            # 混合策略：2-RF 初期（前 5 个 epoch）逐渐从原始目标过渡到 reflow 目标
            epochs_since_2rf_start = self.current_epoch - self.rf_2rf_transition_epoch
            if epochs_since_2rf_start < 5:
                # 逐渐增加 reflow 权重：0.2 -> 0.4 -> 0.6 -> 0.8 -> 1.0
                reflow_weight = 0.2 * (epochs_since_2rf_start + 1)
                rf_target = reflow_weight * self.reflow_z1 + (1 - reflow_weight) * target_embeds.detach()
                print(f"  [2-RF Warmup] epoch={self.current_epoch}, reflow_weight={reflow_weight:.2f}")
            else:
                rf_target = self.reflow_z1

            # Use 1-RF output (Z1) as new target, fixed noise (Z0) as starting point
            fixed_noise = self.reflow_z0

        # === 核心修改：手动计算 Loss 以支持单步预测 ===
        
        batch_size = rf_target.shape[0]
        X1 = rf_target
        X0 = fixed_noise if fixed_noise is not None else torch.randn_like(X1)
        # X0 = start_embeds
        # Sample time step t
        t = torch.rand(batch_size, 1).to(X1.device)
        
        # Interpolate state X_t
        X_t = t * X1 + (1 - t) * X0
        
        # Predict velocity v_pred
        # 这里的 v_pred 是带有梯度的
        v_pred = self.velocity_net(X_t, t, conditions_cat.detach(), 
                                   user_prior=user_prior.detach() if user_prior is not None else None, 
                                   x_1=X1)

        # 1. RF Loss (MSE)
        v_target = X1 - X0
        rf_loss = F.mse_loss(v_pred, v_target)

        # 2. Contrastive Loss (基于单步预测终点)
        # 几何推断：预测的终点 = 当前位置 + 剩余时间 * 预测速度
        pred_x1 = X_t + (1 - t) * v_pred

        # Split predicted embeddings into users and items
        pred_users, pred_items = torch.split(pred_x1, [self.n_users, self.n_items], dim=0)
        target_users, target_items = torch.split(rf_target, [self.n_users, self.n_items], dim=0)

        # Compute CL loss directly on the PREDICTED endpoint
        cl_loss = self._infonce_loss_interaction_based(
            pred_items, target_items, batch_pos_items, self.contrast_temp
        ) + self._infonce_loss_interaction_based(
            pred_users, target_users, batch_users, self.contrast_temp
        )

        # === Total Loss and Optimization ===
        total_loss = rf_loss + self.contrast_weight * cl_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return {
            "rf_loss": rf_loss.item(),
            "cl_loss": cl_loss.item(),
            "total_loss": total_loss.item(),
            "is_2rf": self.is_2rf_active,
        }

    def generate(
        self,
        conditions: List[torch.Tensor],
        n_steps: Optional[int] = None,
        start_noise: Optional[torch.Tensor] = None,
        use_checkpointing: Optional[bool] = None,
        inference_only: bool = False,
        return_trajectory: bool = False,
        ) -> torch.Tensor:
        """
        Generate embeddings using ODE sampling with optional gradient checkpointing.

        Args:
            conditions: List of condition embeddings (will be concatenated)
            n_steps: ODE sampling steps (None to use default value)
            start_noise: Custom starting noise. If None, use random Gaussian noise.
            use_checkpointing: Enable gradient checkpointing. If None, uses self.use_gradient_checkpointing.
            inference_only: If True, disables gradient tracking for inference.
            return_trajectory: If True, return (z_final, trajectory) with trajectory (T+1, B, D).
        Returns:
            generated_embeds, or (generated_embeds, trajectory) if return_trajectory.
        """
        # Concatenate conditions
        conditions_cat = torch.cat(conditions, dim=-1)

        # Get batch size and device
        batch_size = conditions_cat.shape[0]
        device = conditions_cat.device

        # Initialize velocity network if needed
        self._init_velocity_net(conditions_cat.shape[-1], device)

        # Use default sampling steps if not provided
        if n_steps is None:
            n_steps = self.sampling_steps

        # Use default checkpointing setting if not provided
        if use_checkpointing is None:
            use_checkpointing = self.use_gradient_checkpointing

        # 只有当：网络是 training 模式 AND 外部没有 no_grad AND 没开 inference_only 时，才算训练
        should_train = self.velocity_net.training and torch.is_grad_enabled() and not inference_only

        # 临时借用 eval 模式 (关闭 Dropout/BN 更新)
        previous_training_mode = self.velocity_net.training
        self.velocity_net.eval()

        if start_noise is not None:
            z_0 = start_noise
            # 只有需要反向传播时才加 requires_grad
            if should_train and not z_0.requires_grad:
                z_0 = z_0.detach().requires_grad_(True)
        else:
            z_0 = torch.randn(batch_size, self.embedding_dim, device=device)
            if should_train:
                z_0.requires_grad_(True)

        # Solve ODE
        z_t = z_0
        dt = 1.0 / n_steps
        trajectory_list = [z_t.detach().clone()] if return_trajectory else None

        # 使用计算出的 should_train 状态控制梯度
        with torch.set_grad_enabled(should_train):
            for i in range(n_steps):
                t = torch.full((batch_size, 1), i * dt).to(device)

                if should_train and use_checkpointing:
                    v = torch.utils.checkpoint.checkpoint(
                        self.velocity_net,
                        z_t, t, conditions_cat,
                        use_reentrant=True
                    )
                else:
                    v = self.velocity_net(z_t, t, conditions_cat)

                z_t = z_t + v * dt
                if return_trajectory:
                    trajectory_list.append(z_t.detach().clone())

        # 还原模式
        self.velocity_net.train(previous_training_mode)

        if return_trajectory:
            trajectory = torch.stack(trajectory_list, dim=0)  # (T+1, B, D)
            return z_t, trajectory
        return z_t

    def prepare_reflow_dataset(
        self,
        conditions: List[torch.Tensor],
        device: torch.device,
        n_steps: Optional[int] = None,
    ) -> tuple:
        """
        Generate paired dataset (Z0, Z1, conditions) for 2-RF (Reflow) training.

        This method uses the current trained 1-RF model to generate outputs,
        creating the strict coupling (Z0, Z1) required for training straighter flows.

        Args:
            conditions: List of condition embeddings (will be concatenated)
            device: Device to generate tensors on
            n_steps: ODE sampling steps (None to use default value)

        Returns:
            Tuple of (z0, z1, conditions_cat):
                - z0: Random noise used as starting point, shape (batch, embedding_dim)
                - z1: 1-RF generated output from z0, shape (batch, embedding_dim)
                - conditions_cat: Concatenated conditions, shape (batch, condition_dim)

        Usage Example:
            ```python
            # === Step 1: Train 1-RF ===
            rf_gen = RFEmbeddingGenerator(embedding_dim=64, ...)
            for epoch in range(num_epochs_1rf):
                loss = rf_gen.compute_loss_and_step(
                    target_embeds=target,
                    conditions=[image_cond, text_cond],
                )

            # === Step 2: Prepare Reflow Dataset ===
            z0, z1, cond = rf_gen.prepare_reflow_dataset(
                [image_cond, text_cond], device
            )

            # === Step 3: Train 2-RF ===
            for epoch in range(num_epochs_2rf):
                loss = rf_gen.compute_loss_and_step(
                    target_embeds=z1,       # 1-RF output as target
                    conditions=[image_cond, text_cond],
                    fixed_noise=z0,         # Paired noise as fixed starting point
                )

            # === Step 4: Fast inference (can use fewer steps) ===
            output = rf_gen.generate([image_cond, text_cond], n_steps=1)
            ```
        """
        # Concatenate conditions
        conditions_cat = torch.cat(conditions, dim=-1)
        batch_size = conditions_cat.shape[0]

        with torch.no_grad():
            z0 = torch.randn(batch_size, self.embedding_dim).to(device)
            # 这里调用 generate 时强制为纯推理模式
            z1 = self.generate(conditions, n_steps=n_steps, start_noise=z0, inference_only=True)

        return z0, z1, conditions_cat

    def mix_embeddings(
        self,
        original_embeds: torch.Tensor,
        generated_embeds: torch.Tensor,
        training: bool = True,
        epoch: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Mix original and generated embeddings.

        Automatically decides mixing strategy based on warmup status and mix_ratio.
        
        **Training mode**: Always returns original embeddings (no mixing).
        **Inference mode**: Mixes embeddings according to inference_mix_ratio.

        Args:
            original_embeds: Original embedding
            generated_embeds: RF-generated embedding
            training: Training/inference mode
            epoch: Current epoch (for warmup control)

        Returns:
            mixed_embeds: Mixed embeddings
        """
        # Update epoch if provided
        if epoch is not None:
            self.current_epoch = epoch

        # During warmup: use pure original embeddings
        if self.current_epoch < self.warmup_epochs:
            return original_embeds

        # Training mode: NO mixing, always use original embeddings
        if training:
            return original_embeds
        
        # Inference mode: apply mixing with inference_mix_ratio
        mixed_embeds = original_embeds + self.inference_mix_ratio * generated_embeds

        return mixed_embeds
