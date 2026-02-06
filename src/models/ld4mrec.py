# coding: utf-8
# Desc: Core code of the LD4MRec.
# Author: OrangeAI Research Team
# Time: 2026-01-04
# paper: "LD4MRec: Simplifying and Powering Diffusion Model for Multimedia Recommendation" (WWW2024, LD4MRec)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import svds
from common.abstract_recommender import GeneralRecommender

class CNet(nn.Module):
    def __init__(self, n_items, hidden_size, cond_dim, n_layers=3, dropout=0.1):
        super(CNet, self).__init__()
        self.n_items = n_items
        self.hidden_size = hidden_size
        
        # Input Projection
        self.item_proj = nn.Linear(n_items, hidden_size)
        
        # Condition Projection (User SVD + MM)
        self.cond_proj = nn.Linear(cond_dim, hidden_size)
        
        # Time Embedding Projection
        self.time_proj = nn.Linear(hidden_size, hidden_size)
        
        # Blocks (Cascaded BD/BI Blocks simplified as Conditional ResBlocks)
        self.layers = nn.ModuleList([
            ConditionalBlock(hidden_size, dropout) for _ in range(n_layers)
        ])
        
        # Output Projection
        self.output_proj = nn.Linear(hidden_size, n_items)
        
    def forward(self, x_t, t_emb, condition):
        # x_t: [B, n_items]
        # t_emb: [B, hidden_size]
        # condition: [B, cond_dim]
        
        h = self.item_proj(x_t)
        cond_h = self.cond_proj(condition)
        time_h = self.time_proj(t_emb)
        
        # Combine condition and time
        # Using a simple addition or modulation
        global_cond = cond_h + time_h
        
        for layer in self.layers:
            h = layer(h, global_cond)
            
        out = self.output_proj(h)
        return out

class ConditionalBlock(nn.Module):
    def __init__(self, hidden_size, dropout):
        super(ConditionalBlock, self).__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        
        # FiLM-like modulation
        self.cond_scale = nn.Linear(hidden_size, hidden_size)
        self.cond_shift = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, x, cond):
        # x: [B, H]
        # cond: [B, H]
        
        residual = x
        x = self.norm1(x)
        
        # Modulation
        scale = self.cond_scale(cond)
        shift = self.cond_shift(cond)
        x = x * (1 + scale) + shift
        
        x = self.linear1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.linear2(x)
        
        return residual + x

class LD4MRec(GeneralRecommender):
    def __init__(self, config, dataset):
        super(LD4MRec, self).__init__(config, dataset)
        self.config = config
        
        # Parameters
        self.embedding_size = config['embedding_size']
        self.steps = config['steps']
        self.noise_schedule = config['noise_schedule']
        self.noise_min = config['noise_min']
        self.noise_max = config['noise_max']
        self.svd_k = config['svd_k']
        self.smoothing_gamma = config['smoothing_gamma']
        self.cnet_hidden = config['cnet_hidden_size']
        self.cnet_layers = config['cnet_n_layers']
        
        # Data
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        
        # SVD Encoder
        self._init_svd()
        
        # Multimodal Encoder
        self._init_multimodal(dataset)
        
        # Model
        # Condition dim = SVD_dim + MM_dim
        # MM_dim depends on dataset features.
        self.mm_dim = 0
        if self.v_feat is not None: self.mm_dim += self.v_feat.shape[1]
        if self.t_feat is not None: self.mm_dim += self.t_feat.shape[1]
        
        # Project MM dim to embedding size if too large? 
        # Paper says "weighted aggregating... e_u,m". e_u,m has same dim as item feats.
        # We'll project it to keep things manageable.
        self.mm_project = nn.Linear(self.mm_dim, self.embedding_size) if self.mm_dim > 0 else None
        
        cond_dim = self.svd_k + (self.embedding_size if self.mm_dim > 0 else 0)
        
        self.cnet = CNet(self.n_items, self.cnet_hidden, cond_dim, self.cnet_layers, config['dropout'])
        
        # Time Embedding
        self.time_emb_dim = self.cnet_hidden
        self.register_buffer('loss_history', torch.ones(self.steps)) # For importance sampling
        
        # Noise Schedule
        self._init_noise_schedule()
        
        # Learnable t_in
        self.t_in = nn.Parameter(torch.zeros(1)) # Initialized near 0
        
    def _init_svd(self):
        # Perform SVD on interaction matrix
        # U, S, Vt = svds(R, k)
        # User embedding: U * sqrt(S) or just U?
        # Paper: "employ SVD scheme as encoder". Typically U for users.
        try:
            # Use scipy sparse svd
            # Convert to float for svds
            R = self.interaction_matrix
            u, s, vt = svds(R, k=self.svd_k)
            # Sort singular values (svds returns them in increasing order)
            # We want largest.
            u = u[:, ::-1]
            s = s[::-1]
            # User embeddings: U * S^0.5
            self.user_svd_emb = torch.from_numpy(u * np.sqrt(s)).float().to(self.device)
        except Exception as e:
            print(f"SVD failed: {e}. Using random embeddings.")
            self.user_svd_emb = torch.randn(self.n_users, self.svd_k).to(self.device)
            
    def _init_multimodal(self, dataset):
        # Pre-compute user multimodal preference
        # e_{u,m} = sum_{i in N_u} 1/sqrt(|N_u||N_i|) * e_{i,m}
        # This is essentially one layer of LightGCN aggregation on item features.
        
        self.user_mm_emb = None
        
        feats = []
        if self.v_feat is not None: feats.append(self.v_feat)
        if self.t_feat is not None: feats.append(self.t_feat)
        
        if len(feats) > 0:
            item_feats = torch.cat(feats, dim=1).to(self.device)
            
            # Build normalized adjacency matrix for aggregation
            # We only need User <- Item aggregation.
            # A_hat = D^{-1/2} A D^{-1/2}
            # R is (n_users, n_items).
            # We need R_norm = D_u^{-1/2} R D_i^{-1/2}
            
            # Interaction matrix R
            R = self.interaction_matrix
            
            # Degrees
            row_sum = np.array(R.sum(1)).flatten()
            col_sum = np.array(R.sum(0)).flatten()
            
            d_u_inv = np.power(row_sum, -0.5)
            d_u_inv[np.isinf(d_u_inv)] = 0.
            
            d_i_inv = np.power(col_sum, -0.5)
            d_i_inv[np.isinf(d_i_inv)] = 0.
            
            d_u_mat = sp.diags(d_u_inv)
            d_i_mat = sp.diags(d_i_inv)
            
            R_norm = d_u_mat.dot(R).dot(d_i_mat)
            R_norm = R_norm.tocoo()
            
            indices = torch.from_numpy(np.vstack((R_norm.row, R_norm.col)).astype(np.int64))
            values = torch.from_numpy(R_norm.data).float()
            shape = torch.Size(R_norm.shape)
            
            R_tensor = torch.sparse.FloatTensor(indices, values, shape).to(self.device)
            
            # Aggregate
            self.user_mm_emb = torch.sparse.mm(R_tensor, item_feats)
        
    def _init_noise_schedule(self):
        # Linear schedule for 1 - alpha_bar
        # 1 - alpha_bar_t = s * (alpha_min + (t-1)/(T-1) * (1 - alpha_min))
        # Wait, paper Eq 3: 1 - \bar{alpha}_t = ...
        # Let beta_t = 1 - alpha_t. \bar{alpha}_t = prod(alpha_i).
        
        # We use standard linear beta schedule or the specific one from paper.
        # Paper Eq 3:
        # 1 - \bar{alpha}_t = s * [ alpha_min + (t-1)/(T-1) * (1 - alpha_min) ]
        # Here s is scale (usually 1?). Paper says s \in [0, 1].
        # alpha_min \in (0, 1).
        
        s = 1.0 # Default scale
        alpha_min = self.config['min_noise_level'] # e.g. 0.001
        
        t = torch.arange(1, self.steps + 1, dtype=torch.float32).to(self.device)
        one_minus_alpha_bar = s * (alpha_min + (t - 1) / (self.steps - 1) * (1 - alpha_min))
        self.alpha_bar = 1 - one_minus_alpha_bar
        
        # Calculate betas from alpha_bar
        # alpha_bar_t = alpha_t * alpha_bar_{t-1}
        # alpha_t = alpha_bar_t / alpha_bar_{t-1}
        # beta_t = 1 - alpha_t
        
        alpha_bar_prev = torch.cat([torch.tensor([1.0]).to(self.device), self.alpha_bar[:-1]])
        self.alphas = self.alpha_bar / alpha_bar_prev
        self.betas = 1 - self.alphas
        
        # Clamp betas to avoid numerical issues
        self.betas = torch.clamp(self.betas, min=0.0001, max=0.9999)
        self.alphas = 1 - self.betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)
        
    def get_time_embedding(self, timesteps):
        # Sinusoidal embedding
        # timesteps: [B]
        half_dim = self.time_emb_dim // 2
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=self.device) * -np.log(10000.0) / (half_dim - 1))
        emb = timesteps[:, None].float() * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if self.time_emb_dim % 2 == 1:
            emb = torch.nn.functional.pad(emb, (0, 1))
        return emb

    def q_sample(self, x_start, t):
        # x_start: [B, N]
        # t: [B]
        
        # q(x_t | x_0) = N(sqrt(alpha_bar_t) * x_0, (1 - alpha_bar_t) * I)
        # Note: Paper defines x_in as x_start.
        
        noise = torch.randn_like(x_start)
        
        alpha_bar_t = self.alpha_bar[t].unsqueeze(1) # [B, 1]
        
        return torch.sqrt(alpha_bar_t) * x_start + torch.sqrt(1 - alpha_bar_t) * noise, noise

    def calculate_loss(self, interaction):
        user = interaction[0]
        
        # Construct x_in (multi-hot)
        # We need efficient way to get multi-hot vectors for batch users
        # Usually datasets in GenMMRec/RecBole don't provide this directly in interaction.
        # interaction provides [user, item, ...] pairs.
        # We need to aggregate history.
        # But `train_data` yields interactions.
        # For diffusion, we usually need the full history vector per user in the batch.
        # We can construct it on the fly or pre-process.
        # Given `user` batch, we can fetch their history from `dataset`.
        
        # Accessing dataset from model is possible via self.dataset (if we saved it, which I didn't in init)
        # Actually GeneralRecommender doesn't save dataset object usually? 
        # Wait, I called `super(LD4MRec, self).__init__(config, dataset)`.
        # GeneralRecommender doesn't save `dataset` as `self.dataset`.
        # I need to save it or access `self.interaction_matrix`.
        # self.interaction_matrix is sparse COO.
        
        # Get dense history vectors for batch users
        # This might be slow if done naively.
        # Fast way: Slicing CSR matrix.
        # Convert interaction_matrix to CSR once in init.
        
        if not hasattr(self, 'interaction_csr'):
            self.interaction_csr = sp.csr_matrix(self.interaction_matrix)
            
        # Get rows for batch users
        # user is Tensor on device. Move to cpu numpy.
        batch_users_np = user.cpu().numpy()
        batch_vectors = self.interaction_csr[batch_users_np].toarray()
        x_in = torch.from_numpy(batch_vectors).float().to(self.device)
        
        # Label Smoothing for x_0 target
        # f_S(x)
        gamma = self.smoothing_gamma
        x_0_target = x_in * (1 - gamma) + (1 - x_in) * gamma
        
        # Importance Sampling for t
        # p_t \propto sqrt(E[L^2])
        # We use stored history.
        loss_history_np = self.loss_history.cpu().numpy()
        probs = np.sqrt(loss_history_np ** 2)
        probs /= probs.sum()
        
        # Sample t
        t_indices = np.random.choice(self.steps, size=len(user), p=probs)
        t = torch.from_numpy(t_indices).long().to(self.device)
        
        # Forward Diffusion
        x_t, noise = self.q_sample(x_in, t)
        
        # Condition
        u_svd = self.user_svd_emb[user]
        u_mm = self.user_mm_emb[user] if self.user_mm_emb is not None else None
        
        if self.mm_project is not None and u_mm is not None:
            u_mm = self.mm_project(u_mm)
            
        condition = torch.cat([u_svd, u_mm], dim=1) if u_mm is not None else u_svd
        
        # Time Emb
        t_emb = self.get_time_embedding(t)
        
        # Predict x_0
        pred_x0 = self.cnet(x_t, t_emb, condition)
        
        # Loss
        # MSE between pred_x0 and x_0_target
        loss = F.mse_loss(pred_x0, x_0_target, reduction='none').mean(dim=1)
        
        # Update history
        # We need to scatter reduce or loop.
        # Since t is random, we can just update moving average.
        with torch.no_grad():
            for i, time_step in enumerate(t_indices):
                self.loss_history[time_step] = 0.9 * self.loss_history[time_step] + 0.1 * loss[i].item()
                
        return loss.mean()

    def full_sort_predict(self, interaction):
        user = interaction[0]
        
        # Inference
        # Start from x_in (history)
        # But wait, predict is for evaluation. interaction[0] are users.
        # We need their history x_in.
        
        if not hasattr(self, 'interaction_csr'):
            self.interaction_csr = sp.csr_matrix(self.interaction_matrix)
            
        batch_users_np = user.cpu().numpy()
        batch_vectors = self.interaction_csr[batch_users_np].toarray()
        x_in = torch.from_numpy(batch_vectors).float().to(self.device)
        
        # Time condition: t_in
        # Paper says t_in is learnable.
        # We expand t_in to batch size.
        # t_in is a float? Or an embedding?
        # Paper says "learnable input time representation t_in".
        # It's likely a vector or a scalar time step.
        # "assume that t_in approximates t_0".
        # If t is discrete 1..T, t_in should probably be mapped to embedding.
        # If I defined t_in as parameter(1), I can treat it as a continuous time and use same embedding function?
        # Or just use t=0?
        # Let's use t=0 for simplicity or use the learned parameter passed to time_emb.
        
        # Actually, t in diffusion is usually discrete index.
        # But get_time_embedding supports float input.
        # Let's use t_in (clamped to range?).
        
        t_val = torch.abs(self.t_in).expand(len(user)) # [B]
        t_emb = self.get_time_embedding(t_val)
        
        # Condition
        u_svd = self.user_svd_emb[user]
        u_mm = self.user_mm_emb[user] if self.user_mm_emb is not None else None
        if self.mm_project is not None and u_mm is not None:
            u_mm = self.mm_project(u_mm)
        condition = torch.cat([u_svd, u_mm], dim=1) if u_mm is not None else u_svd
        
        # One-step prediction
        # Input to C-Net is x_in (noisy observation)
        scores = self.cnet(x_in, t_emb, condition)
        
        return scores
