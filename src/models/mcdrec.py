# coding: utf-8
# Desc: Core code of the LD4MRec.
# Author: OrangeAI Research Team
# Time: 2026-01-04
# paper: "Multimodal Conditioned Diffusion Model for Recommendation" (WWW2024, MCDRec)
# Ref Link: https://dl.acm.org/doi/pdf/10.1145/3589335.3651956

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
import math
from common.abstract_recommender import GeneralRecommender

class UNetConditional(nn.Module):
    def __init__(self, embedding_size):
        super(UNetConditional, self).__init__()
        self.embedding_size = embedding_size
        self.height = int(math.sqrt(embedding_size))
        assert self.height * self.height == embedding_size, "Embedding size must be a perfect square for U-Net reshaping"
        
        # 3 Channels: Item (Noisy), Visual, Textual
        self.in_channels = 3 
        self.out_channels = 1 # Output Item embedding
        
        # Encoder
        self.enc1 = self.conv_block(self.in_channels, 16)
        self.enc2 = self.conv_block(16, 32)
        
        # Decoder
        self.dec1 = self.conv_block(32 + 16, 16) # Skip connection
        self.final = nn.Conv2d(16, self.out_channels, kernel_size=1)
        
        # Time Embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(embedding_size, embedding_size),
            nn.SiLU(),
            nn.Linear(embedding_size, embedding_size)
        )

    def conv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.SiLU(),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.SiLU()
        )

    def forward(self, x, t, h_v, h_t):
        # x: [B, D]
        # t: [B]
        # h_v: [B, D]
        # h_t: [B, D]
        
        B, D = x.shape
        H = self.height
        
        # Reshape to images [B, C, H, W]
        x_img = x.view(B, 1, H, H)
        v_img = h_v.view(B, 1, H, H)
        t_img = h_t.view(B, 1, H, H)
        
        input_img = torch.cat([x_img, v_img, t_img], dim=1) # [B, 3, H, H]
        
        # Time embedding
        t_emb = timestep_embedding(t, D).to(x.device)
        t_emb = self.time_mlp(t_emb).view(B, 1, H, H)
        
        # Inject time?
        # Simple way: add to input or feature maps
        # The paper says "add the step representation t_i ... to each convolutional neural network block"
        # For simplicity, we add it to the input here, or broadcast add.
        
        # Encoder
        e1 = self.enc1(input_img + t_emb)
        e2 = self.enc2(e1)
        
        # Decoder
        d1 = self.dec1(torch.cat([e2, e1], dim=1))
        out = self.final(d1)
        
        return out.view(B, D)

def timestep_embedding(timesteps, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

class GaussianDiffusion(nn.Module):
    def __init__(self, config, device):
        super(GaussianDiffusion, self).__init__()
        self.steps = config['steps']
        self.noise_scale = config['noise_scale']
        self.noise_min = config['noise_min']
        self.noise_max = config['noise_max']
        self.device = device
        
        self.betas = torch.tensor(self.get_betas(), dtype=torch.float32).to(device)
        self.calculate_for_diffusion()
        
    def get_betas(self):
        start = self.noise_scale * self.noise_min
        end = self.noise_scale * self.noise_max
        return np.linspace(start, end, self.steps, dtype=np.float64)

    def calculate_for_diffusion(self):
        alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(alphas, axis=0).to(self.device)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]).to(self.device), self.alphas_cumprod[:-1]]).to(self.device)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)
        
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1.0 - self.alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - self.alphas_cumprod)

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        return (
            self._extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def _extract(self, a, t, x_shape):
        batch_size = t.shape[0]
        out = a.gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

    def p_sample(self, model, x, t, h_v, h_t):
        model_out = model(x, t, h_v, h_t)
        # Assume x0 prediction
        pred_x0 = model_out
        
        # Posterior mean
        model_mean = (
            self._extract(self.posterior_mean_coef1, t, x.shape) * pred_x0 +
            self._extract(self.posterior_mean_coef2, t, x.shape) * x
        )
        model_variance = self._extract(self.posterior_variance, t, x.shape)
        
        noise = torch.randn_like(x)
        # No noise when t == 0
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        
        return model_mean + nonzero_mask * torch.sqrt(model_variance) * noise

class MCDRec(GeneralRecommender):
    def __init__(self, config, dataset):
        super(MCDRec, self).__init__(config, dataset)
        self.config = config
        
        # Params
        self.latent_dim = config['embedding_size']
        self.n_layers = config['lightGCN_n_layers']
        self.lambda_dm = config['lambda_dm']
        self.tau = config['tau']
        self.rho = config['rho']
        self.omega = 0.1 # Weight for fusing diffused item (Pre-defined in paper as diffused weight)
        
        # Graph
        if hasattr(dataset, 'inter_matrix'):
            self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        elif hasattr(dataset, 'dataset') and hasattr(dataset.dataset, 'inter_matrix'):
            self.interaction_matrix = dataset.dataset.inter_matrix(form='coo').astype(np.float32)
        else:
            raise ValueError("Dataset does not have inter_matrix method")
            
        self.norm_adj = self.get_norm_adj_mat(self.interaction_matrix)
        self.Graph = self.norm_adj.to(self.device)
        
        # Embeddings
        self.embedding_user = nn.Embedding(self.n_users, self.latent_dim)
        self.embedding_item = nn.Embedding(self.n_items, self.latent_dim)
        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_item.weight, std=0.1)
        
        # Multimodal
        self.v_mlp = nn.Linear(self.v_feat.shape[1], self.latent_dim) if self.v_feat is not None else None
        self.t_mlp = nn.Linear(self.t_feat.shape[1], self.latent_dim) if self.t_feat is not None else None
        
        # Diffusion
        self.diffusion = GaussianDiffusion(config, self.device)
        self.unet = UNetConditional(self.latent_dim)
        
        # Storage for diffused items (for DGD)
        self.diffused_items = None

    def get_norm_adj_mat(self, interaction_matrix):
        R = interaction_matrix
        n_users = self.n_users
        n_items = self.n_items
        row = np.concatenate([R.row, R.col + n_users])
        col = np.concatenate([R.col + n_users, R.row])
        data = np.ones_like(row)
        A = sp.coo_matrix((data, (row, col)), shape=(n_users + n_items, n_users + n_items))
        rowsum = np.array(A.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        norm_adj = d_mat_inv_sqrt.dot(A).dot(d_mat_inv_sqrt)
        coo = norm_adj.tocoo()
        indices = torch.from_numpy(np.vstack((coo.row, coo.col)).astype(np.int64))
        values = torch.from_numpy(coo.data.astype(np.float32))
        return torch.sparse.FloatTensor(indices, values, torch.Size(coo.shape))

    def pre_epoch_processing(self):
        # DGD: Diffusion-guided Graph Denoising
        # 1. Generate diffused item representations (inference mode for all items)
        # This is expensive, so we might do it only every N epochs or simplify.
        # Paper says "at the beginning of each training epoch".
        
        self.eval()
        with torch.no_grad():
            # Get Multimodal features
            if self.v_feat is not None:
                h_v = self.v_mlp(self.v_feat.to(self.device))
            else:
                h_v = torch.zeros(self.n_items, self.latent_dim).to(self.device)
                
            if self.t_feat is not None:
                h_t = self.t_mlp(self.t_feat.to(self.device))
            else:
                h_t = torch.zeros(self.n_items, self.latent_dim).to(self.device)
            
            # Start from noise
            x = torch.randn(self.n_items, self.latent_dim).to(self.device)
            
            # Reverse process
            indices = list(range(self.diffusion.steps))[::-1]
            for i in indices:
                t = torch.tensor([i] * self.n_items).to(self.device)
                x = self.diffusion.p_sample(self.unet, x, t, h_v, h_t)
            
            # Final item representation
            # \tilde{e} = x_0 + \omega * x_p (x_p is the diffused/predicted one)
            # x_0 is the current item embedding?
            # Paper: "we denote \tilde{e} = x_0 + \omega * x_p"
            # Here x is x_p
            self.diffused_items = self.embedding_item.weight + self.omega * x
            
            # 2. Graph Denoising
            # Compute scores S_ui = e_u^T \tilde{e}_i
            # This requires O(U*I) computation, which is too huge for large datasets.
            # Paper: "For each edge e_ui in G, we can update its weight... Finally sample a denoised sub-graph"
            # So we only compute scores for EXISTING edges.
            
            users = self.interaction_matrix.row
            items = self.interaction_matrix.col
            
            u_emb = self.embedding_user(torch.tensor(users).to(self.device))
            i_diff = self.diffused_items[torch.tensor(items).to(self.device)]
            
            scores = (u_emb * i_diff).sum(dim=1) # Dot product
            
            # Probability P_ui = 1 / sqrt(d_u d_i). 
            # Wait, paper says: "update its weight as (1 + \tau * s_ui)... sample... with probability P_ui"
            # Actually, standard LightGCN uses 1/sqrt(du di).
            # DGD: "identifies authentic noise edges and smoothly prunes them"
            # "sample a denoised sub-graph... with the probability P_ui = 1 / sqrt(d_u d_i) of each edge... until number of edges reach |E|(1-rho)"
            # This sounds like just DropEdge but guided?
            # Re-reading: "sample a denoised sub-graph G_S from G with the probability P_ui ... until ... |E|(1-rho)"
            # The probability formula in paper is just the normalization term.
            # It seems they use the UPDATED WEIGHTS to guide sampling?
            # "For each edge... update its weight... then re-calculate degrees... Finally sample"
            # If P_ui is just 1/sqrt(du di), it doesn't involve the score S_ui.
            # Maybe the text implies P_ui depends on the updated weight?
            # Let's assume Probability is proportional to updated weight (1 + tau * s_ui).
            
            edge_weights = 1.0 + self.tau * scores
            edge_weights = F.relu(edge_weights) + 1e-8 # Ensure non-negative and non-zero
            
            # Normalize to probability
            probs = edge_weights / edge_weights.sum()
            
            # Sample edges
            n_edges = len(users)
            n_keep = int(n_edges * (1 - self.rho))
            
            # Ensure n_keep > 0
            if n_keep <= 0:
                 n_keep = 1
            
            # Fix potential nan/inf in probs
            if torch.isnan(probs).any() or torch.isinf(probs).any() or (probs < 0).any():
                probs = torch.ones_like(probs) / len(probs)
            
            indices = torch.multinomial(probs, n_keep, replacement=False).cpu().numpy()
            
            # Construct new graph
            new_users = users[indices]
            new_items = items[indices]
            
            # Build new Adjacency
            # We need to construct symmetric adj from this subset
            # Re-use get_norm_adj_mat logic but with subset
            
            data = np.ones_like(new_users)
            new_R_coo = sp.coo_matrix((data, (new_users, new_items)), shape=(self.n_users, self.n_items))
            
            self.Graph = self.get_norm_adj_mat(new_R_coo).to(self.device)
            
        self.train()

    def forward(self):
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        embs = [all_emb]
        
        for layer in range(self.n_layers):
            all_emb = torch.sparse.mm(self.Graph, all_emb)
            embs.append(all_emb)
            
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.n_users, self.n_items])
        return users, items

    def calculate_loss(self, interaction):
        user = interaction[0]
        pos_item = interaction[1]
        neg_item = interaction[2]
        
        # 1. BPR Loss
        all_users, all_items = self.forward()
        u_e = all_users[user]
        pos_e = all_items[pos_item]
        neg_e = all_items[neg_item]
        
        pos_scores = torch.mul(u_e, pos_e).sum(dim=1)
        neg_scores = torch.mul(u_e, neg_e).sum(dim=1)
        
        bpr_loss = -torch.log(torch.sigmoid(pos_scores - neg_scores)).mean()
        
        # 2. Diffusion Loss
        # Sample items for diffusion training (can use pos_items from batch)
        # Target: Item Embedding (x0)
        target_items = pos_item
        x_0 = self.embedding_item(target_items) # Raw embedding as target
        
        # Multimodal Condition
        if self.v_feat is not None:
            h_v = self.v_mlp(self.v_feat[target_items].to(self.device))
        else:
            h_v = torch.zeros(len(target_items), self.latent_dim).to(self.device)
            
        if self.t_feat is not None:
            h_t = self.t_mlp(self.t_feat[target_items].to(self.device))
        else:
            h_t = torch.zeros(len(target_items), self.latent_dim).to(self.device)
            
        # Noise
        t = torch.randint(0, self.diffusion.steps, (len(target_items),)).to(self.device)
        noise = torch.randn_like(x_0)
        x_t = self.diffusion.q_sample(x_0, t, noise)
        
        # Predict
        pred_x0 = self.unet(x_t, t, h_v, h_t)
        
        dm_loss = F.mse_loss(pred_x0, x_0)
        
        return bpr_loss + self.lambda_dm * dm_loss

    def full_sort_predict(self, interaction):
        user = interaction[0]
        all_users, all_items = self.forward()
        u_e = all_users[user]
        return torch.matmul(u_e, all_items.transpose(0, 1))
