# coding: utf-8
# Desc: Core code of the DDRM.
# Author: OrangeAI Research Team
# Time: 2026-01-06
# paper: "Denoising Diffusion Recommender Model" (SIGIR2024, DDRM)
# Ref Link: https://dl.acm.org/doi/pdf/10.1145/3626772.3657825

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
import math
from common.abstract_recommender import GeneralRecommender

class LightGCN_Encoder(nn.Module):
    def __init__(self, n_users, n_items, config):
        super(LightGCN_Encoder, self).__init__()
        self.config = config
        self.num_users = n_users
        self.num_items = n_items
        self.latent_dim = config['embedding_size']
        self.n_layers = config['lightGCN_n_layers']
        self.keep_prob = config['keep_prob']
        self.A_split = config['A_split']
        
        self.embedding_user = nn.Embedding(self.num_users, self.latent_dim)
        self.embedding_item = nn.Embedding(self.num_items, self.latent_dim)
        
        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_item.weight, std=0.1)
        
        self.Graph = None # Will be set externally

    def __dropout_x(self, x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index]/keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g
    
    def __dropout(self, keep_prob):
        if self.A_split:
            graph = []
            for g in self.Graph:
                graph.append(self.__dropout_x(g, keep_prob))
        else:
            graph = self.__dropout_x(self.Graph, keep_prob)
        return graph
    
    def forward(self):
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        embs = [all_emb]

        if self.config['dropout'] and self.training:
             g_droped = self.__dropout(self.keep_prob)
        else:
             g_droped = self.Graph    
        
        for layer in range(self.n_layers):
            all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items

class DNN(nn.Module):
    """
    A deep neural network for the reverse diffusion process.
    """
    def __init__(self, in_dims, out_dims, emb_size, time_type="cat", norm=False, dropout=0.5, act='tanh'):
        super(DNN, self).__init__()
        self.in_dims = in_dims
        self.out_dims = out_dims
        assert out_dims[0] == in_dims[-1], "In and out dimensions must equal to each other."
        self.time_type = time_type
        self.time_emb_dim = emb_size
        self.norm = norm
        self.act = act

        self.emb_layer = nn.Linear(self.time_emb_dim, self.time_emb_dim)

        if self.time_type == "cat":
            in_dims_temp = [self.in_dims[0]*2 + self.time_emb_dim] + self.in_dims[1:]
        else:
            raise ValueError("Unimplemented timestep embedding type %s" % self.time_type)
        out_dims_temp = self.out_dims
        
        self.in_layers = nn.ModuleList([nn.Linear(d_in, d_out) \
            for d_in, d_out in zip(in_dims_temp[:-1], in_dims_temp[1:])])
        self.out_layers = nn.ModuleList([nn.Linear(d_in, d_out) \
            for d_in, d_out in zip(out_dims_temp[:-1], out_dims_temp[1:])])
        
        self.drop = nn.Dropout(dropout)
        self.init_weights()
    
    def init_weights(self):
        for layer in self.in_layers:
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)
            layer.bias.data.normal_(0.0, 0.001)
        
        for layer in self.out_layers:
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)
            layer.bias.data.normal_(0.0, 0.001)
        
        size = self.emb_layer.weight.size()
        fan_out = size[0]
        fan_in = size[1]
        std = np.sqrt(2.0 / (fan_in + fan_out))
        self.emb_layer.weight.data.normal_(0.0, std)
        self.emb_layer.bias.data.normal_(0.0, 0.001)

    def forward(self, noise_emb, con_emb, timesteps):
        time_emb = timestep_embedding(timesteps, self.time_emb_dim).to(noise_emb.device)
        emb = self.emb_layer(time_emb)
        if self.norm:
            noise_emb = F.normalize(noise_emb)
        noise_emb = self.drop(noise_emb)

        all_emb = torch.cat([noise_emb, emb, con_emb], dim=-1)

        for i, layer in enumerate(self.in_layers):
            all_emb = layer(all_emb)
            if self.act == 'tanh':
                all_emb = torch.tanh(all_emb)
            elif self.act == 'sigmoid':
                all_emb = torch.sigmoid(all_emb)
            elif self.act == 'relu':
                all_emb = F.relu(all_emb)
        for i, layer in enumerate(self.out_layers):
            all_emb = layer(all_emb)
            if i != len(self.out_layers) - 1:
                if self.act == 'tanh':
                    all_emb = torch.tanh(all_emb)
                elif self.act == 'sigmoid':
                    all_emb = torch.sigmoid(all_emb)
                elif self.act == 'relu':
                    all_emb = F.relu(all_emb)
        return all_emb

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

def mean_flat(tensor):
    return tensor.mean(dim=1)

class GaussianDiffusion(nn.Module):
    def __init__(self, noise_scale, noise_min, noise_max, steps, device='cpu', history_num_per_term=10, beta_fixed=True, noise_schedule='linear-var'):
        super(GaussianDiffusion, self).__init__()

        self.noise_scale = noise_scale
        self.noise_min = noise_min
        self.noise_max = noise_max
        self.steps = steps
        self.device = device
        self.noise_schedule = noise_schedule

        self.history_num_per_term = history_num_per_term
        self.Lt_history = torch.zeros(steps, history_num_per_term, dtype=torch.float64).to(device)
        self.Lt_count = torch.zeros(steps, dtype=int).to(device)

        if noise_scale != 0:
            self.betas = torch.tensor(self.get_betas(), dtype=torch.float64).to(device)
            if beta_fixed:
                self.betas[0] = 0.00001
            self.calculate_for_diffusion()

    def get_betas(self):
        if self.noise_schedule == "linear" or self.noise_schedule == "linear-var":
            start = self.noise_scale * self.noise_min
            end = self.noise_scale * self.noise_max
            if self.noise_schedule == "linear":
                return np.linspace(start, end, self.steps, dtype=np.float64)
            else:
                return betas_from_linear_variance(self.steps, np.linspace(start, end, self.steps, dtype=np.float64))
        else:
             start = self.noise_scale * self.noise_min
             end = self.noise_scale * self.noise_max
             return np.linspace(start, end, self.steps, dtype=np.float64)
    
    def calculate_for_diffusion(self):
        alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(alphas, axis=0).to(self.device)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]).to(self.device), self.alphas_cumprod[:-1]]).to(self.device)
        self.alphas_cumprod_next = torch.cat([self.alphas_cumprod[1:], torch.tensor([0.0]).to(self.device)]).to(self.device)

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)

        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = torch.log(torch.cat([self.posterior_variance[1].unsqueeze(0), self.posterior_variance[1:]]))
        self.posterior_mean_coef1 = (self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))
        self.posterior_mean_coef2 = ((1.0 - self.alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - self.alphas_cumprod))

    def _extract_into_tensor(self, arr, timesteps, broadcast_shape):
        arr = arr.to(timesteps.device)
        res = arr[timesteps].float()
        while len(res.shape) < len(broadcast_shape):
            res = res[..., None]
        return res.expand(broadcast_shape)

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        return self._extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start + self._extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise

    def p_mean_variance(self, model, x, con_emb, t):
        model_output = model(x, con_emb, t)

        model_variance = self.posterior_variance
        model_log_variance = self.posterior_log_variance_clipped

        model_variance = self._extract_into_tensor(model_variance, t, x.shape)
        model_log_variance = self._extract_into_tensor(model_log_variance, t, x.shape)

        # Assuming x0 prediction (mean_type='x0')
        pred_xstart = model_output
        
        model_mean, _, _ = self.q_posterior_mean_variance(x_start=pred_xstart, x_t=x, t=t)

        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }
    
    def q_posterior_mean_variance(self, x_start, x_t, t):
        posterior_mean = (
            self._extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + self._extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = self._extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = self._extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def sample_timesteps(self, batch_size, device, method='uniform', uniform_prob=0.001):
        if method == 'importance':
            if not (self.Lt_count == self.history_num_per_term).all():
                return self.sample_timesteps(batch_size, device, method='uniform')
            
            Lt_sqrt = torch.sqrt(torch.mean(self.Lt_history ** 2, axis=-1))
            pt_all = Lt_sqrt / torch.sum(Lt_sqrt)
            pt_all *= 1- uniform_prob
            pt_all += uniform_prob / len(pt_all)
            
            t = torch.multinomial(pt_all, num_samples=batch_size, replacement=True)
            pt = pt_all.gather(dim=0, index=t) * len(pt_all)
            return t, pt
        else:
            t = torch.randint(0, self.steps, (batch_size,), device=device).long()
            pt = torch.ones_like(t).float()
            return t, pt
    
    def get_reconstruct_loss(self, cat_emb, re_emb, pt):
        loss = mean_flat((cat_emb - re_emb) ** 2)
        # In official code, loss /= pt seems commented out or used differently
        # "terms['loss'] /= pt" in training_losses
        # Here just return raw MSE? 
        # In LightGCN.computer: recons_loss = (user_recons + item_recons) / 2
        # It doesn't seem to use pt here.
        return loss

def betas_from_linear_variance(steps, variance, max_beta=0.999):
    alpha_bar = 1 - variance
    betas = []
    betas.append(1 - alpha_bar[0])
    for i in range(1, steps):
        betas.append(min(1 - alpha_bar[i] / alpha_bar[i - 1], max_beta))
    return np.array(betas)

class DDRM(GeneralRecommender):
    def __init__(self, config, dataset):
        super(DDRM, self).__init__(config, dataset)
        self.config = config
        
        # Parameters
        self.latent_dim = config['embedding_size']
        self.steps = config['steps']
        self.noise_scale = config['noise_scale']
        self.noise_min = config['noise_min']
        self.noise_max = config['noise_max']
        self.reg_weight = config['reg_weight']
        self.alpha = config['alpha']
        self.beta = config['beta']
        
        # Graph
        if hasattr(dataset, 'inter_matrix'):
            self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        elif hasattr(dataset, 'dataset') and hasattr(dataset.dataset, 'inter_matrix'):
            self.interaction_matrix = dataset.dataset.inter_matrix(form='coo').astype(np.float32)
        else:
            raise ValueError("Dataset does not have inter_matrix method")
            
        self.norm_adj = self.get_norm_adj_mat()
        
        # Models
        self.rec_model = LightGCN_Encoder(self.n_users, self.n_items, config)
        self.rec_model.Graph = self.norm_adj.to(self.device)
        
        dims = config['dims'] if isinstance(config['dims'], list) else [config['dims']]
        out_dims = dims + [self.latent_dim]
        in_dims = out_dims[::-1]
        
        self.user_reverse_model = DNN(in_dims, out_dims, self.latent_dim, time_type="cat", norm=config['norm'], act=config['act'])
        self.item_reverse_model = DNN(in_dims, out_dims, self.latent_dim, time_type="cat", norm=config['norm'], act=config['act'])
        
        self.diffusion = GaussianDiffusion(
            noise_scale=self.noise_scale,
            noise_min=self.noise_min,
            noise_max=self.noise_max,
            steps=self.steps,
            device=self.device,
            noise_schedule=config['noise_schedule']
        )
        
    def get_norm_adj_mat(self):
        # Build normalized adjacency matrix for LightGCN
        R = self.interaction_matrix
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
        shape = torch.Size(coo.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

    def apply_noise(self, user_emb, item_emb):
        emb_size = user_emb.shape[0]
        ts, pt = self.diffusion.sample_timesteps(emb_size, self.device, 'uniform')
        
        user_noise = torch.randn_like(user_emb)
        item_noise = torch.randn_like(item_emb)
        
        user_noise_emb = self.diffusion.q_sample(user_emb, ts, user_noise)
        item_noise_emb = self.diffusion.q_sample(item_emb, ts, item_noise)
        
        return user_noise_emb, item_noise_emb, ts, pt

    def calculate_loss(self, interaction):
        user = interaction[0]
        pos_item = interaction[1]
        neg_item = interaction[2]
        
        all_users, all_items = self.rec_model()
        
        users_emb = all_users[user]
        pos_emb = all_items[pos_item]
        neg_emb = all_items[neg_item]
        
        userEmb0 = self.rec_model.embedding_user(user)
        posEmb0 = self.rec_model.embedding_item(pos_item)
        negEmb0 = self.rec_model.embedding_item(neg_item)
        
        # Diffusion Process
        noise_user_emb, noise_item_emb, ts, pt = self.apply_noise(users_emb, pos_emb)
        
        user_model_output = self.user_reverse_model(noise_user_emb, pos_emb, ts)
        item_model_output = self.item_reverse_model(noise_item_emb, users_emb, ts)
        
        user_recons = self.diffusion.get_reconstruct_loss(users_emb, user_model_output, pt)
        item_recons = self.diffusion.get_reconstruct_loss(pos_emb, item_model_output, pt)
        reconstruct_loss = (user_recons + item_recons) / 2
        
        # BPR Loss
        reg_loss = (1/2)*(userEmb0.norm(2).pow(2) + 
                         posEmb0.norm(2).pow(2)  +
                         negEmb0.norm(2).pow(2))/float(len(user))
                         
        pos_scores = torch.mul(users_emb, pos_emb).sum(dim=1)
        neg_scores = torch.mul(users_emb, neg_emb).sum(dim=1)
        
        loss = F.softplus(neg_scores - pos_scores).mean()
        
        # Denoise Data Weighting
        pos_score_ = torch.sigmoid(pos_scores).detach()
        weight = torch.pow(pos_score_, self.beta)
        
        final_loss = (1 - self.alpha) * (loss + reg_loss * self.reg_weight) + self.alpha * reconstruct_loss.mean()
        final_loss = final_loss * weight.mean() # Simple scaling? Official code does element-wise weight * loss then mean.
        # Official: loss = loss * weight; loss = loss.mean()
        # Here my loss is already meaned.
        # Let's re-calculate loss without mean first
        
        loss_element = F.softplus(neg_scores - pos_scores)
        loss_element = (1 - self.alpha) * (loss_element + reg_loss * self.reg_weight) + self.alpha * reconstruct_loss
        loss_element = loss_element * weight
        
        return loss_element.mean()

    def full_sort_predict(self, interaction):
        user = interaction[0]
        
        all_users, all_items = self.rec_model()
        user_emb = all_users[user]
        
        # Inference / Denoising
        # Note: Official code computer_infer
        # noise_user_emb = user_emb (no noise added for user in inference?)
        # Official: noise_user_emb = user_emb
        noise_user_emb = user_emb
        
        # Item Denoising? 
        # Official: noise_emb = self.apply_T_noise(all_aver_item_emb, diff_model)
        # Wait, LightGCN inference usually just dot product user_emb and item_emb.
        # DDRM adds a refinement step using diffusion.
        # It takes average item embedding?
        # In computer_infer: all_aver_item_emb ...
        # For full sort predict, we have too many items. 
        # Applying diffusion to ALL items for EACH user is too expensive?
        # Official code computer_infer takes 'allPos' which implies it might be doing something specific.
        # Actually computer_infer is called by getUsersRating.
        # And getUsersRating is used in Test.
        # In Test: rating = Recmodel.getUsersRating(batch_users_gpu, allPos, ...)
        # It seems it refines the prediction using history items (allPos).
        
        # But for GenMMRec full_sort_predict, we need scores for ALL items.
        # If we follow official logic strictly:
        # 1. Get user_emb from LightGCN
        # 2. Get item_emb from LightGCN
        # 3. Refine item_emb using Diffusion conditioned on user_emb?
        # Official computer_infer logic:
        # noise_emb = apply_T_noise(all_aver_item_emb) -> p_sample -> noise_emb
        # return noise_emb, items
        # And then rounding_inner(item_emb, all_items)
        
        # Wait, what is all_aver_item_emb?
        # "for pos_item in allPos: aver_item_emb = torch.mean(item_emb, dim=0)"
        # It seems to be a prototype item for the user?
        # If we want to predict for all items, we probably just use the LightGCN embeddings directly 
        # OR we need to implement the exact inference logic which seems to generate an ideal item embedding for the user
        # and then find nearest neighbors?
        
        # In DDRM paper/code:
        # "rating = self.rounding_inner(item_emb, all_items)"
        # item_emb is the generated/denoised item embedding (ideal item for user).
        # all_items is the embedding of all candidate items.
        # rounding_inner calculates dot product.
        
        # So the flow is:
        # 1. Generate ideal item embedding for user (using Diffusion, conditioned on user_emb)
        #    Start from noise? Or start from User's history average?
        #    Official: all_aver_item_emb = mean(pos_items).
        #    noise_emb = apply_T_noise(all_aver_item_emb)
        #    denoise...
        
        # We need user's history (pos_items) to compute start point.
        # interaction only has user indices.
        # We can get history from dataset.
        
        # However, GenMMRec's full_sort_predict doesn't pass history.
        # We can access self.interaction_csr or similar.
        
        if not hasattr(self, 'interaction_csr'):
            self.interaction_csr = sp.csr_matrix(self.interaction_matrix)
            
        batch_users_np = user.cpu().numpy()
        # We need to get pos items for these users.
        # self.interaction_csr[u] gives row.
        
        generated_items = []
        
        # We can process in batch, but need to handle variable length history?
        # Official code iterates allPos.
        
        # Let's approximate:
        # If we can't easily get history in tensor friendly way, maybe we can iterate.
        
        # For now, let's try to implement the core idea.
        # 1. Calculate centroid of user history (in LightGCN item embedding space)
        # 2. Add noise and Denoise it to get "Ideal Item"
        # 3. Dot product "Ideal Item" with all Item Embeddings.
        
        # But wait, LightGCN also gives user_emb.
        # Official code: noise_user_emb = user_emb.
        # item_reverse_model(noise_emb, noise_user_emb, t)
        # So it uses LightGCN user_emb as condition.
        
        # What is the starting x_T for item generation?
        # Official: noise_emb = self.apply_T_noise(all_aver_item_emb, diff_model)
        # So it starts from "Noisy Average History".
        
        # Get all item embeddings from LightGCN
        _, all_item_embs = self.rec_model()
        
        # Calculate Average History Embedding
        # This is expensive to do on the fly if not optimized.
        # interaction_csr * all_item_embs ?
        # (Batch_Users x N_Items) * (N_Items x Dim) -> (Batch_Users x Dim)
        # This gives sum. We need mean.
        # We can divide by degree.
        
        batch_csr = self.interaction_csr[batch_users_np]
        # Convert to torch sparse?
        # Or just use CPU mm if batch is small?
        # batch_csr is sparse.
        
        # Let's construct sparse tensor
        coo = batch_csr.tocoo()
        indices = torch.from_numpy(np.vstack((coo.row, coo.col)).astype(np.int64)).to(self.device)
        values = torch.from_numpy(coo.data.astype(np.float32)).to(self.device)
        shape = torch.Size(coo.shape)
        batch_sparse = torch.sparse.FloatTensor(indices, values, shape)
        
        # Sum embeddings
        # sparse mm: (B, I) * (I, D) -> (B, D)
        user_history_sum = torch.sparse.mm(batch_sparse, all_item_embs)
        
        # Count
        user_history_count = torch.sparse.mm(batch_sparse, torch.ones(self.n_items, 1).to(self.device))
        user_history_count = user_history_count.clamp(min=1)
        
        user_history_mean = user_history_sum / user_history_count
        
        # Now Denoise
        x_start = user_history_mean
        
        sampling_steps = self.config['sampling_steps']
        if sampling_steps is None:
            sampling_steps = 0
            
        # Add T noise
        t = torch.tensor([self.steps - 1] * x_start.shape[0]).to(self.device)
        noise = torch.randn_like(x_start)
        x_T = self.diffusion.q_sample(x_start, t, noise)
        
        # Denoise loop
        x_t = x_T
        indices = list(range(sampling_steps))[::-1]
        
        # Condition is User Embedding
        condition = user_emb
        
        for i in indices:
            t = torch.tensor([i] * x_t.shape[0]).to(self.device)
            out = self.diffusion.p_mean_variance(self.item_reverse_model, x_t, condition, t)
            if self.config['sampling_noise']:
                 noise = torch.randn_like(x_t)
                 nonzero_mask = ((t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1))))
                 x_t = out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise
            else:
                 x_t = out["mean"]
        
        generated_item_emb = x_t
        
        # Match with all items
        # scores = generated_item_emb @ all_item_embs.T
        scores = torch.matmul(generated_item_emb, all_item_embs.transpose(0, 1))
        
        return scores
