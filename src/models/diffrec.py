# coding: utf-8
# Desc: Core code of the DiffRec.
# Author: OrangeAI Research Team
# Time: 2026-01-05
# paper: "Diffusion Recommender Model" (SIGIR 2023, DiffRec)
# Ref Link: https://arxiv.org/abs/2304.04971  
# Ref Code: https://github.com/YiyanXu/DiffRec/tree/main/DiffRec

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from common.abstract_recommender import GeneralRecommender

class DNN(nn.Module):
    """
    A deep neural network for the reverse diffusion process.
    """
    def __init__(self, in_dims, out_dims, emb_size, time_type="cat", norm=False, dropout=0.5):
        super(DNN, self).__init__()
        self.in_dims = in_dims
        self.out_dims = out_dims
        assert out_dims[0] == in_dims[-1], "In and out dimensions must equal to each other."
        self.time_type = time_type
        self.time_emb_dim = emb_size
        self.norm = norm

        self.emb_layer = nn.Linear(self.time_emb_dim, self.time_emb_dim)

        if self.time_type == "cat":
            in_dims_temp = [self.in_dims[0] + self.time_emb_dim] + self.in_dims[1:]
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
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for weights
            layer.bias.data.normal_(0.0, 0.001)
        
        for layer in self.out_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for weights
            layer.bias.data.normal_(0.0, 0.001)
        
        size = self.emb_layer.weight.size()
        fan_out = size[0]
        fan_in = size[1]
        std = np.sqrt(2.0 / (fan_in + fan_out))
        self.emb_layer.weight.data.normal_(0.0, std)
        self.emb_layer.bias.data.normal_(0.0, 0.001)
    
    def forward(self, x, timesteps):
        time_emb = timestep_embedding(timesteps, self.time_emb_dim).to(x.device)
        emb = self.emb_layer(time_emb)
        if self.norm:
            x = F.normalize(x)
        x = self.drop(x)
        h = torch.cat([x, emb], dim=-1)
        for i, layer in enumerate(self.in_layers):
            h = layer(h)
            h = torch.tanh(h)
        
        for i, layer in enumerate(self.out_layers):
            h = layer(h)
            if i != len(self.out_layers) - 1:
                h = torch.tanh(h)
        
        return h

def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    """
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
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))

class GaussianDiffusion(nn.Module):
    def __init__(self, mean_type, noise_schedule, noise_scale, noise_min, noise_max,\
            steps, device, history_num_per_term=10, beta_fixed=True):
        super(GaussianDiffusion, self).__init__()

        self.mean_type = mean_type
        self.noise_schedule = noise_schedule
        self.noise_scale = noise_scale
        self.noise_min = noise_min
        self.noise_max = noise_max
        self.steps = steps
        self.device = device

        self.history_num_per_term = history_num_per_term
        self.Lt_history = torch.zeros(steps, history_num_per_term, dtype=torch.float64).to(device)
        self.Lt_count = torch.zeros(steps, dtype=int).to(device)

        if noise_scale != 0.:
            self.betas = torch.tensor(self.get_betas(), dtype=torch.float64).to(self.device)
            if beta_fixed:
                self.betas[0] = 0.00001
            assert len(self.betas.shape) == 1, "betas must be 1-D"
            assert len(self.betas) == self.steps, "num of betas must equal to diffusion steps"
            assert (self.betas > 0).all() and (self.betas <= 1).all(), "betas out of range"

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
            # Fallback or other schedules
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

        self.posterior_log_variance_clipped = torch.log(
            torch.cat([self.posterior_variance[1].unsqueeze(0), self.posterior_variance[1:]])
        )
        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * torch.sqrt(alphas)
            / (1.0 - self.alphas_cumprod)
        )
    
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        return (
            self._extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + self._extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * noise
        )
    
    def p_mean_variance(self, model, x, t):
        model_output = model(x, t)

        model_variance = self.posterior_variance
        model_log_variance = self.posterior_log_variance_clipped

        model_variance = self._extract_into_tensor(model_variance, t, x.shape)
        model_log_variance = self._extract_into_tensor(model_log_variance, t, x.shape)
        
        # We assume mean_type is always x0 for DiffRec as per default config
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

    def _extract_into_tensor(self, arr, timesteps, broadcast_shape):
        arr = arr.to(timesteps.device)
        res = arr[timesteps].float()
        while len(res.shape) < len(broadcast_shape):
            res = res[..., None]
        return res.expand(broadcast_shape)
    
    def SNR(self, t):
        self.alphas_cumprod = self.alphas_cumprod.to(t.device)
        return self.alphas_cumprod[t] / (1 - self.alphas_cumprod[t])

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

    def training_losses(self, model, x_start, reweight=False):
        batch_size, device = x_start.size(0), x_start.device
        ts, pt = self.sample_timesteps(batch_size, device, 'importance' if reweight else 'uniform')
        noise = torch.randn_like(x_start)
        if self.noise_scale != 0.:
            x_t = self.q_sample(x_start, ts, noise)
        else:
            x_t = x_start

        terms = {}
        model_output = model(x_t, ts)
        target = x_start # Assuming x0 prediction

        mse = mean_flat((target - model_output) ** 2)

        if reweight:
            weight = self.SNR(ts - 1) - self.SNR(ts)
            weight = torch.where((ts == 0), torch.tensor(1.0).to(device), weight)
            loss = mse
        else:
            weight = torch.tensor([1.0] * len(target)).to(device)

        terms["loss"] = weight * loss
        
        # Update history for importance sampling
        # We detach to avoid graph retention
        if reweight:
            for t_val, loss_val in zip(ts, terms["loss"]):
                t_idx = t_val.item()
                if self.Lt_count[t_idx] < self.history_num_per_term:
                    self.Lt_history[t_idx, self.Lt_count[t_idx]] = loss_val.detach()
                    self.Lt_count[t_idx] += 1
                else:
                    self.Lt_history[t_idx, :-1] = self.Lt_history[t_idx, 1:].clone()
                    self.Lt_history[t_idx, -1] = loss_val.detach()

        terms["loss"] /= pt
        return terms
    
    def p_sample(self, model, x_start, steps, sampling_noise=False):
        # x_start is the history (user vector)
        if steps == 0:
            x_t = x_start
        else:
            t = torch.tensor([steps - 1] * x_start.shape[0]).to(x_start.device)
            x_t = self.q_sample(x_start, t)
            
        indices = list(range(self.steps))[::-1]

        for i in indices:
            t = torch.tensor([i] * x_t.shape[0]).to(x_start.device)
            out = self.p_mean_variance(model, x_t, t)
            if sampling_noise:
                noise = torch.randn_like(x_t)
                nonzero_mask = ((t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1))))
                x_t = out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise
            else:
                x_t = out["mean"]
        return x_t


class DiffRec(GeneralRecommender):
    def __init__(self, config, dataset):
        super(DiffRec, self).__init__(config, dataset)
        self.config = config
        
        # Parameters
        self.steps = config['steps']
        self.noise_scale = config['noise_scale']
        self.noise_min = config['noise_min']
        self.noise_max = config['noise_max']
        
        # Data
        import scipy.sparse as sp
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        
        # Build Gaussian Diffusion
        self.diffusion = GaussianDiffusion(
            mean_type='x0',
            noise_schedule=config['noise_schedule'],
            noise_scale=self.noise_scale,
            noise_min=self.noise_min,
            noise_max=self.noise_max,
            steps=self.steps,
            device=self.device
        )
        
        # Build MLP
        dims = config['dims'] if isinstance(config['dims'], list) else [config['dims']]
        # Official code: out_dims = eval(args.dims) + [n_item]
        # in_dims = out_dims[::-1]
        out_dims = dims + [self.n_items]
        in_dims = out_dims[::-1]
        
        self.model = DNN(
            in_dims=in_dims,
            out_dims=out_dims,
            emb_size=config['embedding_size'],
            time_type="cat",
            norm=False, # Default in official code
            dropout=config['dropout']
        )

    def calculate_loss(self, interaction):
        user = interaction[0]
        
        # Get history vectors
        if not hasattr(self, 'interaction_csr'):
            import scipy.sparse as sp
            self.interaction_csr = sp.csr_matrix(self.interaction_matrix)
            
        batch_users_np = user.cpu().numpy()
        batch_vectors = self.interaction_csr[batch_users_np].toarray()
        x_start = torch.from_numpy(batch_vectors).float().to(self.device)
        
        terms = self.diffusion.training_losses(self.model, x_start, reweight=self.config['reweight'])
        return terms['loss'].mean()
        
    def full_sort_predict(self, interaction):
        user = interaction[0]
        
        # Get history vectors
        if not hasattr(self, 'interaction_csr'):
            import scipy.sparse as sp
            self.interaction_csr = sp.csr_matrix(self.interaction_matrix)
            
        batch_users_np = user.cpu().numpy()
        batch_vectors = self.interaction_csr[batch_users_np].toarray()
        x_start = torch.from_numpy(batch_vectors).float().to(self.device)
        
        # Inference
        sampling_steps = self.config['sampling_steps']
        if sampling_steps is None:
            sampling_steps = 0
        prediction = self.diffusion.p_sample(self.model, x_start, sampling_steps, sampling_noise=False)
        
        return prediction

def betas_from_linear_variance(steps, variance, max_beta=0.999):
    alpha_bar = 1 - variance
    betas = []
    betas.append(1 - alpha_bar[0])
    for i in range(1, steps):
        betas.append(min(1 - alpha_bar[i] / alpha_bar[i - 1], max_beta))
    return np.array(betas)
