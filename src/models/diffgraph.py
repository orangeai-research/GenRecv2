import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
import math
import dgl
import dgl.function as fn
from common.abstract_recommender import GeneralRecommender

class DGLLayer(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 weight=False,
                 bias=False,
                 activation=None):
        super(DGLLayer, self).__init__()
        self.bias = bias
        self._in_feats = in_feats
        self._out_feats = out_feats
        self.weight = weight
        if self.weight:
            self.u_w = nn.Parameter(torch.Tensor(in_feats, out_feats))
            self.v_w = nn.Parameter(torch.Tensor(in_feats, out_feats))
            nn.init.xavier_uniform_(self.u_w)
            nn.init.xavier_uniform_(self.v_w)
            
        self._activation = activation

    def forward(self, graph, u_f, v_f):
        with graph.local_scope():
            if self.weight:
                u_f = torch.mm(u_f, self.u_w)
                v_f = torch.mm(v_f, self.v_w)
                
            node_f = torch.cat([u_f, v_f], dim=0)
            
            # D^-1/2
            degs = graph.out_degrees().to(u_f.device).float().clamp(min=1)
            norm = torch.pow(degs, -0.5).view(-1, 1)

            node_f = node_f * norm

            graph.ndata['n_f'] = node_f
            graph.update_all(fn.copy_u(u='n_f', out='m'), reduce_func=fn.sum(msg='m', out='n_f'))

            rst = graph.ndata['n_f']

            degs = graph.in_degrees().to(u_f.device).float().clamp(min=1)
            norm = torch.pow(degs, -0.5).view(-1, 1)
            rst = rst * norm

            if self._activation is not None:
                rst = self._activation(rst)

            return rst

class Denoise(nn.Module):
    def __init__(self, in_dims, out_dims, emb_size, norm=False, dropout=0.5, device='cpu'):
        super(Denoise, self).__init__()
        self.in_dims = in_dims
        self.out_dims = out_dims
        self.time_emb_dim = emb_size
        self.norm = norm
        self.device = device

        self.emb_layer = nn.Linear(self.time_emb_dim, self.time_emb_dim)

        in_dims_temp = [self.in_dims[0] + self.time_emb_dim] + self.in_dims[1:]
        out_dims_temp = self.out_dims

        self.in_layers = nn.ModuleList([nn.Linear(d_in, d_out) for d_in, d_out in zip(in_dims_temp[:-1], in_dims_temp[1:])])
        self.out_layers = nn.ModuleList([nn.Linear(d_in, d_out) for d_in, d_out in zip(out_dims_temp[:-1], out_dims_temp[1:])])

        self.drop = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self):
        for layer in self.in_layers:
            size = layer.weight.size()
            std = np.sqrt(2.0 / (size[0] + size[1]))
            layer.weight.data.normal_(0.0, std)
            layer.bias.data.normal_(0.0, 0.001)
        
        for layer in self.out_layers:
            size = layer.weight.size()
            std = np.sqrt(2.0 / (size[0] + size[1]))
            layer.weight.data.normal_(0.0, std)
            layer.bias.data.normal_(0.0, 0.001)

        size = self.emb_layer.weight.size()
        std = np.sqrt(2.0 / (size[0] + size[1]))
        self.emb_layer.weight.data.normal_(0.0, std)
        self.emb_layer.bias.data.normal_(0.0, 0.001)

    def forward(self, x, timesteps, mess_dropout=True):
        freqs = torch.exp(-math.log(10000) * torch.arange(start=0, end=self.time_emb_dim//2, dtype=torch.float32) / (self.time_emb_dim//2)).to(self.device)
        temp = timesteps[:, None].float() * freqs[None]
        time_emb = torch.cat([torch.cos(temp), torch.sin(temp)], dim=-1)
        if self.time_emb_dim % 2:
            time_emb = torch.cat([time_emb, torch.zeros_like(time_emb[:, :1])], dim=-1)
        emb = self.emb_layer(time_emb)
        if self.norm:
            x = F.normalize(x)
        if mess_dropout:
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

class GaussianDiffusion(nn.Module):
    def __init__(self, noise_scale, noise_min, noise_max, steps, device='cpu', beta_fixed=True):
        super(GaussianDiffusion, self).__init__()

        self.noise_scale = noise_scale
        self.noise_min = noise_min
        self.noise_max = noise_max
        self.steps = steps
        self.device = device

        self.history_num_per_term = 10
        self.Lt_history = torch.zeros(steps, 10, dtype=torch.float64).to(device)
        self.Lt_count = torch.zeros(steps, dtype=int).to(device)

        if noise_scale != 0:
            self.betas = torch.tensor(self.get_betas(), dtype=torch.float64).to(device)
            if beta_fixed:
                self.betas[0] = 0.0001

            self.calculate_for_diffusion()

    def get_betas(self):
        start = self.noise_scale * self.noise_min
        end = self.noise_scale * self.noise_max
        variance = np.linspace(start, end, self.steps, dtype=np.float64)
        alpha_bar = 1 - variance
        betas = []
        betas.append(1 - alpha_bar[0])
        for i in range(1, self.steps):
            betas.append(min(1 - alpha_bar[i] / alpha_bar[i-1], 0.999))
        return np.array(betas)
    
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

    def p_sample(self, model, x_start, steps):
        if steps == 0:
            x_t = x_start
        else:
            t = torch.tensor([steps-1] * x_start.shape[0]).to(self.device)
            x_t = self.q_sample(x_start, t)
        
        indices = list(range(self.steps))[::-1]

        for i in indices:
            t = torch.tensor([i] * x_t.shape[0]).to(self.device)
            model_mean, model_log_variance = self.p_mean_variance(model, x_t, t)
            x_t = model_mean
        return x_t
            
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        return self._extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start + self._extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise

    def _extract_into_tensor(self, arr, timesteps, broadcast_shape):
        arr = arr.to(self.device)
        res = arr[timesteps].float()
        while len(res.shape) < len(broadcast_shape):
            res = res[..., None]
        return res.expand(broadcast_shape)

    def p_mean_variance(self, model, x, t):
        model_output = model(x, t, False)

        model_variance = self.posterior_variance
        model_log_variance = self.posterior_log_variance_clipped

        model_variance = self._extract_into_tensor(model_variance, t, x.shape)
        model_log_variance = self._extract_into_tensor(model_log_variance, t, x.shape)

        model_mean = (self._extract_into_tensor(self.posterior_mean_coef1, t, x.shape) * model_output + self._extract_into_tensor(self.posterior_mean_coef2, t, x.shape) * x)
        
        return model_mean, model_log_variance

    def training_losses2(self, model, targetEmbeds, x_start, batch):
        batch_size = x_start.size(0)
        # ts, pt = self.sample_timesteps(batch_size, device,'importance')
        ts = torch.randint(0, self.steps, (batch_size,)).long().to(self.device)
        noise = torch.randn_like(x_start)
        if self.noise_scale != 0:
            x_t = self.q_sample(x_start, ts, noise)
        else:
            x_t = x_start

        model_output = model(x_t, ts)
        mse = self.mean_flat((targetEmbeds - model_output) ** 2)
        # mse = cal_infonce_loss(targetEmbeds,model_output,args.temp)
        weight = self.SNR(ts - 1) - self.SNR(ts)
        weight = torch.where((ts == 0), torch.tensor(1.0).to(self.device), weight)
        diff_loss = weight * mse
        
        # Note: Original code indexed diff_loss with [batch]. 
        # But here x_start is already full embedding (N, dim).
        # And batch is indices.
        # However, diff_loss is (N,).
        # We should return full diff_loss or indexed?
        # Model.py: u_diff_loss, ... = training_losses2(..., ancs)
        # diff_loss = diff_loss[batch]
        # So we should index it here.
        diff_loss = diff_loss[batch]
        return diff_loss, model_output
        
    def mean_flat(self, tensor):
        return tensor.mean(dim=list(range(1, len(tensor.shape))))

    def SNR(self, t):
        self.alphas_cumprod = self.alphas_cumprod.to(self.device)
        return self.alphas_cumprod[t] / (1 - self.alphas_cumprod[t])

class DiffGraph(GeneralRecommender):
    def __init__(self, config, dataset):
        super(DiffGraph, self).__init__(config, dataset)
        self.config = config
        
        # Parameters
        self.latdim = config['embedding_size'] # latdim
        self.gcn_layer = config['gcn_layer']
        self.steps = config['steps']
        self.noise_scale = config['noise_scale']
        self.noise_min = config['noise_min']
        self.noise_max = config['noise_max']
        self.reg_weight = config['reg_weight']
        
        # Graph Construction
        if hasattr(dataset, 'inter_matrix'):
            self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        elif hasattr(dataset, 'dataset') and hasattr(dataset.dataset, 'inter_matrix'):
            self.interaction_matrix = dataset.dataset.inter_matrix(form='coo').astype(np.float32)
        else:
            raise ValueError("Dataset does not have inter_matrix method")
            
        self.target_adj = self.get_norm_adj_mat().to(self.device)
        self.behavior_mats = [self.target_adj] # Assuming single behavior for now
        
        # Model Components
        self.embedding_dict = self.init_weight(self.n_users, self.n_items, self.latdim)
        self.act = nn.LeakyReLU(0.5, inplace=True)
        self.layers = nn.ModuleList()
        self.hter_layers = nn.ModuleList()
        self.weight = False # args.weight default? Not in params.py, but Model.py says weight=False
        
        for i in range(0, self.gcn_layer):
            self.layers.append(DGLLayer(self.latdim, self.latdim, weight=self.weight, bias=False, activation=self.act))
            
        for i in range(0, len(self.behavior_mats)):
            single_layers = nn.ModuleList()
            for j in range(0, self.gcn_layer):
                single_layers.append(DGLLayer(self.latdim, self.latdim, weight=self.weight, bias=False, activation=self.act))
            self.hter_layers.append(single_layers)
            
        self.diffusion_process = GaussianDiffusion(self.noise_scale, self.noise_min, self.noise_max, self.steps, device=self.device)
        
        dims = config['dims'] if isinstance(config['dims'], list) else [config['dims']]
        out_dims = dims + [self.latdim]
        in_dims = out_dims[::-1]
        
        self.usr_denoiser = Denoise(in_dims, out_dims, config['d_emb_size'], norm=config['norm'], dropout=config['dropout'], device=self.device)
        self.item_denoiser = Denoise(in_dims, out_dims, config['d_emb_size'], norm=config['norm'], dropout=config['dropout'], device=self.device)
        self.final_act = nn.LeakyReLU(negative_slope=0.5)

    def init_weight(self, userNum, itemNum, hide_dim):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(userNum, hide_dim))),
            'item_emb': nn.Parameter(initializer(torch.empty(itemNum, hide_dim))),
        })
        return embedding_dict

    def get_norm_adj_mat(self):
        # Build DGL Graph
        R = self.interaction_matrix
        n_users = self.n_users
        n_items = self.n_items
        
        # Create A
        # In DGL, we want a graph with user and item nodes
        # row: users, col: items
        # We need a unified node ID space: users [0, n_users), items [n_users, n_users+n_items)
        
        u_ids = R.row
        i_ids = R.col + n_users
        
        # Add bidirectional edges
        src = np.concatenate([u_ids, i_ids])
        dst = np.concatenate([i_ids, u_ids])
        
        g = dgl.graph((src, dst), num_nodes=n_users+n_items)
        
        # Add self-loops? LightGCN typically does not add self-loops for A, but adds I later or handles it.
        # DGL implementation in DiffGraph/Model.py:
        # degs = graph.out_degrees()...
        # It seems they rely on graph structure.
        # Let's check if they add self-loops.
        # Official code uses DataHandler to load graph.
        # Usually LightGCN-style graph does not have self-loops in A.
        
        return g

    def forward(self):
        init_embedding = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], dim=0)
        init_heter_embedding = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], dim=0)
        
        all_embeddings = [init_embedding]
        all_heter_embeddings = []
        
        # Target Graph Propagation
        for i, layer in enumerate(self.layers):
            if i == 0:
                embeddings = layer(self.target_adj, self.embedding_dict['user_emb'], self.embedding_dict['item_emb'])
            else:
                embeddings = layer(self.target_adj, embeddings[:self.n_users], embeddings[self.n_users:])
                
            norm_embeddings = F.normalize(embeddings, p=2, dim=1)
            all_embeddings.append(norm_embeddings)
            
        ui_embeddings = sum(all_embeddings)
        
        # Behavior Graphs Propagation
        for i in range(len(self.behavior_mats)):
            sub_heter_embeddings = [init_heter_embedding]
            for j, layer in enumerate(self.hter_layers[i]):
                if j == 0:
                    embeddings = layer(self.behavior_mats[i], self.embedding_dict['user_emb'], self.embedding_dict['item_emb'])
                else:
                    embeddings = layer(self.behavior_mats[i], embeddings[:self.n_users], embeddings[self.n_users:])
                    
                norm_embeddings = F.normalize(embeddings, p=2, dim=1)
                sub_heter_embeddings.append(norm_embeddings)
            sub_heter_embeddings = sum(sub_heter_embeddings)
            all_heter_embeddings.append(sub_heter_embeddings)
            
        all_heter_embeddings = sum(all_heter_embeddings)
        
        target_user_embedding = ui_embeddings[:self.n_users]
        target_item_embedding = ui_embeddings[self.n_users:]
        
        heter_user_embedding = all_heter_embeddings[:self.n_users]
        heter_item_embedding = all_heter_embeddings[self.n_users:]
        
        return target_user_embedding, target_item_embedding, heter_user_embedding, heter_item_embedding

    def calculate_loss(self, interaction):
        user = interaction[0]
        pos_item = interaction[1]
        neg_item = interaction[2]
        
        usrEmbeds, itmEmbeds, h_usrEmbeds, h_itemEmbeds = self.forward()
        
        u_diff_loss, diff_usrEmbeds = self.diffusion_process.training_losses2(self.usr_denoiser, usrEmbeds, h_usrEmbeds, user)
        i_diff_loss, diff_itemEmbeds = self.diffusion_process.training_losses2(self.item_denoiser, itmEmbeds, h_itemEmbeds, pos_item) # Note: DiffGraph uses 'poss' for item diffusion loss
        
        diff_loss = (u_diff_loss.mean() + i_diff_loss.mean())
        
        # Refine Embeddings
        # Note: In Model.py, diff_usrEmbeds returned from training_losses2 IS the model_output (predicted target).
        # And usrEmbeds = usrEmbeds + diff_usrEmbeds
        # BUT wait, usrEmbeds is ALREADY the target (UI graph embedding).
        # diff_usrEmbeds is the PREDICTED target from Heter graph.
        # Adding them together effectively ensembles the GCN embedding and Diffusion-predicted embedding.
        usrEmbeds = usrEmbeds + diff_usrEmbeds
        itmEmbeds = itmEmbeds + diff_itemEmbeds
        
        ancEmbeds = usrEmbeds[user]
        posEmbeds = itmEmbeds[pos_item]
        negEmbeds = itmEmbeds[neg_item]
        
        # BPR Loss
        pos_scores = torch.sum(ancEmbeds * posEmbeds, dim=1)
        neg_scores = torch.sum(ancEmbeds * negEmbeds, dim=1)
        scoreDiff = pos_scores - neg_scores
        
        bprLoss = - (scoreDiff).sigmoid().log().mean()
        regLoss = (torch.norm(ancEmbeds) ** 2 + torch.norm(posEmbeds) ** 2 + torch.norm(negEmbeds) ** 2) * self.reg_weight / len(user)
        
        loss = bprLoss + regLoss + diff_loss
        return loss

    def full_sort_predict(self, interaction):
        user = interaction[0]
        
        usrEmbeds, itmEmbeds, h_usrEmbeds, h_itemEmbeds = self.forward()
        
        sampling_steps = self.config['sampling_steps']
        if sampling_steps is None:
            sampling_steps = 0
            
        denoised_u = self.diffusion_process.p_sample(self.usr_denoiser, h_usrEmbeds, sampling_steps)
        denoised_i = self.diffusion_process.p_sample(self.item_denoiser, h_itemEmbeds, sampling_steps)
        
        usrEmbeds = usrEmbeds + denoised_u
        itmEmbeds = itmEmbeds + denoised_i
        
        user_e = usrEmbeds[user]
        all_item_e = itmEmbeds
        
        scores = torch.matmul(user_e, all_item_e.transpose(0, 1))
        return scores
