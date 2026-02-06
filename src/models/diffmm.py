# coding: utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import scipy.sparse as sp
from common.abstract_recommender import GeneralRecommender

# 生成质量四指标（与 GenRecv2 一致，便于跨模型对比）
from models.genrecv2 import sliced_wasserstein_distance, fid_embedding, trajectory_linearity

init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform

class DiffMM(GeneralRecommender):
    def __init__(self, config, dataset):
        super(DiffMM, self).__init__(config, dataset)

        # Config parameters
        self.latdim = config['embedding_size']
        self.gnn_layer = config['n_layers']
        self.keepRate = config['keep_rate']
        self.trans = config['trans_type'] # 0, 1, 2
        self.ris_adj_lambda = config['ris_adj_lambda']
        self.ris_lambda = config['ris_lambda']
        self.cl_method = config['cl_method']
        self.ssl_reg = config['ssl_reg']
        self.temp = config['temperature']
        self.reg_weight = config['reg_weight']
        
        # Diffusion parameters
        self.noise_scale = config['noise_scale']
        self.noise_min = config['noise_min']
        self.noise_max = config['noise_max']
        self.steps = config['steps']
        self.e_loss = config['e_loss']
        self.sampling_steps = config['sampling_steps']
        self.sampling_noise = config['sampling_noise']
        self.rebuild_k = config['rebuild_k']
        self.d_emb_size = config['d_emb_size']
        self.norm = config['norm']

        # Embeddings
        self.uEmbeds = nn.Parameter(init(torch.empty(self.n_users, self.latdim)))
        self.iEmbeds = nn.Parameter(init(torch.empty(self.n_items, self.latdim)))
        self.gcnLayers = nn.ModuleList([GCNLayer() for i in range(self.gnn_layer)])
        
        self.edgeDropper = SpAdjDropEdge(self.keepRate)

        # Feature Transformation
        # v_feat and t_feat are loaded by GeneralRecommender
        self.image_feat_dim = self.v_feat.shape[1] if self.v_feat is not None else 0
        self.text_feat_dim = self.t_feat.shape[1] if self.t_feat is not None else 0

        if self.trans == 1:
            self.image_trans = nn.Linear(self.image_feat_dim, self.latdim)
            self.text_trans = nn.Linear(self.text_feat_dim, self.latdim)
        elif self.trans == 0:
            self.image_trans = nn.Parameter(init(torch.empty(size=(self.image_feat_dim, self.latdim))))
            self.text_trans = nn.Parameter(init(torch.empty(size=(self.text_feat_dim, self.latdim))))
        else:
            self.image_trans = nn.Parameter(init(torch.empty(size=(self.image_feat_dim, self.latdim))))
            self.text_trans = nn.Linear(self.text_feat_dim, self.latdim)
            
        self.modal_weight = nn.Parameter(torch.Tensor([0.5, 0.5]))
        self.softmax = nn.Softmax(dim=0)
        self.dropout = nn.Dropout(p=0.1)
        self.leakyrelu = nn.LeakyReLU(0.2)

        # Diffusion Models
        self.diffusion_model = GaussianDiffusion(self.noise_scale, self.noise_min, self.noise_max, self.steps).to(self.device)
        
        # Denoise Models
        # Dims need to be defined based on config or data
        # Typically [input_dim, hidden_dims..., item_num]
        dims = config['dims'] # e.g. [1000]
        out_dims = dims + [self.n_items]
        in_dims = out_dims[::-1]
        
        self.denoise_model_image = Denoise(in_dims, out_dims, self.d_emb_size, norm=self.norm).to(self.device)
        self.denoise_model_text = Denoise(in_dims, out_dims, self.d_emb_size, norm=self.norm).to(self.device)

        # Generated Matrices (Placeholders)
        self.image_UI_matrix = None
        self.text_UI_matrix = None
        
        # Pre-calculate graph
        self.norm_adj = self.get_norm_adj_mat(dataset.inter_matrix(form='coo').astype(np.float32)).to(self.device)

    def get_norm_adj_mat(self, interaction_matrix):
        A = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        inter_M = interaction_matrix
        inter_M_t = interaction_matrix.transpose()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.n_users), [1] * inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row + self.n_users, inter_M_t.col), [1] * inter_M_t.nnz)))
        # A._update(data_dict)
        for key, value in data_dict.items() :
            A[key] = value
        sumArr = (A > 0).sum(axis=1)
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(L.data)
        return torch.sparse.FloatTensor(i, data, torch.Size((self.n_users + self.n_items, self.n_users + self.n_items)))

    def getItemEmbeds(self):
        return self.iEmbeds
    
    def getUserEmbeds(self):
        return self.uEmbeds
    
    def getImageFeats(self):
        if self.trans == 0 or self.trans == 2:
            image_feats = self.leakyrelu(torch.mm(self.v_feat, self.image_trans))
            return image_feats
        else:
            return self.image_trans(self.v_feat)
    
    def getTextFeats(self):
        if self.trans == 0:
            text_feats = self.leakyrelu(torch.mm(self.t_feat, self.text_trans))
            return text_feats
        else:
            return self.text_trans(self.t_feat)

    def forward_MM(self, adj, image_adj, text_adj):
        image_feats = self.getImageFeats()
        text_feats = self.getTextFeats()

        weight = self.softmax(self.modal_weight)

        embedsImageAdj = torch.concat([self.uEmbeds, self.iEmbeds])
        embedsImageAdj = torch.spmm(image_adj, embedsImageAdj)

        embedsImage = torch.concat([self.uEmbeds, F.normalize(image_feats)])
        embedsImage = torch.spmm(adj, embedsImage)

        embedsImage_ = torch.concat([embedsImage[:self.n_users], self.iEmbeds])
        embedsImage_ = torch.spmm(adj, embedsImage_)
        embedsImage += embedsImage_
        
        embedsTextAdj = torch.concat([self.uEmbeds, self.iEmbeds])
        embedsTextAdj = torch.spmm(text_adj, embedsTextAdj)

        embedsText = torch.concat([self.uEmbeds, F.normalize(text_feats)])
        embedsText = torch.spmm(adj, embedsText)

        embedsText_ = torch.concat([embedsText[:self.n_users], self.iEmbeds])
        embedsText_ = torch.spmm(adj, embedsText_)
        embedsText += embedsText_

        embedsImage += self.ris_adj_lambda * embedsImageAdj
        embedsText += self.ris_adj_lambda * embedsTextAdj
        
        embedsModal = weight[0] * embedsImage + weight[1] * embedsText

        embeds = embedsModal
        embedsLst = [embeds]
        for gcn in self.gcnLayers:
            embeds = gcn(adj, embedsLst[-1])
            embedsLst.append(embeds)
        embeds = sum(embedsLst)

        embeds = embeds + self.ris_lambda * F.normalize(embedsModal)

        return embeds[:self.n_users], embeds[self.n_users:]

    def forward_cl_MM(self, adj, image_adj, text_adj):
        image_feats = self.getImageFeats()
        text_feats = self.getTextFeats()

        embedsImage = torch.concat([self.uEmbeds, F.normalize(image_feats)])
        embedsImage = torch.spmm(image_adj, embedsImage)

        embedsText = torch.concat([self.uEmbeds, F.normalize(text_feats)])
        embedsText = torch.spmm(text_adj, embedsText)

        embeds1 = embedsImage
        embedsLst1 = [embeds1]
        for gcn in self.gcnLayers:
            embeds1 = gcn(adj, embedsLst1[-1])
            embedsLst1.append(embeds1)
        embeds1 = sum(embedsLst1)

        embeds2 = embedsText
        embedsLst2 = [embeds2]
        for gcn in self.gcnLayers:
            embeds2 = gcn(adj, embedsLst2[-1])
            embedsLst2.append(embeds2)
        embeds2 = sum(embedsLst2)

        return embeds1[:self.n_users], embeds1[self.n_users:], embeds2[:self.n_users], embeds2[self.n_users:]

    def reg_loss(self):
        ret = 0
        ret += self.uEmbeds.norm(2).square()
        ret += self.iEmbeds.norm(2).square()
        return ret

    def calculate_loss(self, interaction):
        # This is for the Rec step
        # interaction: [users, pos_items, neg_items]
        # But wait, MMRec passes a BatchInteraction object or something similar. 
        # Actually GeneralRecommender usually handles interaction in trainer.
        # But here we assume we get users, pos_items, neg_items tensors directly or extract them.
        
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]
        
        if self.image_UI_matrix is None or self.text_UI_matrix is None:
             # Fallback if training starts without generated matrices (shouldn't happen with correct trainer)
             return torch.tensor(0.0, requires_grad=True).to(self.device)

        usrEmbeds, itmEmbeds = self.forward_MM(self.norm_adj, self.image_UI_matrix, self.text_UI_matrix)
        
        ancEmbeds = usrEmbeds[users]
        posEmbeds = itmEmbeds[pos_items]
        negEmbeds = itmEmbeds[neg_items]
        
        # BPR Loss
        pos_scores = torch.mul(ancEmbeds, posEmbeds).sum(dim=1)
        neg_scores = torch.mul(ancEmbeds, negEmbeds).sum(dim=1)
        bprLoss = -torch.log(1e-10 + torch.sigmoid(pos_scores - neg_scores)).mean()
        
        regLoss = self.reg_loss() * self.reg_weight
        
        # CL Loss
        usrEmbeds1, itmEmbeds1, usrEmbeds2, itmEmbeds2 = self.forward_cl_MM(self.norm_adj, self.image_UI_matrix, self.text_UI_matrix)
        
        # M vs M
        clLoss_m_vs_m = (self.contrastLoss(usrEmbeds1, usrEmbeds2, users, self.temp) + self.contrastLoss(itmEmbeds1, itmEmbeds2, pos_items, self.temp)) * self.ssl_reg
        
        # M vs Main
        clLoss1 = (self.contrastLoss(usrEmbeds, usrEmbeds1, users, self.temp) + self.contrastLoss(itmEmbeds, itmEmbeds1, pos_items, self.temp)) * self.ssl_reg
        clLoss2 = (self.contrastLoss(usrEmbeds, usrEmbeds2, users, self.temp) + self.contrastLoss(itmEmbeds, itmEmbeds2, pos_items, self.temp)) * self.ssl_reg
        clLoss_m_vs_main = clLoss1 + clLoss2
        
        if self.cl_method == 1:
            clLoss = clLoss_m_vs_main
        else:
            clLoss = clLoss_m_vs_m

        loss = bprLoss + regLoss + clLoss
        
        return loss

    def contrastLoss(self, embeds1, embeds2, nodes, temp):
        embeds1 = F.normalize(embeds1 + 1e-8, p=2)
        embeds2 = F.normalize(embeds2 + 1e-8, p=2)
        pckEmbeds1 = embeds1[nodes]
        pckEmbeds2 = embeds2[nodes]
        nume = torch.exp(torch.sum(pckEmbeds1 * pckEmbeds2, dim=-1) / temp)
        deno = torch.exp(pckEmbeds1 @ embeds2.T / temp).sum(-1)
        return -torch.log(nume / deno).mean()

    def full_sort_predict(self, interaction):
        user = interaction[0]
        
        if self.image_UI_matrix is None:
             # During eval if not trained yet? or if just loaded.
             # We might need to generate or just use empty/identity if allowed.
             # For safety, let's use norm_adj as fallback or zeros.
             # But really, eval happens after training.
             # If loading from checkpoint, we might miss the generated matrices since they aren't params.
             # We should probably regenerate them or save them.
             # For now, let's assume they exist in memory during training-eval loop.
             pass

        # If evaluating without training (e.g. load model), we need to run diffusion once to get matrices.
        # This is a limitation we should note.
        
        usrEmbeds, itmEmbeds = self.forward_MM(self.norm_adj, self.image_UI_matrix, self.text_UI_matrix)
        score_mat_ui = torch.matmul(usrEmbeds[user], itmEmbeds.transpose(0, 1))
        return score_mat_ui

    @torch.no_grad()
    def _get_origin_and_generated_embeds(self):
        """原始 = 不用生成图时的 forward_MM；生成 = 扩散生成的图经 GCN(forward_MM) 后的兴趣表征。"""
        if self.image_UI_matrix is None or self.text_UI_matrix is None:
            return None, None
        # origin: 用 norm_adj 替代 image/text 邻接（不用扩散生成的图）
        usr_orig, itm_orig = self.forward_MM(self.norm_adj, self.norm_adj, self.norm_adj)
        origin = torch.cat([usr_orig, itm_orig], dim=0)
        # generated: 扩散生成的图经 forward_MM 得到的兴趣表征
        usr_gen, itm_gen = self.forward_MM(self.norm_adj, self.image_UI_matrix, self.text_UI_matrix)
        generated = torch.cat([usr_gen, itm_gen], dim=0)
        return origin, generated

    def compute_gen_quality_metrics(self, n_proj=32, fid_eps=1e-6):
        """
        计算生成兴趣表征质量指标：SWD、平均余弦、MSE、FID。
        原始 = forward_MM(norm_adj, norm_adj, norm_adj)；生成 = forward_MM(norm_adj, image_UI_matrix, text_UI_matrix)。
        """
        origin, gen = self._get_origin_and_generated_embeds()
        if origin is None or gen is None:
            return {}
        origin = origin.detach().float()
        gen = gen.detach().float()
        o_n = F.normalize(origin, dim=1)
        g_n = F.normalize(gen, dim=1)
        cos_per_sample = (o_n * g_n).sum(dim=1)
        swd_val = float(sliced_wasserstein_distance(origin, gen, n_proj=n_proj, chunk_size=4096).cpu().item())
        cosine_val = float(cos_per_sample.mean().cpu().item())
        mse_val = float(F.mse_loss(gen, origin).cpu().item())
        fid_val = float(fid_embedding(origin, gen, eps=fid_eps).cpu().item())
        return {
            "gen_swd": swd_val,
            "gen_cosine": cosine_val,
            "gen_mse": mse_val,
            "gen_fid": fid_val,
        }

    def compute_gen_quality_metrics_cross(self, origin_ref, n_proj=32, fid_eps=1e-6):
        """
        跨模型可比：本模型「生成」表征 vs 共同参考 origin_ref（GenRecv2 的原始兴趣编码）。
        origin_ref 需与 generated 同 shape (n_users+n_items, latdim)，且 latdim 与 GenRecv2 的 embedding_dim 一致。
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

    @torch.no_grad()
    def compute_trajectory_linearity(self, n_steps=None, return_trajectory_for_viz=False, start_noise=None):
        """
        轨迹线性度 (Trajectory Linearity)：扩散逆过程轨迹的直线性，与 GenRecv2 同指标便于对比。
        使用 image 模态的 denoise 与扩散步数，轨迹形状 (T+1, n_users, n_items)。
        """
        steps = n_steps if n_steps is not None else self.sampling_steps
        steps = min(steps, self.diffusion_model.steps)
        if steps < 1:
            return {}
        # UI 块：norm_adj 的 [0:n_users, n_users:n_users+n_items]
        idx = self.norm_adj._indices()
        v = self.norm_adj._values()
        mask = (idx[0] < self.n_users) & (idx[1] >= self.n_users)
        r = idx[0][mask]
        c = idx[1][mask] - self.n_users
        x_start = torch.zeros(self.n_users, self.n_items, device=self.device, dtype=torch.float32)
        x_start[r, c] = v[mask].float()
        _, trajectory = self.diffusion_model.p_sample_with_trajectory(
            self.denoise_model_image, x_start, steps=steps, sampling_noise=self.sampling_noise
        )
        linearity = trajectory_linearity(trajectory)
        result = {"trajectory_linearity": float(linearity.cpu().item())}
        if return_trajectory_for_viz:
            result["trajectory"] = trajectory.cpu()
        return result

class GCNLayer(nn.Module):
    def __init__(self):
        super(GCNLayer, self).__init__()

    def forward(self, adj, embeds):
        return torch.spmm(adj, embeds)

class SpAdjDropEdge(nn.Module):
    def __init__(self, keepRate):
        super(SpAdjDropEdge, self).__init__()
        self.keepRate = keepRate

    def forward(self, adj):
        vals = adj._values()
        idxs = adj._indices()
        edgeNum = vals.size()
        mask = ((torch.rand(edgeNum) + self.keepRate).floor()).type(torch.bool)

        newVals = vals[mask] / self.keepRate
        newIdxs = idxs[:, mask]

        return torch.sparse.FloatTensor(newIdxs, newVals, adj.shape)

class Denoise(nn.Module):
    def __init__(self, in_dims, out_dims, emb_size, norm=False, dropout=0.5):
        super(Denoise, self).__init__()
        self.in_dims = in_dims
        self.out_dims = out_dims
        self.time_emb_dim = emb_size
        self.norm = norm

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
        freqs = torch.exp(-math.log(10000) * torch.arange(start=0, end=self.time_emb_dim//2, dtype=torch.float32) / (self.time_emb_dim//2)).to(x.device)
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
    def __init__(self, noise_scale, noise_min, noise_max, steps, beta_fixed=True):
        super(GaussianDiffusion, self).__init__()

        self.noise_scale = noise_scale
        self.noise_min = noise_min
        self.noise_max = noise_max
        self.steps = steps

        if noise_scale != 0:
            self.betas = torch.tensor(self.get_betas(), dtype=torch.float64)
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
        self.alphas_cumprod = torch.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), self.alphas_cumprod[:-1]])
        self.alphas_cumprod_next = torch.cat([self.alphas_cumprod[1:], torch.tensor([0.0])])

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

    def p_sample(self, model, x_start, steps, sampling_noise=False):
        if steps == 0:
            x_t = x_start
        else:
            t = torch.tensor([steps-1] * x_start.shape[0]).to(x_start.device)
            x_t = self.q_sample(x_start, t)
        
        indices = list(range(self.steps))[::-1]

        for i in indices:
            t = torch.tensor([i] * x_t.shape[0]).to(x_t.device)
            model_mean, model_log_variance = self.p_mean_variance(model, x_t, t)
            if sampling_noise:
                noise = torch.randn_like(x_t)
                nonzero_mask = ((t!=0).float().view(-1, *([1]*(len(x_t.shape)-1))))
                x_t = model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise
            else:
                x_t = model_mean
        return x_t

    def p_sample_with_trajectory(self, model, x_start, steps, sampling_noise=False):
        """Run reverse diffusion and return (x_0, trajectory (T+1, B, D)) for trajectory_linearity."""
        if steps == 0:
            trajectory = x_start.unsqueeze(0)
            return x_start, trajectory
        t = torch.tensor([steps - 1] * x_start.shape[0]).to(x_start.device)
        x_t = self.q_sample(x_start, t)
        trajectory_list = [x_t.detach().clone()]
        indices = list(range(self.steps))[::-1]
        for i in indices:
            t = torch.tensor([i] * x_t.shape[0]).to(x_t.device)
            model_mean, model_log_variance = self.p_mean_variance(model, x_t, t)
            if sampling_noise:
                noise = torch.randn_like(x_t)
                nonzero_mask = ((t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1))))
                x_t = model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise
            else:
                x_t = model_mean
            trajectory_list.append(x_t.detach().clone())
        trajectory = torch.stack(trajectory_list, dim=0)
        return x_t, trajectory

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        return self._extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start + self._extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise

    def _extract_into_tensor(self, arr, timesteps, broadcast_shape):
        arr = arr.to(timesteps.device)
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

    def training_losses(self, model, x_start, itmEmbeds, batch_index, model_feats):
        batch_size = x_start.size(0)

        ts = torch.randint(0, self.steps, (batch_size,)).long().to(x_start.device)
        noise = torch.randn_like(x_start)
        if self.noise_scale != 0:
            x_t = self.q_sample(x_start, ts, noise)
        else:
            x_t = x_start

        model_output = model(x_t, ts)

        mse = self.mean_flat((x_start - model_output) ** 2)

        weight = self.SNR(ts - 1) - self.SNR(ts)
        weight = torch.where((ts == 0), 1.0, weight)

        diff_loss = weight * mse

        usr_model_embeds = torch.mm(model_output, model_feats)
        usr_id_embeds = torch.mm(x_start, itmEmbeds)

        gc_loss = self.mean_flat((usr_model_embeds - usr_id_embeds) ** 2)

        return diff_loss, gc_loss
        
    def mean_flat(self, tensor):
        return tensor.mean(dim=list(range(1, len(tensor.shape))))
    
    def SNR(self, t):
        self.alphas_cumprod = self.alphas_cumprod.to(t.device)
        return self.alphas_cumprod[t] / (1 - self.alphas_cumprod[t])
