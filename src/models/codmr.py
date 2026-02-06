# coding: utf-8
# Desc: Core code of the CoDMR.
# Author: OrangeAI Research Team
# Time: 2026-01-03
# paper: "Collaborative Diffusion Models for Recommendation" (SIGIR2025, CoDMR)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
from common.abstract_recommender import GeneralRecommender
from utils.utils import build_knn_normalized_graph
from models.codmr_modules import gaussian_diffusioncondit as gd
from models.codmr_modules.conditdenoiser import cdenosier
from models.codmr_modules.Nonconditdenoiser import Nodenoiser

class GCN_layer(nn.Module):
    def __init__(self, hide_dim):
        super(GCN_layer, self).__init__()
        self.hide_dim = hide_dim
        self.weight = nn.Parameter(torch.FloatTensor(self.hide_dim, self.hide_dim))
        nn.init.xavier_normal_(self.weight.data)

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        if type(sparse_mx) != sp.coo_matrix:
            sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data).float()
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

    def normalize_adj(self, adj):
        """Symmetrically normalize adjacency matrix."""
        # Check if already normalized or tensor
        if isinstance(adj, torch.Tensor):
            return adj # Assuming already normalized if tensor
            
        adj = sp.coo_matrix(adj)
        rowsum = np.array(adj.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        return (d_mat_inv_sqrt).dot(adj).dot(d_mat_inv_sqrt).tocoo()    

    def forward(self, features, Mat, index):
        subset_Mat = Mat
        subset_features = features
        
        # In GenMMRec, Mat might already be a sparse tensor on device
        if isinstance(Mat, torch.Tensor) and Mat.is_sparse:
             subset_sparse_tensor = Mat
        else:
             subset_Mat = self.normalize_adj(subset_Mat)
             subset_sparse_tensor = self.sparse_mx_to_torch_sparse_tensor(subset_Mat).to(features.device)
             
        out_features = torch.spmm(subset_sparse_tensor, subset_features)
        
        # If index is provided, we update only those features? 
        # CoDMR logic: new_features[index] = out_features; others kept same.
        # But wait, out_features has shape of features.
        # If Mat is full matrix, out_features is full update.
        # CoDMR: new_features = torch.empty(features.shape)
        # new_features[index] = out_features (This implies out_features is smaller? No, out_features is spmm result)
        # Actually in CoDMR, Mat is passed as self.uiMat, which is full size.
        # index is passed as ui_index.
        # In CoDMR: out_features = spmm(Mat, features).
        # new_features[index] = out_features.
        # Wait, if Mat is full, out_features is full.
        # If index covers all nodes, then new_features == out_features.
        # In CoDMR forward, ui_index covers all users and items.
        # So essentially: return out_features.
        
        return out_features

class CoDMR(GeneralRecommender):
    def __init__(self, config, dataset):
        super(CoDMR, self).__init__(config, dataset)
        self.config = config
        
        # Parameters
        self.hide_dim = config['embedding_size'] # Mapping embedding_size to hide_dim
        self.uiLayerNums = config['uiLayers']
        self.au_uiLayerNums = config['au_uiLayers']
        self.lr = config['learning_rate']
        self.reg = config['reg_weight']
        self.ssl_temp = config['ssl_temp']
        self.steps = config['steps']
        self.sampling_steps = config['sampling_steps']
        
        # Data
        self.uiMat = dataset.inter_matrix(form='coo').astype(np.float32)
        # Make symmetric
        self.uiMat = self._build_ui_mat(self.uiMat)
        self.uiMat_tensor = self._sparse_mx_to_torch_sparse_tensor(self.normalize_adj(self.uiMat)).to(self.device)
        
        # Auxiliary Matrices (Item-Item)
        # Construct KNN graphs if not provided
        # GenMMRec uses t_feat and v_feat
        self.iciMat = self._build_knn_adj(self.t_feat) if self.t_feat is not None else self._build_identity_adj()
        self.icaiMat = self._build_knn_adj(self.v_feat) if self.v_feat is not None else self.iciMat # Fallback
        
        self.iciMat_tensor = self._sparse_mx_to_torch_sparse_tensor(self.normalize_adj(self.iciMat)).to(self.device)
        self.icaiMat_tensor = self._sparse_mx_to_torch_sparse_tensor(self.normalize_adj(self.icaiMat)).to(self.device)
        
        # Text Embeddings
        # Item Text
        if self.t_feat is not None:
             self.item_emb_text = self.t_feat.to(self.device)
             self.text_dim = self.t_feat.shape[1]
        else:
             # Random if no text
             self.text_dim = self.hide_dim
             self.item_emb_text = nn.Parameter(torch.randn(self.n_items, self.text_dim)).to(self.device)

        # User Text (CoDMR uses pre-trained, we use learnable)
        self.user_emb_text = nn.Parameter(torch.randn(self.n_users, self.text_dim)).to(self.device)
        nn.init.xavier_uniform_(self.user_emb_text)

        # Layers
        self.gcnLayers = nn.ModuleList([GCN_layer(self.hide_dim) for _ in range(4)])
        self.au_gcnLayers = nn.ModuleList([GCN_layer(self.hide_dim) for _ in range(4)])
        
        # Diffusion
        if config['mean_type'] == 'x0':
            mean_type = gd.ModelMeanType.START_X
        else:
            mean_type = gd.ModelMeanType.EPSILON
            
        self.diffusion = gd.GaussianDiffusion(
            mean_type, config['noise_schedule'], config['noise_scale'],
            config['noise_min'], config['noise_max'], config['steps'],
            self.device
        ).to(self.device)
        
        if config['mean_typeNon'] == 'x0':
             mean_type_non = gd.ModelMeanType.START_X
        else:
             mean_type_non = gd.ModelMeanType.EPSILON
             
        self.diffusionNon = gd.GaussianDiffusion(
            mean_type_non, config['noise_schedule'], config['noise_scale'],
            config['noise_min'], config['noise_max'], 6, # Fixed steps_Non=6 in CoDMR
            self.device
        ).to(self.device)
        
        # Denoisers
        # Parse dims from string "[8]" to list
        # CoDMR uses eval(args.mlp_dims). Assuming config provides list or int
        mlp_dims = config['mlp_dims']
        if isinstance(mlp_dims, int): mlp_dims = [mlp_dims]
        
        # Logic from CoDMR to determine dims
        # out_dims = eval(args.out_dims) -> [8]
        # in_dims = eval(args.in_dims)[::-1] -> [8]
        # latent_size = in_dims[-1] -> 8
        # mlp_out_dims = eval(args.mlp_dims) + [latent_size] -> [8, 8]
        # mlp_in_dims = args.hide_dim -> 32
        
        # We simplify configuration
        latent_size = config['latent_size'] # e.g. 8
        mlp_out_dims = mlp_dims + [latent_size]
        mlp_in_dims = self.hide_dim
        
        self.cdnmodel = cdenosier(mlp_in_dims, mlp_out_dims[-1], config['emb_size'], norm=config['norm'], act_func=config['mlp_act_func']).to(self.device)
        self.Nonmodel = Nodenoiser(mlp_in_dims, mlp_out_dims[-1], config['emb_size'], norm=config['norm'], act_func=config['mlp_act_func']).to(self.device)
        
        # Projections
        self.item_text_net = nn.Linear(self.text_dim, self.hide_dim, bias=False)
        self.encodecon1 = nn.Sequential(
            nn.Linear(self.hide_dim, self.hide_dim),
            nn.ReLU(True),
            nn.Linear(self.hide_dim, self.hide_dim))
        self.encodecon2 = nn.Sequential(
            nn.Linear(self.hide_dim, self.hide_dim),
            nn.ReLU(True),
            nn.Linear(self.hide_dim, self.hide_dim))
            
        # Embeddings Dictionary
        self.embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(torch.empty(self.n_users, self.hide_dim)),
            'item_emb': nn.Parameter(torch.empty(self.n_items, self.hide_dim)),
            'uinterest_emb': nn.Parameter(torch.empty(self.n_users, self.hide_dim)),
        })
        self._init_weights()
        
        # Pre-calculate norms for Loss
        # uinorm and iunorm should be based on full uiMat (N+M)
        # uiMat is (n_users + n_items, n_users + n_items)
        uiMat_coo = self.uiMat.tocoo()
        self.uinorm = torch.from_numpy(uiMat_coo.sum(1)).float().to(self.device).flatten()
        self.iunorm = torch.from_numpy(uiMat_coo.sum(0)).float().to(self.device).flatten()
        
    def _init_weights(self):
        initializer = nn.init.xavier_uniform_
        initializer(self.embedding_dict['user_emb'])
        initializer(self.embedding_dict['item_emb'])
        initializer(self.embedding_dict['uinterest_emb'])
        nn.init.xavier_uniform_(self.item_text_net.weight)

    def _build_ui_mat(self, adj):
        # Construct symmetric matrix [0, R; R.T, 0] + I
        # GenMMRec dataset.inter_matrix is (n_users, n_items)
        R = adj
        n_users, n_items = R.shape
        
        # Build symmetric
        #      0   R
        #      R.T 0
        
        R_coo = R.tocoo()
        row = np.concatenate([R_coo.row, R_coo.col + n_users])
        col = np.concatenate([R_coo.col + n_users, R_coo.row])
        data = np.concatenate([R_coo.data, R_coo.data])
        
        mat = sp.coo_matrix((data, (row, col)), shape=(n_users + n_items, n_users + n_items))
        # Add identity? CoDMR doesn't seem to add I explicitly in main.py logic, 
        # but GCN usually adds self-loop. CoDMR's normalize_adj adds it? No.
        # GCN_layer normalize_adj: d_inv_sqrt * adj * d_inv_sqrt.
        # Usually we add self-loop. Let's add it.
        mat = mat + sp.eye(mat.shape[0])
        return mat

    def _build_knn_adj(self, feature):
        # feature: tensor
        feature = F.normalize(feature, p=2, dim=-1)
        sim_adj = torch.mm(feature, feature.transpose(1, 0))
        # Use utils to sparsify
        sim_adj_sparse = build_knn_normalized_graph(sim_adj, topk=self.config['knn_k'], is_sparse=True, norm_type='sym')
        # Convert back to scipy coo for consistency with CoDMR logic which expects scipy in init (though we converted to tensor)
        # Actually I stored them as tensors directly.
        # But GCN_layer expects tensor.
        return sim_adj_sparse # This returns torch sparse tensor. 
    
    def _build_identity_adj(self):
        # Return identity sparse tensor
        i = torch.arange(self.n_items)
        indices = torch.stack([i, i])
        values = torch.ones(self.n_items)
        return torch.sparse.FloatTensor(indices, values, (self.n_items, self.n_items))

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        # Helper for non-GCN usages if any
        return self._sparse_mx_to_torch_sparse_tensor(sparse_mx)

    def _sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        if isinstance(sparse_mx, torch.Tensor):
            return sparse_mx
        if type(sparse_mx) != sp.coo_matrix:
            sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data).float()
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)
        
    def normalize_adj(self, adj):
        if isinstance(adj, torch.Tensor):
            return adj
        adj = sp.coo_matrix(adj)
        rowsum = np.array(adj.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        return (d_mat_inv_sqrt).dot(adj).dot(d_mat_inv_sqrt).tocoo() 

    def ssl_loss(self, data1, data2, index, mask):
        # data1/2 shape: [Batch_Size, Dim] (Assuming data1/2 are already filtered by index if passed as such)
        # But wait, in calculate_loss, we pass `diffuitem_batch` which is [ItemIndexSize, Dim].
        # And `index` is `itemindex` [ItemIndexSize].
        # So we don't need to re-index data1/2 with index if they are already corresponding to index.
        
        # Let's check calculate_loss call:
        # self.ssl_loss(diffuitem_batch, diffuitem1_batch, itemindex, None)
        # diffuitem_batch is returned by forward, corresponding to itemindex.
        # So diffuitem_batch already has size len(itemindex).
        
        # If I do `embeddings1 = data1[index]`, I am indexing into data1 using item IDs.
        # But data1 is already the subset for these items! It likely has shape [len(index), dim].
        # The indices in `index` are global item IDs (e.g. 3429).
        # data1 has indices 0..len(index)-1.
        # So `data1[index]` will fail if index contains values > len(data1).
        
        # Correction:
        # data1 and data2 are already aligned with index. We just need to use them directly.
        # OR, if data1 was the FULL embedding matrix, we would index it.
        # But forward() returns batch-specific embeddings.
        
        embeddings1 = data1
        embeddings2 = data2
        
        norm_embeddings1 = F.normalize(embeddings1, p=2, dim=1)
        norm_embeddings2 = F.normalize(embeddings2, p=2, dim=1)
        
        # InfoNCE
        # Positive pairs are (i, i) in the batch
        pos_score = torch.sum(torch.mul(norm_embeddings1, norm_embeddings2), dim=1)
        pos_score = torch.exp(pos_score / self.ssl_temp)
        
        # Negative pairs: all other items in the batch
        all_score = torch.mm(norm_embeddings1, norm_embeddings2.T)
        all_score = torch.sum(torch.exp(all_score / self.ssl_temp), dim=1)
        
        ssl_loss = (-torch.sum(torch.log(pos_score / all_score)) / len(embeddings1))
        return ssl_loss

    def forward(self, iftraining, user, itemi, itemj, norm=0):
        # user, itemi, itemj are batch indices
        item_index = torch.arange(0, self.n_items).to(self.device)
        user_index = torch.arange(0, self.n_users).to(self.device)
        
        ui_userembed = self.embedding_dict['user_emb']
        ui_itemembed = self.embedding_dict['item_emb']
        user_interested = self.embedding_dict['uinterest_emb']
        
        # Target Domain Encoding
        ui_index = torch.cat([user_index, item_index + self.n_users]) # Indices in big graph
        self.ui_embeddings = torch.cat([ui_userembed, ui_itemembed], 0)
        self.all_ui_embeddings = [self.ui_embeddings]
        
        uiEmbeddings0 = self.ui_embeddings
        for i in range(self.uiLayerNums):
            layer = self.gcnLayers[i]
            uiEmbeddings0 = layer(uiEmbeddings0, self.uiMat_tensor, ui_index)
            if norm == 1:
                norm_embeddings = F.normalize(uiEmbeddings0, p=2, dim=1)
                self.all_ui_embeddings.append(norm_embeddings)
                
        self.uiEmbedding = torch.stack(self.all_ui_embeddings, dim=1)
        self.uiEmbedding = torch.mean(self.uiEmbedding, dim=1)
        self.ui_userEmbedding, self.ui_itemEmbedding = torch.split(self.uiEmbedding, [self.n_users, self.n_items])
        
        # Auxiliary Domain Encoding
        item_embed0 = ui_itemembed
        item_embed1 = ui_itemembed
        
        conditiembed_i = self.item_text_net(self.item_emb_text)
        conditiembed_u = self.item_text_net(self.user_emb_text)
        self.ui_embeddings_text = torch.cat([conditiembed_u, conditiembed_i], 0)
        
        self.all_item_embeddings0 = [item_embed0]
        self.all_item_embeddings1 = [item_embed1]
        self.all_ui_embeddings_tx = [self.ui_embeddings_text]
        
        uiEmbeddings_tx = self.ui_embeddings_text
        itemEmbeddings0 = item_embed0
        itemEmbeddings1 = item_embed1
        
        for i in range(self.au_uiLayerNums):
            layer = self.au_gcnLayers[i]
            # In CoDMR, it passes Mat and index.
            # Mat is iciMat_tensor / icaiMat_tensor
            uiEmbeddings_tx = layer(uiEmbeddings_tx, self.uiMat_tensor, ui_index)
            itemEmbeddings0 = layer(itemEmbeddings0, self.iciMat_tensor, item_index)
            itemEmbeddings1 = layer(itemEmbeddings1, self.icaiMat_tensor, item_index)
            
            if norm == 1:
                self.all_ui_embeddings_tx.append(F.normalize(uiEmbeddings_tx, p=2, dim=1))
                self.all_item_embeddings0.append(F.normalize(itemEmbeddings0, p=2, dim=1))
                self.all_item_embeddings1.append(F.normalize(itemEmbeddings1, p=2, dim=1))
        
        self.uiEmbedding_tx = torch.mean(torch.stack(self.all_ui_embeddings_tx, dim=1), dim=1)
        self.ui_userEmbedding_tx, self.ui_itemEmbedding_tx = torch.split(self.uiEmbedding_tx, [self.n_users, self.n_items])
        
        self.itemEmbedding0 = torch.mean(torch.stack(self.all_item_embeddings0, dim=1), dim=1)
        self.itemEmbedding1 = torch.mean(torch.stack(self.all_item_embeddings1, dim=1), dim=1)
        
        # Diffusion Logic
        elboii = elboNonii = elbo_txi = elbo_txu = elboNon_txi = elboNon_txu = 0
        mse = 0
        
        reuseredtx = reitemedtx = reitemedii = None # For inference return
        
        if iftraining:
            # Prepare batch latent vars (placeholders in CoDMR, but we need to compute losses)
            # We only compute losses for unique batch indices to save time
            userindex = torch.unique(user)
            itemindex = torch.unique(torch.cat((itemi, itemj)))
            
            # Extract conditional features
            conditionembed_ui2 = self.encodecon2(self.ui_userEmbedding.detach())
            conditionembed_ui = self.encodecon1(self.ui_itemEmbedding.detach())
            
            conditionembed = conditionembed_ui[itemindex]
            
            # Aux features to be denoised
            startembed = (self.itemEmbedding0 + self.itemEmbedding1) / 2.0
            startembed = startembed[itemindex]
            
            # Stage 1: Unconditional denoising item-item
            terms = self.diffusion.training_losses(self.Nonmodel, iftraining, (startembed.detach(), conditionembed), self.config['reweight'])
            elboNonii = terms["loss"].mean()
            batch_latent_reconNonindex = terms["pred_xstart"]
            
            # Stage 2: Conditional
            batch_latent_reconNonindex = (batch_latent_reconNonindex.detach() + startembed.detach()) / 2.0
            terms = self.diffusion.training_losses(self.cdnmodel, iftraining, (batch_latent_reconNonindex, conditionembed_ui[itemindex].detach()), self.config['reweight'])
            elboii = terms["loss"].mean()
            batch_latent_recon_ii = terms["pred_xstart"]
            
            # Textual Domain Diffusion
            # Stage 1: Unconditional
            # Item
            terms1 = self.diffusion.training_losses(self.Nonmodel, iftraining, (self.ui_itemEmbedding_tx[itemindex].detach(), conditionembed), self.config['reweight'])
            elboNon_txi = terms1["loss"].mean()
            batch_latent_reconiNonindex = terms1["pred_xstart"]
            
            # User
            terms1 = self.diffusion.training_losses(self.Nonmodel, iftraining, (self.ui_userEmbedding_tx[userindex].detach(), conditionembed_ui2[userindex]), self.config['reweight']) # Use conditionembed_ui2 for user? CoDMR uses conditionembed (from items) for user?
            # CoDMR: terms1 = self.diffusion.training_losses(self.Nonmodel,iftraining,(self.ui_userEmbedding_tx[userindex].detach(),  conditionembed), ...)
            # Wait, conditionembed is from items (size of itemindex). userindex might have different size!
            # CoDMR line 302: (..., conditionembed).
            # This looks like a bug in CoDMR if sizes mismatch.
            # But maybe userindex and itemindex happen to be same size or broadcast? Unlikely.
            # Ah, conditionembed in CoDMR line 262: conditionembed = conditionembed_ui[itemindex].
            # line 302 uses it with user features.
            # Unless itemindex and userindex are same length, this fails.
            # Let's look at CoDMR line 323 (Conditional Stage for User):
            # terms = self.diffusion.training_losses(..., (..., conditionembed_ui2[userindex].detach()), ...)
            # Here it uses conditionembed_ui2[userindex].
            # I suspect line 302 should use conditionembed_ui2[userindex] (or similar user-derived condition).
            # I will assume `conditionembed_ui2[userindex]` is correct for User unconditional stage too, or a corresponding user condition.
            # CoDMR line 259: conditionembed_ui2 = self.encodecon2(self.ui_userEmbedding.detach())
            # I'll use `conditionembed_ui2[userindex]` for user diffusion condition to be safe and logical.
            
            elboNon_txu = terms1["loss"].mean()
            batch_latent_reconuNonindex = terms1["pred_xstart"] # Placeholder, logical correction
            
            # Stage 2: Conditional
            # Item
            batch_latent_reconiNonindex = (batch_latent_reconiNonindex.detach() + self.ui_itemEmbedding_tx[itemindex].detach()) / 2.0
            terms = self.diffusion.training_losses(self.cdnmodel, iftraining, (batch_latent_reconiNonindex, conditionembed_ui[itemindex].detach()), self.config['reweight'])
            elbo_txi = terms["loss"].mean()
            batch_latent_reconi = terms["pred_xstart"]
            
            # User
            batch_latent_reconuNonindex = (batch_latent_reconuNonindex.detach() + self.ui_userEmbedding_tx[userindex].detach()) / 2.0
            terms = self.diffusion.training_losses(self.cdnmodel, iftraining, (batch_latent_reconuNonindex, conditionembed_ui2[userindex].detach()), self.config['reweight'])
            elbo_txu = terms["loss"].mean()
            batch_latent_reconu = terms["pred_xstart"]
            
            # MSE Structure Similarity
            # userembed = spmm(uiadj, conditiembed1) / uinorm
            # conditiembed1 is concatenation of User and Item conditional embeddings
            
            full_cond_embed = torch.cat([conditionembed_ui2, conditionembed_ui], 0)
            
            # User side (Actually Full Graph Recon)
            userembed_recon = torch.spmm(self.uiMat_tensor, full_cond_embed) / (self.uinorm.unsqueeze(1) + 1e-8)
            mse1 = ((userembed_recon - self.uiEmbedding)**2).sum(1)
            
            # Item side (Same if symmetric, but following CoDMR logic)
            itemembed_recon = torch.spmm(self.uiMat_tensor.t(), full_cond_embed) / (self.iunorm.unsqueeze(1) + 1e-8)
            mse2 = ((itemembed_recon - self.uiEmbedding)**2).sum(1)
            
            mse = mse1.mean() + mse2.mean()
            
            # Returns for Loss calculation in calculate_loss
            # (elboloss, mse, (diffuser, diffuitem), (preuser_ii, diffuitem1))
            # diffuser, diffuitem are from Conditional stage output (reconstructed features)
            # We need them for "Construction of prediction loss for denoised..."
            # CoDMR: pred_posx, pred_negx = predictModel(diffuser[user], diffuitem[item_i], ...)
            # So we need the RECONSTRUCTED features for the batch.
            # batch_latent_reconidx (from Stage 2) is the reconstructed feature for the batch.
            
            # Item Recon (from Text)
            # batch_latent_reconi (Stage 2 Item Text) -> diffuitem (part of it)
            # Item Recon (from II)
            # batch_latent_recon_ii (Stage 2 Item II) -> diffuitem1 (part of it)
            
            # User Recon (from Text)
            # batch_latent_reconu (Stage 2 User Text) -> diffuser
            
            # CoDMR combines them:
            # reitemedtx = batch_latent_reconi * 0.5 + batch_latent_reconiNon * 0.5
            # ...
            # Actually, CoDMR returns (reuseredtx, reitemedtx) and (user_interested, reitemedii) in tuple.
            # And then calculates BPR on them.
            
            # We need to construct these reconstructed features for the batch indices.
            
            # User Reconstructed (Text)
            # We have batch_latent_reconu (Stage 2) and batch_latent_reconuNonindex (Stage 1 inputs to Stage 2)
            # Wait, batch_latent_reconu is output of Stage 2 (p_sample or training_losses returns pred_xstart).
            # CoDMR logic in training_losses returns pred_xstart.
            # So `batch_latent_reconu` IS the reconstructed x0.
            
            diffuser_batch = batch_latent_reconu # Reconstructed User (Text)
            diffuitem_batch = batch_latent_reconi # Reconstructed Item (Text)
            diffuitem1_batch = batch_latent_recon_ii # Reconstructed Item (II)
            
            # We return these for use in calculate_loss
            
            return elboii, elboNonii, elbo_txi, elbo_txu, elboNon_txi, elboNon_txu, mse, \
                   diffuser_batch, diffuitem_batch, diffuitem1_batch, \
                   userindex, itemindex
                   
        else:
            # Inference / Testing
            # Perform full generation
            startembed = (self.itemEmbedding0 + self.itemEmbedding1) / 2.0
            conditionembed_ui = self.encodecon1(self.ui_itemEmbedding.detach())
            conditionembed_ui2 = self.encodecon2(self.ui_userEmbedding.detach())
            
            # Stage 1
            batch_latent_reconNon_ii = self.diffusion.p_sample(self.Nonmodel, iftraining, (startembed, conditionembed_ui), self.sampling_steps)
            batch_latent_reconiNon = self.diffusion.p_sample(self.Nonmodel, iftraining, (self.ui_itemEmbedding_tx, conditionembed_ui), self.sampling_steps)
            batch_latent_reconuNon = self.diffusion.p_sample(self.Nonmodel, iftraining, (self.ui_userEmbedding_tx, conditionembed_ui2), self.sampling_steps)
            
            # Stage 2
            batch_latent_reconNon_ii = (batch_latent_reconNon_ii + startembed) / 2.0
            batch_latent_recon_ii = self.diffusion.p_sample(self.cdnmodel, iftraining, (batch_latent_reconNon_ii, conditionembed_ui), self.sampling_steps)
            
            batch_latent_reconiNon = (batch_latent_reconiNon + self.ui_itemEmbedding_tx) / 2.0
            batch_latent_reconi = self.diffusion.p_sample(self.cdnmodel, iftraining, (batch_latent_reconiNon, conditionembed_ui), self.sampling_steps)
            
            batch_latent_reconuNon = (batch_latent_reconuNon + self.ui_userEmbedding_tx) / 2.0
            batch_latent_reconu = self.diffusion.p_sample(self.cdnmodel, iftraining, (batch_latent_reconuNon, conditionembed_ui2), self.sampling_steps)
            
            # Combine
            reitemedtx = batch_latent_reconi * 0.5 + batch_latent_reconiNon * 0.5
            reuseredtx = batch_latent_reconu * 0.5 + batch_latent_reconuNon * 0.5
            reitemedii = batch_latent_recon_ii * 0.5 + batch_latent_reconNon_ii * 0.5
            
            recouserembed = (user_interested + reuseredtx) / 2.0
            recoitemembed = (reitemedtx + reitemedii) / 2.0
            
            return recouserembed, recoitemembed

    def calculate_loss(self, interaction):
        user = interaction[0]
        pos_item = interaction[1]
        neg_item = interaction[2]
        
        # Forward Training
        elboii, elboNonii, elbo_txi, elbo_txu, elboNon_txi, elboNon_txu, mse, \
        diffuser_batch, diffuitem_batch, diffuitem1_batch, \
        userindex, itemindex = self.forward(True, user, pos_item, neg_item, norm=1)
        
        # Embeddings for BPR
        u_emb = self.ui_userEmbedding[user]
        p_emb = self.ui_itemEmbedding[pos_item]
        n_emb = self.ui_itemEmbedding[neg_item]
        
        # 1. BPR Loss
        pos_scores = torch.sum(u_emb * p_emb, dim=1)
        neg_scores = torch.sum(u_emb * n_emb, dim=1)
        bpr_loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))
        
        # 2. Reg Loss
        reg_loss = (u_emb.norm(2).pow(2) + p_emb.norm(2).pow(2) + n_emb.norm(2).pow(2)) / len(user)
        
        # 3. Diffusion/Reconstruction BPR Loss
        # We need to map batch users/items to their indices in unique list (userindex, itemindex)
        # to retrieve reconstructed features.
        # But wait, forward returned diffuser_batch corresponding to userindex.
        # We need to gather them back for (user, pos, neg) triplets.
        
        # Map global ID to local index in userindex/itemindex
        # This is slow. Better if forward returned full reconstructed matrix or we pass batch directly.
        # In my forward implementation, I returned `diffuser_batch` which corresponds to `userindex`.
        # I need to map `user` to `diffuser_batch`.
        
        # Optimization: Pass the full batch (user, pos, neg) to forward, and let forward return predictions directly?
        # Or construct a mapping.
        
        # Let's do a mapping
        def get_emb(indices, unique_indices, embs):
            # indices: [batch]
            # unique_indices: [U]
            # embs: [U, dim]
            # We want [batch, dim]
            # Use searchsorted or similar?
            # Or just use the fact that we computed embs for unique_indices.
            # Maybe easier:
            # Create a temporary full tensor (sparse update)? No.
            # Use dictionary? Slow.
            # Re-index:
            # map global id -> 0..N
            # But unique_indices is sorted.
            # torch.bucketize can help.
            pos = torch.bucketize(indices, unique_indices)
            return embs[pos]

        # Ensure sorted for bucketize
        # userindex is unique(user), so sorted.
        diffuser_u = get_emb(user, userindex, diffuser_batch)
        diffuitem_p = get_emb(pos_item, itemindex, diffuitem_batch)
        diffuitem_n = get_emb(neg_item, itemindex, diffuitem_batch)
        
        diffuitem1_p = get_emb(pos_item, itemindex, diffuitem1_batch)
        diffuitem1_n = get_emb(neg_item, itemindex, diffuitem1_batch)
        
        preuser_ii = self.embedding_dict['uinterest_emb'][user] # CoDMR uses this for second diff loss
        
        # Diff Loss 1 (Text Recon)
        pos_scores_diff = torch.sum(diffuser_u * diffuitem_p, dim=1)
        neg_scores_diff = torch.sum(diffuser_u * diffuitem_n, dim=1)
        bpr_loss_diff = -torch.mean(F.logsigmoid(pos_scores_diff - neg_scores_diff))
        reg_loss_diff = (diffuser_u.norm(2).pow(2) + diffuitem_p.norm(2).pow(2) + diffuitem_n.norm(2).pow(2)) / len(user)
        loss_diff1 = 0.95 * (bpr_loss_diff + reg_loss_diff * self.reg)
        
        # Diff Loss 2 (II Recon)
        pos_scores_diff2 = torch.sum(preuser_ii * diffuitem1_p, dim=1)
        neg_scores_diff2 = torch.sum(preuser_ii * diffuitem1_n, dim=1)
        bpr_loss_diff2 = -torch.mean(F.logsigmoid(pos_scores_diff2 - neg_scores_diff2))
        reg_loss_diff2 = (preuser_ii.norm(2).pow(2) + diffuitem1_p.norm(2).pow(2) + diffuitem1_n.norm(2).pow(2)) / len(user)
        loss_diff2 = 0.95 * (bpr_loss_diff2 + reg_loss_diff2 * self.reg)
        
        loss_diff = (loss_diff1 + loss_diff2) / 2.0
        
        # 4. ELBO Loss
        elboloss = (elbo_txi + elboNon_txi + elbo_txu + elboNon_txu) + (elboii + elboNonii)
        
        # 5. Contrastive Loss (SSL)
        # item1, item2 from forward?
        # CoDMR: item1 = reitemedtx, item2 = reitemedii (reconstructed features)
        # We need to reconstruct them fully or for the batch.
        # In forward, we have diffuitem_batch (Text) and diffuitem1_batch (II).
        # And user versions.
        # ssloss = (ssl_loss(item1, item2) + ssl_loss(user1, user2)) / 2
        
        # We need to map preuser_ii to userindex
        preuser_ii_unique = self.embedding_dict['uinterest_emb'][userindex]

        ssloss = (self.ssl_loss(diffuitem_batch, diffuitem1_batch, itemindex, None) + \
                  self.ssl_loss(diffuser_batch, preuser_ii_unique, userindex, None)) / 2.0
                  
        # Total Loss
        # Weights from config
        total_loss = 0.95 * (bpr_loss + reg_loss * self.reg) + \
                     elboloss * self.config['elbo_w'] + \
                     loss_diff * self.config['di_pre_w'] + \
                     mse * self.config['con_fe_w'] + \
                     ssloss * self.config['ssl_reg']
                     
        return total_loss

    def full_sort_predict(self, interaction):
        user = interaction[0]
        
        # Get reconstructed embeddings (Inference)
        # We cache them to avoid re-computing for every batch in evaluation
        if not hasattr(self, 'cached_preds'):
            self.cached_user_emb, self.cached_item_emb = self.forward(False, None, None, None, norm=1)
            self.cached_preds = True
            
        u_emb = self.cached_user_emb[user]
        i_emb = self.cached_item_emb
        
        scores = torch.matmul(u_emb, i_emb.t())
        return scores
        
    def train(self, mode=True):
        super().train(mode)
        if mode and hasattr(self, 'cached_preds'):
            del self.cached_preds
            del self.cached_user_emb
            del self.cached_item_emb
