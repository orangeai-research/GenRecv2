# coding: utf-8
# Desc: MVDiff model migrated from MVDiff folder
# Multi-View Diffusion for multimodal recommender system

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import scipy.sparse as sp
from scipy.sparse import coo_matrix
from common.abstract_recommender import GeneralRecommender
from utils.utils import build_knn_normalized_graph

init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform


class MVDiff(GeneralRecommender):
    """
    Multi-View Diffusion for multimodal recommender system
    """
    def __init__(self, config, dataset):
        super(MVDiff, self).__init__(config, dataset)
        
        # Config parameters
        self.latdim = config['embedding_size']
        self.gcn_layer_num = config['gcn_layer_num'] if 'gcn_layer_num' in config else 1
        self.keepRate = config['keep_rate'] if 'keep_rate' in config else 0.5
        self.reg_weight = config['reg_weight'] if 'reg_weight' in config else 1e-5
        self.modal_fusion = config['modal_fusion'] if 'modal_fusion' in config else True
        self.ssl_reg = config['ssl_reg'] if 'ssl_reg' in config else 1e-2
        self.temp = config['temperature'] if 'temperature' in config else 0.7
        self.sparse_temp = config['sparse_temp'] if 'sparse_temp' in config else 0.2
        
        # Diffusion parameters
        self.noise_scale = config['noise_scale'] if 'noise_scale' in config else 0.1
        self.noise_min = config['noise_min'] if 'noise_min' in config else 0.0001
        self.noise_max = config['noise_max'] if 'noise_max' in config else 0.02
        self.steps = config['steps'] if 'steps' in config else 5
        self.d_emb_size = config['d_emb_size'] if 'd_emb_size' in config else 10
        self.norm = config['norm'] if 'norm' in config else False
        self.sampling_steps = config['sampling_steps'] if 'sampling_steps' in config else 0
        self.sampling_noise = config['sampling_noise'] if 'sampling_noise' in config else False
        self.rebuild_k = config['rebuild_k'] if 'rebuild_k' in config else 10
        self.high_order_topk = config['high_order_topk'] if 'high_order_topk' in config else 2
        self.e_loss = config['e_loss'] if 'e_loss' in config else 0.1
        self.alpha_sparity = config['alpha_sparity'] if 'alpha_sparity' in config else 0.01
        self.beta_sparity = config['beta_sparity'] if 'beta_sparity' in config else 0.01
        self.postive_gain_degree = config['postive_gain_degree'] if 'postive_gain_degree' in config else 0.9
        self.knn_k = config['knn_k'] if 'knn_k' in config else 5
        
        # Audio modality support
        self.audio_modality = config['audio_modality'] if 'audio_modality' in config else False
        
        # Load features
        self.image_embedding = self.v_feat if self.v_feat is not None else None
        self.text_embedding = self.t_feat if self.t_feat is not None else None
        
        # Try to load audio features if available
        if self.audio_modality:
            dataset_path = os.path.abspath(config['data_path'] + config['dataset'])
            audio_feat_file = config['audio_feature_file'] if 'audio_feature_file' in config else 'audio_feat.npy'
            audio_feat_file_path = os.path.join(dataset_path, audio_feat_file)
            if os.path.isfile(audio_feat_file_path):
                self.audio_embedding = torch.from_numpy(np.load(audio_feat_file_path, allow_pickle=True)).type(torch.FloatTensor).to(self.device)
            else:
                self.audio_embedding = None
                self.audio_modality = False
        else:
            self.audio_embedding = None
        
        # Pre-calculate graph
        self.norm_adj = self.get_norm_adj_mat(dataset.inter_matrix(form='coo').astype(np.float32)).to(self.device)
        self.R = self._get_user_item_matrix(dataset.inter_matrix(form='coo').astype(np.float32)).to(self.device)
        
        # Initialize model components
        self._init_embeddings()
        self._init_modal_projections()
        self._init_gates()
        self._init_diffusion_models()
        
        # Generated matrices (will be set during training)
        self.image_UI_matrix = None
        self.text_UI_matrix = None
        self.audio_UI_matrix = None
        self.image_II_matrix = None
        self.text_II_matrix = None
        self.audio_II_matrix = None
        self.modal_fusion_II_matrix = None
        
        self.edgeDropper = SpAdjDropEdge(self.keepRate)
    
    def _init_embeddings(self):
        """Initialize user and item embeddings"""
        self.user_embedding = nn.Embedding(self.n_users, self.latdim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.latdim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)
    
    def _init_modal_projections(self):
        """Initialize modal feature projection layers"""
        if self.image_embedding is not None:
            image_dim = self.image_embedding.shape[1]
            self.image_residual_project = nn.Sequential(
                nn.Linear(in_features=image_dim, out_features=image_dim//4),
                nn.BatchNorm1d(image_dim//4),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(in_features=image_dim//4, out_features=image_dim//8),
                nn.BatchNorm1d(image_dim//8),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(in_features=image_dim//8, out_features=self.latdim),
                nn.BatchNorm1d(self.latdim),
                nn.ReLU(),
                nn.Dropout(0.1),
            )
            self.image_modal_project = nn.Sequential(
                nn.Linear(in_features=self.latdim, out_features=self.latdim),
                nn.BatchNorm1d(self.latdim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(in_features=self.latdim, out_features=self.latdim),
                nn.BatchNorm1d(self.latdim),
                nn.ReLU(),
                nn.Dropout(0.1)
            )
            self._init_weights(self.image_residual_project)
            self._init_weights(self.image_modal_project)
        
        if self.text_embedding is not None:
            text_dim = self.text_embedding.shape[1]
            self.text_residual_project = nn.Sequential(
                nn.Linear(in_features=text_dim, out_features=text_dim//4),
                nn.BatchNorm1d(text_dim//4),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(in_features=text_dim//4, out_features=text_dim//8),
                nn.BatchNorm1d(text_dim//8),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(in_features=text_dim//8, out_features=self.latdim),
                nn.BatchNorm1d(self.latdim),
                nn.ReLU(),
                nn.Dropout(0.1),
            )
            self.text_modal_project = nn.Sequential(
                nn.Linear(in_features=self.latdim, out_features=self.latdim),
                nn.BatchNorm1d(self.latdim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(in_features=self.latdim, out_features=self.latdim),
                nn.BatchNorm1d(self.latdim),
                nn.ReLU(),
                nn.Dropout(0.1)
            )
            self._init_weights(self.text_residual_project)
            self._init_weights(self.text_modal_project)
        
        if self.audio_embedding is not None:
            audio_dim = self.audio_embedding.shape[1]
            self.audio_residual_project = nn.Sequential(
                nn.Linear(in_features=audio_dim, out_features=self.latdim),
                nn.BatchNorm1d(self.latdim),
                nn.LeakyReLU(),
                nn.Dropout(0.1),
                nn.Linear(in_features=self.latdim, out_features=self.latdim),
                nn.BatchNorm1d(self.latdim),
                nn.LeakyReLU(),
                nn.Dropout(0.1)
            )
            self.audio_modal_project = nn.Sequential(
                nn.Linear(in_features=self.latdim, out_features=self.latdim),
                nn.BatchNorm1d(self.latdim),
                nn.LeakyReLU(),
                nn.Dropout(0.1),
                nn.Linear(in_features=self.latdim, out_features=self.latdim),
                nn.BatchNorm1d(self.latdim),
                nn.LeakyReLU(),
                nn.Dropout(0.1)
            )
            self._init_weights(self.audio_residual_project)
            self._init_weights(self.audio_modal_project)
    
    def _init_weights(self, module):
        """Initialize weights for a module"""
        for layer in module:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
    
    def _init_gates(self):
        """Initialize gate networks"""
        self.softmax = nn.Softmax(dim=-1)
        
        self.gate_image_modal = nn.Sequential(
            nn.Linear(self.latdim, self.latdim),
            nn.BatchNorm1d(self.latdim),
            nn.Sigmoid()
        )
        self.gate_text_modal = nn.Sequential(
            nn.Linear(self.latdim, self.latdim),
            nn.BatchNorm1d(self.latdim),
            nn.Sigmoid()
        )
        if self.audio_modality:
            self.gate_audio_modal = nn.Sequential(
                nn.Linear(self.latdim, self.latdim),
                nn.BatchNorm1d(self.latdim),
                nn.Sigmoid()
            )
        
        self.caculate_common = nn.Sequential(
            nn.Linear(self.latdim, self.latdim),
            nn.BatchNorm1d(self.latdim),
            nn.Tanh(),
            nn.Linear(self.latdim, 1, bias=False)
        )
        
        self._init_weights(self.gate_image_modal)
        self._init_weights(self.gate_text_modal)
        if self.audio_modality:
            self._init_weights(self.gate_audio_modal)
        self._init_weights(self.caculate_common)
    
    def _init_diffusion_models(self):
        """Initialize diffusion models"""
        # Gaussian diffusion for multimodal features
        self.diffusion_model = GaussianDiffusion(
            self.noise_scale, self.noise_min, self.noise_max, self.steps
        ).to(self.device)
        
        # Sparsity diffusion for user-item interactions
        self.sparity_diffusion_model = SparityDiffusion(
            self.noise_scale, self.noise_min, self.noise_max, self.steps,
            self.alpha_sparity, self.beta_sparity, self.postive_gain_degree, self.sparse_temp
        ).to(self.device)
        
        # Denoise models for user-item interactions
        item_num = self.n_items
        self.denoise_model_image = ModalDenoise(item_num, item_num, self.d_emb_size, norm=self.norm).to(self.device)
        self.denoise_model_text = ModalDenoise(item_num, item_num, self.d_emb_size, norm=self.norm).to(self.device)
        if self.audio_modality:
            self.denoise_model_audio = ModalDenoise(item_num, item_num, self.d_emb_size, norm=self.norm).to(self.device)
        
        # Multimodal denoise models for features
        image_dim = self.image_embedding.shape[1] if self.image_embedding is not None else 0
        text_dim = self.text_embedding.shape[1] if self.text_embedding is not None else 0
        audio_dim = self.audio_embedding.shape[1] if self.audio_embedding is not None else 0
        
        # Ensure at least image or text exists
        assert image_dim > 0 or text_dim > 0, "At least image or text features must be provided"
        
        if image_dim > 0:
            self.image_modal_denoise_model = MultimodalDenoiseModel(
                image_in_dims=image_dim,
                text_in_dims=text_dim,
                audio_in_dims=audio_dim,
                out_dims=image_dim,
                time_emb_size=self.d_emb_size,
                modal_flag='image',
                audio_modality=self.audio_modality
            ).to(self.device)
        else:
            self.image_modal_denoise_model = None
        
        if text_dim > 0:
            self.text_modal_denoise_model = MultimodalDenoiseModel(
                image_in_dims=image_dim,
                text_in_dims=text_dim,
                audio_in_dims=audio_dim,
                out_dims=text_dim,
                time_emb_size=self.d_emb_size,
                modal_flag='text',
                audio_modality=self.audio_modality
            ).to(self.device)
        else:
            self.text_modal_denoise_model = None
        
        if self.audio_modality and audio_dim > 0:
            self.audio_modal_denoise_model = MultimodalDenoiseModel(
                image_in_dims=image_dim,
                text_in_dims=text_dim,
                audio_in_dims=audio_dim,
                out_dims=audio_dim,
                time_emb_size=self.d_emb_size,
                modal_flag='audio',
                audio_modality=self.audio_modality
            ).to(self.device)
        else:
            self.audio_modal_denoise_model = None
    
    def get_norm_adj_mat(self, interaction_matrix):
        """Build normalized adjacency matrix"""
        A = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        inter_M = interaction_matrix
        inter_M_t = interaction_matrix.transpose()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.n_users), [1] * inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row + self.n_users, inter_M_t.col), [1] * inter_M_t.nnz)))
        for key, value in data_dict.items():
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
    
    def _get_user_item_matrix(self, interaction_matrix):
        """Build user-item sparse matrix"""
        # Ensure it's in COO format
        if not isinstance(interaction_matrix, sp.coo_matrix):
            mat = interaction_matrix.tocoo()
        else:
            mat = interaction_matrix
        idxs = torch.from_numpy(np.vstack([mat.row, mat.col]).astype(np.int64))
        vals = torch.from_numpy(mat.data.astype(np.float32))
        shape = torch.Size(mat.shape)
        return torch.sparse.FloatTensor(idxs, vals, shape)
    
    def getItemEmbeds(self):
        return self.item_id_embedding.weight
    
    def getUserEmbeds(self):
        return self.user_embedding.weight
    
    def getImageFeats(self):
        if self.image_embedding is not None:
            x = self.image_residual_project(self.image_embedding)
            image_modal_feature = self.image_modal_project(x)
            image_modal_feature += x
            return image_modal_feature
        return None
    
    def getTextFeats(self):
        if self.text_embedding is not None:
            x = self.text_residual_project(self.text_embedding)
            text_modal_feature = self.text_modal_project(x)
            text_modal_feature += x
            return text_modal_feature
        return None
    
    def getAudioFeats(self):
        if self.audio_embedding is not None:
            x = self.audio_residual_project(self.audio_embedding)
            audio_modal_feature = self.audio_modal_project(x)
            audio_modal_feature += x
            return audio_modal_feature
        return None
    
    def user_item_GCN(self, adj):
        """User-Item GCN"""
        cat_embedding = torch.cat([self.user_embedding.weight, self.item_id_embedding.weight], dim=0)
        all_embeddings = [cat_embedding]
        for i in range(self.gcn_layer_num):
            temp_embeddings = torch.sparse.mm(adj, cat_embedding)
            cat_embedding = temp_embeddings
            all_embeddings += [cat_embedding]
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
        return all_embeddings
    
    def item_item_GCN(self, R, original_ui_adj, diffusion_ii_image_adj, diffusion_ii_text_adj, diffusion_ii_audio_adj=None):
        """Item-Item GCN"""
        image_modal_feature = self.getImageFeats()
        image_item_id_embedding = torch.multiply(self.item_id_embedding.weight, self.gate_image_modal(image_modal_feature))
        
        text_modal_feature = self.getTextFeats()
        text_item_id_embedding = torch.multiply(self.item_id_embedding.weight, self.gate_text_modal(text_modal_feature))
        
        if self.audio_modality:
            audio_modal_feature = self.getAudioFeats()
            audio_item_id_embedding = torch.multiply(self.item_id_embedding.weight, self.gate_audio_modal(audio_modal_feature))
        
        # Image modality
        for _ in range(self.gcn_layer_num):
            image_item_id_embedding = torch.sparse.mm(diffusion_ii_image_adj, image_item_id_embedding)
        image_user_embedding = torch.sparse.mm(R, image_item_id_embedding)
        image_ui_embedding = torch.cat([image_user_embedding, image_item_id_embedding], dim=0)
        
        # Text modality
        for _ in range(self.gcn_layer_num):
            text_item_id_embedding = torch.sparse.mm(diffusion_ii_text_adj, text_item_id_embedding)
        text_user_embedding = torch.sparse.mm(R, text_item_id_embedding)
        text_ui_embedding = torch.cat([text_user_embedding, text_item_id_embedding], dim=0)
        
        if self.audio_modality:
            for _ in range(self.gcn_layer_num):
                audio_item_id_embedding = torch.sparse.mm(diffusion_ii_audio_adj, audio_item_id_embedding)
            audio_user_embedding = torch.sparse.mm(R, audio_item_id_embedding)
            audio_ui_embedding = torch.cat([audio_user_embedding, audio_item_id_embedding], dim=0)
            return (image_ui_embedding, text_ui_embedding, audio_ui_embedding)
        else:
            return (image_ui_embedding, text_ui_embedding)
    
    def gate_attention_fusion(self, image_ui_embedding, text_ui_embedding, audio_ui_embedding=None):
        """GAT Attention Fusion"""
        if self.audio_modality and audio_ui_embedding is not None:
            attention_common = torch.cat([
                self.caculate_common(image_ui_embedding),
                self.caculate_common(text_ui_embedding),
                self.caculate_common(audio_ui_embedding)
            ], dim=-1)
            weight_common = self.softmax(attention_common)
            common_embedding = (weight_common[:, 0].unsqueeze(dim=1) * image_ui_embedding +
                              weight_common[:, 1].unsqueeze(dim=1) * text_ui_embedding +
                              weight_common[:, 2].unsqueeze(dim=1) * audio_ui_embedding)
            sepcial_image_ui_embedding = image_ui_embedding - common_embedding
            special_text_ui_embedding = text_ui_embedding - common_embedding
            special_audio_ui_embedding = audio_ui_embedding - common_embedding
            return sepcial_image_ui_embedding, special_text_ui_embedding, special_audio_ui_embedding, common_embedding
        else:
            attention_common = torch.cat([
                self.caculate_common(image_ui_embedding),
                self.caculate_common(text_ui_embedding)
            ], dim=-1)
            weight_common = self.softmax(attention_common)
            common_embedding = (weight_common[:, 0].unsqueeze(dim=1) * image_ui_embedding +
                              weight_common[:, 1].unsqueeze(dim=1) * text_ui_embedding)
            sepcial_image_ui_embedding = image_ui_embedding - common_embedding
            special_text_ui_embedding = text_ui_embedding - common_embedding
            return sepcial_image_ui_embedding, special_text_ui_embedding, common_embedding
    
    def forward(self, R, original_ui_adj, diffusion_ui_adj, diffusion_ii_image_adj, diffusion_ii_text_adj, 
                diffusion_ii_audio_adj=None, diffusion_modal_fusion_ii_matrix=None):
        """Forward pass"""
        content_embedding = self.user_item_GCN(original_ui_adj + diffusion_ui_adj)
        
        if self.audio_modality:
            if self.modal_fusion:
                diffusion_ii_image_adj += diffusion_modal_fusion_ii_matrix
                diffusion_ii_text_adj += diffusion_modal_fusion_ii_matrix
                diffusion_ii_audio_adj += diffusion_modal_fusion_ii_matrix
            
            image_ui_embedding, text_ui_embedding, audio_ui_embedding = self.item_item_GCN(
                R, original_ui_adj, diffusion_ii_image_adj, diffusion_ii_text_adj, diffusion_ii_audio_adj)
            
            sepcial_image_ui_embedding, special_text_ui_embedding, special_audio_ui_embedding, common_embedding = \
                self.gate_attention_fusion(image_ui_embedding, text_ui_embedding, audio_ui_embedding)
            
            image_prefer_embedding = self.gate_image_modal(content_embedding)
            text_prefer_embedding = self.gate_text_modal(content_embedding)
            audio_prefer_embedding = self.gate_audio_modal(content_embedding)
            
            sepcial_image_ui_embedding = torch.multiply(image_prefer_embedding, sepcial_image_ui_embedding)
            special_text_ui_embedding = torch.multiply(text_prefer_embedding, special_text_ui_embedding)
            special_audio_ui_embedding = torch.multiply(audio_prefer_embedding, special_audio_ui_embedding)
            
            side_embedding = (sepcial_image_ui_embedding + special_text_ui_embedding + 
                             special_audio_ui_embedding + common_embedding) / 4
            all_embedding = content_embedding + side_embedding
        else:
            if self.modal_fusion:
                diffusion_ii_image_adj += diffusion_modal_fusion_ii_matrix
                diffusion_ii_text_adj += diffusion_modal_fusion_ii_matrix
            
            image_ui_embedding, text_ui_embedding = self.item_item_GCN(
                R, original_ui_adj, diffusion_ii_image_adj, diffusion_ii_text_adj, diffusion_ii_audio_adj=None)
            
            sepcial_image_ui_embedding, special_text_ui_embedding, common_embedding = \
                self.gate_attention_fusion(image_ui_embedding, text_ui_embedding, audio_ui_embedding=None)
            
            image_prefer_embedding = self.gate_image_modal(content_embedding)
            text_prefer_embedding = self.gate_text_modal(content_embedding)
            
            sepcial_image_ui_embedding = torch.multiply(image_prefer_embedding, sepcial_image_ui_embedding)
            special_text_ui_embedding = torch.multiply(text_prefer_embedding, special_text_ui_embedding)
            
            side_embedding = (sepcial_image_ui_embedding + special_text_ui_embedding + common_embedding) / 4
            all_embedding = content_embedding + side_embedding
        
        all_embeddings_users, all_embeddings_items = torch.split(all_embedding, [self.n_users, self.n_items], dim=0)
        
        return all_embeddings_users, all_embeddings_items, side_embedding, content_embedding
    
    def bpr_loss(self, anc_embeds, pos_embeds, neg_embeds):
        """BPR loss"""
        pos_scores = torch.sum(torch.mul(anc_embeds, pos_embeds), dim=-1)
        neg_scores = torch.sum(torch.mul(anc_embeds, neg_embeds), dim=-1)
        diff_scores = pos_scores - neg_scores
        bpr_loss = -1 * torch.mean(F.logsigmoid(diff_scores))
        
        regularizer = 1.0 / 2 * (anc_embeds ** 2).sum() + 1.0 / 2 * (pos_embeds ** 2).sum() + 1.0 / 2 * (neg_embeds ** 2).sum()
        regularizer = regularizer / anc_embeds.shape[0]
        emb_loss = self.reg_weight * regularizer
        
        reg_loss = self.reg_loss() * self.reg_weight
        
        return bpr_loss, emb_loss, reg_loss
    
    def infoNCE_loss(self, view1, view2, temperature):
        """InfoNCE loss"""
        view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
        pos_score = torch.sum((view1 * view2), dim=-1)
        pos_score = torch.exp(pos_score / temperature)
        
        neg_score = (view1 @ view2.T) / temperature
        neg_score = torch.exp(neg_score).sum(dim=1)
        contrast_loss = -1 * torch.log(pos_score / neg_score).mean()
        
        return contrast_loss
    
    def reg_loss(self):
        ret = 0
        ret += self.user_embedding.weight.norm(2).square()
        ret += self.item_id_embedding.weight.norm(2).square()
        return ret
    
    def calculate_loss(self, interaction):
        """Calculate training loss"""
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]
        
        if self.image_UI_matrix is None or self.text_UI_matrix is None:
            return torch.tensor(0.0, requires_grad=True).to(self.device)
        
        diffusion_ui_adj = self.image_UI_matrix + self.text_UI_matrix
        if self.audio_modality and self.audio_UI_matrix is not None:
            diffusion_ui_adj = diffusion_ui_adj + self.audio_UI_matrix
        
        usrEmbeds, itmEmbeds, side_Embeds, content_Emebeds = self.forward(
            self.R, self.norm_adj, diffusion_ui_adj,
            self.image_II_matrix, self.text_II_matrix,
            self.audio_II_matrix if self.audio_modality else None,
            self.modal_fusion_II_matrix if self.modal_fusion else None
        )
        
        ancEmbeds = usrEmbeds[users]
        posEmbeds = itmEmbeds[pos_items]
        negEmbeds = itmEmbeds[neg_items]
        
        bprLoss, _, regLoss = self.bpr_loss(ancEmbeds, posEmbeds, negEmbeds)
        
        # Contrastive loss
        side_embeds_users, side_embeds_items = torch.split(side_Embeds, [self.n_users, self.n_items], dim=0)
        content_embeds_user, content_embeds_items = torch.split(content_Emebeds, [self.n_users, self.n_items], dim=0)
        
        clLoss1 = (self.infoNCE_loss(side_embeds_items[pos_items], content_embeds_items[pos_items], self.temp) +
                  self.infoNCE_loss(side_embeds_users[users], content_embeds_user[users], self.temp))
        
        clLoss2 = (self.infoNCE_loss(usrEmbeds[users], content_embeds_items[pos_items], self.temp) +
                  self.infoNCE_loss(usrEmbeds[users], side_embeds_items[pos_items], self.temp))
        
        clLoss = (clLoss1 + clLoss2) * self.ssl_reg
        
        loss = bprLoss + regLoss + clLoss
        return loss
    
    def full_sort_predict(self, interaction):
        """Full sort prediction"""
        user = interaction[0]
        
        if self.image_UI_matrix is None:
            # Fallback to norm_adj if matrices not generated
            usrEmbeds, itmEmbeds, _, _ = self.forward(
                self.R, self.norm_adj, self.norm_adj,
                self.norm_adj[:self.n_items, :self.n_items] if hasattr(self, 'norm_adj') else None,
                self.norm_adj[:self.n_items, :self.n_items] if hasattr(self, 'norm_adj') else None,
                None, None
            )
        else:
            diffusion_ui_adj = self.image_UI_matrix + self.text_UI_matrix
            if self.audio_modality and self.audio_UI_matrix is not None:
                diffusion_ui_adj = diffusion_ui_adj + self.audio_UI_matrix
            
            usrEmbeds, itmEmbeds, _, _ = self.forward(
                self.R, self.norm_adj, diffusion_ui_adj,
                self.image_II_matrix, self.text_II_matrix,
                self.audio_II_matrix if self.audio_modality else None,
                self.modal_fusion_II_matrix if self.modal_fusion else None
            )
        
        score_mat_ui = torch.matmul(usrEmbeds[user], itmEmbeds.transpose(0, 1))
        return score_mat_ui


# Helper classes and functions

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


class ModalDenoise(nn.Module):
    """Modal denoise model for item-item interactions"""
    def __init__(self, in_dims, out_dims, emb_size, norm=False, dropout=0.1):
        super(ModalDenoise, self).__init__()
        self.in_dims = in_dims
        self.out_dims = out_dims
        self.time_emb_dim = emb_size
        self.norm = norm
        
        in_features = in_dims + self.time_emb_dim
        
        self.down_sampling = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=self.in_dims // 2),
            nn.BatchNorm1d(self.in_dims // 2),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(in_features=self.in_dims // 2, out_features=self.in_dims//4),
            nn.BatchNorm1d(self.in_dims//4),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(in_features=self.in_dims // 4, out_features=self.in_dims//8),
            nn.BatchNorm1d(self.in_dims//8),
            nn.LeakyReLU(),
            nn.Dropout(0.1)
        )
        
        self.up_sampling = nn.Sequential(
            nn.Linear(in_features=self.in_dims//8, out_features=self.in_dims//4),
            nn.BatchNorm1d(self.in_dims//4),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(in_features=self.in_dims//4, out_features=self.in_dims//2),
            nn.BatchNorm1d(self.in_dims // 2),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(in_features=self.in_dims//2, out_features=self.in_dims),
            nn.BatchNorm1d(self.in_dims),
            nn.LeakyReLU(),
            nn.Dropout(0.1)
        )
        
        self.emb_layer = nn.Linear(self.time_emb_dim, self.time_emb_dim)
        self.drop = nn.Dropout(dropout)
        self.initialize_weights()
    
    def initialize_weights(self):
        for module_seq in [self.down_sampling, self.up_sampling]:
            for layer in module_seq:
                if isinstance(layer, nn.Linear):
                    size = layer.weight.size()
                    std = np.sqrt(2.0 / (size[0] + size[1]))
                    layer.weight.data.normal_(0.0, std)
                    layer.bias.data.normal_(0.0, 0.001)
    
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
        h = self.down_sampling(h)
        h = self.up_sampling(h)
        return h


class ImageEncoder(nn.Module):
    def __init__(self, image_feature_dim, hidden_dim):
        super(ImageEncoder, self).__init__()
        self.fc1 = nn.Linear(image_feature_dim, hidden_dim)
        self.norm1 = nn.BatchNorm1d(hidden_dim)
        self.relu1 = nn.LeakyReLU()
        self.drop1 = nn.Dropout(0.1)
        
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = nn.BatchNorm1d(hidden_dim)
        self.relu2 = nn.LeakyReLU()
        self.drop2 = nn.Dropout(0.1)
        self._init_weights()
    
    def forward(self, image_features):
        x_ = self.drop1(self.relu1(self.norm1(self.fc1(image_features))))
        x = self.drop2(self.relu2(self.norm2(self.fc2(x_))))
        return x + x_
    
    def _init_weights(self):
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='leaky_relu')
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.constant_(self.fc2.bias, 0)


class TextEncoder(nn.Module):
    def __init__(self, text_feature_dim, hidden_dim):
        super(TextEncoder, self).__init__()
        self.fc1 = nn.Linear(text_feature_dim, hidden_dim)
        self.norm1 = nn.BatchNorm1d(hidden_dim)
        self.relu1 = nn.LeakyReLU()
        self.drop1 = nn.Dropout(0.1)
        
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = nn.BatchNorm1d(hidden_dim)
        self.relu2 = nn.LeakyReLU()
        self.drop2 = nn.Dropout(0.1)
        self._init_weights()
    
    def forward(self, text_embeddings):
        x_ = self.drop1(self.relu1(self.norm1(self.fc1(text_embeddings))))
        x = self.drop2(self.relu2(self.norm2(self.fc2(x_))))
        return x + x_
    
    def _init_weights(self):
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='leaky_relu')
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.constant_(self.fc2.bias, 0)


class AudioEncoder(nn.Module):
    def __init__(self, audio_feature_dim, hidden_dim):
        super(AudioEncoder, self).__init__()
        self.fc1 = nn.Linear(audio_feature_dim, hidden_dim)
        self.norm1 = nn.BatchNorm1d(hidden_dim)
        self.relu1 = nn.LeakyReLU()
        self.drop1 = nn.Dropout(0.1)
        
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = nn.BatchNorm1d(hidden_dim)
        self.relu2 = nn.LeakyReLU()
        self.drop2 = nn.Dropout(0.1)
        self._init_weights()
    
    def forward(self, audio_features):
        x_ = self.drop1(self.relu1(self.norm1(self.fc1(audio_features))))
        x = self.drop2(self.relu2(self.norm2(self.fc2(x_))))
        return x + x_
    
    def _init_weights(self):
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='leaky_relu')
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.constant_(self.fc2.bias, 0)


class CrossModalAttention(nn.Module):
    def __init__(self, dim_key, dim_value, dim_query):
        super(CrossModalAttention, self).__init__()
        self.query_proj = nn.Linear(dim_query, dim_key)
        self.key_proj = nn.Linear(dim_key, dim_key)
        self.value_proj = nn.Linear(dim_value, dim_value)
        self.softmax = nn.Softmax(dim=-1)
        self._init_weights()
    
    def forward(self, query, key, value):
        query = self.query_proj(query)
        key = self.key_proj(key)
        value = self.value_proj(value)
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / (query.size(-1) ** 0.5)
        attention_weights = self.softmax(attention_scores)
        attended_value = torch.matmul(attention_weights, value)
        return attended_value
    
    def _init_weights(self):
        nn.init.kaiming_normal_(self.query_proj.weight, nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.key_proj.weight, nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.value_proj.weight, nonlinearity='leaky_relu')
        nn.init.constant_(self.query_proj.bias, 0)
        nn.init.constant_(self.key_proj.bias, 0)
        nn.init.constant_(self.value_proj.bias, 0)


class MultimodalDenoiseModel(nn.Module):
    """Multimodal denoise model for feature diffusion"""
    def __init__(self, image_in_dims, text_in_dims, audio_in_dims, out_dims, time_emb_size, modal_flag='image', audio_modality=False):
        super(MultimodalDenoiseModel, self).__init__()
        self.image_in_dims = image_in_dims
        self.text_in_dims = text_in_dims
        self.audio_in_dims = audio_in_dims
        self.time_emb_dim = time_emb_size
        self.out_dims = out_dims
        self.modal_flag = modal_flag
        self.audio_modality = audio_modality
        
        self.time_embedding_layer = nn.Linear(self.time_emb_dim, self.time_emb_dim)
        
        self.fusion_hidden_dim = (self.image_in_dims + self.text_in_dims + self.audio_in_dims) if audio_modality else (self.image_in_dims + self.text_in_dims)
        
        self.image_encoder = ImageEncoder(image_feature_dim=self.image_in_dims + self.time_emb_dim, hidden_dim=self.image_in_dims)
        self.text_encoder = TextEncoder(text_feature_dim=self.text_in_dims + self.time_emb_dim, hidden_dim=self.text_in_dims)
        if audio_modality:
            self.audio_encoder = AudioEncoder(audio_feature_dim=self.audio_in_dims + self.time_emb_dim, hidden_dim=self.audio_in_dims)
        
        self.fusion_layer = nn.Sequential(
            nn.Linear(in_features=self.fusion_hidden_dim, out_features=self.out_dims),
            nn.BatchNorm1d(self.out_dims),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(in_features=self.out_dims, out_features=self.out_dims),
            nn.BatchNorm1d(self.out_dims),
            nn.LeakyReLU(),
            nn.Dropout(0.1)
        )
        
        self.image_attention = CrossModalAttention(self.out_dims, self.out_dims, self.out_dims)
        self.text_attention = CrossModalAttention(self.out_dims, self.out_dims, self.out_dims)
        if audio_modality:
            self.audio_attention = CrossModalAttention(self.out_dims, self.out_dims, self.out_dims)
    
    def time_embedding(self, timesteps):
        """Time embedding"""
        freqs = torch.exp(-math.log(10000) * torch.arange(start=0, end=self.time_emb_dim//2, dtype=torch.float32) / (self.time_emb_dim//2)).to(timesteps.device)
        temp = timesteps[:, None].float() * freqs[None]
        time_emb = torch.cat([torch.cos(temp), torch.sin(temp)], dim=-1)
        if self.time_emb_dim % 2:
            time_emb = torch.cat([time_emb, torch.zeros_like(time_emb[:, :1])], dim=-1)
        emb = self.time_embedding_layer(time_emb)
        return emb
    
    def forward(self, x_image, x_text, x_audio, timesteps):
        x_image_feature, x_text_feature, x_audio_feature = None, None, None
        time_emb = self.time_embedding(timesteps)
        
        if x_image is not None:
            x_image = torch.cat([x_image, time_emb], dim=-1)
            x_image_feature = self.image_encoder(x_image)
        if x_text is not None:
            x_text = torch.cat([x_text, time_emb], dim=-1)
            x_text_feature = self.text_encoder(x_text)
        if x_audio is not None and self.audio_modality:
            x_audio = torch.cat([x_audio, time_emb], dim=-1)
            x_audio_feature = self.audio_encoder(x_audio)
        
        # Multimodal fusion
        if self.audio_modality and x_audio_feature is not None:
            fusion_features = torch.cat([x_image_feature, x_text_feature, x_audio_feature], dim=-1)
        else:
            fusion_features = torch.cat([x_image_feature, x_text_feature], dim=-1)
        fusion_features = self.fusion_layer(fusion_features)
        
        # Calculate multimodal attention
        if self.modal_flag == 'image':
            attention_feature = self.image_attention(fusion_features, x_image_feature, x_image_feature)
        elif self.modal_flag == 'text':
            attention_feature = self.text_attention(fusion_features, x_text_feature, x_text_feature)
        elif self.modal_flag == 'audio' and self.audio_modality:
            attention_feature = self.audio_attention(fusion_features, x_audio_feature, x_audio_feature)
        else:
            attention_feature = fusion_features
        
        out = fusion_features + attention_feature
        return out


class GaussianDiffusion(nn.Module):
    """Gaussian diffusion for multimodal features"""
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
        self.alphas = alphas
        self.sqrt_alphas = torch.sqrt(alphas)
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
    
    def p_sample(self, model, x_start_image, x_start_text, x_start_audio, steps, sampling_noise=False, modal_flag='image'):
        batch_size = x_start_image.shape[0]
        if steps == 0:
            x_t_image = x_start_image
            x_t_text = x_start_text
            x_t_audio = x_start_audio
        else:
            t = torch.tensor([steps-1] * batch_size).to(x_start_image.device)
            x_t_image = self.q_sample(x_start_image, t, None)
            x_t_text = self.q_sample(x_start_text, t, None)
            if x_start_audio is not None:
                x_t_audio = self.q_sample(x_start_audio, t, None)
        
        indices = list(range(self.steps))[::-1]
        
        for i in indices:
            t = torch.tensor([i] * batch_size).to(x_start_image.device)
            model_mean, model_log_variance = self.p_mean_variance(model, x_t_image, x_t_text, x_t_audio, t, modal_flag)
            if modal_flag == 'image':
                x_t_image = model_mean
            elif modal_flag == 'text':
                x_t_text = model_mean
            elif modal_flag == 'audio':
                x_t_audio = model_mean
        
        if modal_flag == 'image':
            return x_t_image
        elif modal_flag == 'text':
            return x_t_text
        elif modal_flag == 'audio':
            return x_t_audio
        return x_t_image
    
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        return (self._extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                self._extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)
    
    def _extract_into_tensor(self, arr, timesteps, broadcast_shape):
        arr = arr.to(timesteps.device)
        res = arr[timesteps].float()
        while len(res.shape) < len(broadcast_shape):
            res = res[..., None]
        return res.expand(broadcast_shape)
    
    def p_mean_variance(self, model, x_t_image, x_t_text, x_t_audio, t, modal_flag):
        model_output = model(x_t_image, x_t_text, x_t_audio, t)
        
        model_variance = self.posterior_variance
        model_log_variance = self.posterior_log_variance_clipped
        
        if modal_flag == 'image':
            x = x_t_image
        elif modal_flag == 'text':
            x = x_t_text
        elif modal_flag == 'audio':
            x = x_t_audio
        else:
            x = x_t_image
        
        model_variance = self._extract_into_tensor(model_variance, t, x.shape)
        model_log_variance = self._extract_into_tensor(model_log_variance, t, x.shape)
        
        model_mean = (self._extract_into_tensor(self.posterior_mean_coef1, t, x.shape) * model_output +
                     self._extract_into_tensor(self.posterior_mean_coef2, t, x.shape) * x)
        
        return model_mean, model_log_variance
    
    def training_multimodal_feature_diffusion_losses(self, model, x_start_image, x_start_text, x_start_audio=None, modal_flag='image'):
        batch_size = x_start_image.size(0)
        ts = torch.randint(0, self.steps, (batch_size,)).long().to(x_start_image.device)
        
        noise_image = torch.randn_like(x_start_image)
        noise_text = torch.randn_like(x_start_text)
        x_t_image = self.q_sample(x_start_image, ts, noise_image)
        x_t_text = self.q_sample(x_start_text, ts, noise_text)
        x_t_audio = None
        if x_start_audio is not None:
            noise_audio = torch.randn_like(x_start_audio)
            x_t_audio = self.q_sample(x_start_audio, ts, noise_audio)
        
        model_output = model(x_t_image, x_t_text, x_t_audio, ts)
        
        if modal_flag == 'image':
            mse = self.mean_flat((noise_image - model_output) ** 2)
        elif modal_flag == 'text':
            mse = self.mean_flat((noise_text - model_output) ** 2)
        elif modal_flag == 'audio' and x_start_audio is not None:
            mse = self.mean_flat((noise_audio - model_output) ** 2)
        else:
            mse = self.mean_flat((noise_image - model_output) ** 2)
        
        return mse
    
    def mean_flat(self, tensor):
        return tensor.mean(dim=list(range(1, len(tensor.shape))))


class SparityDiffusion(nn.Module):
    """Sparsity diffusion for user-item interactions"""
    def __init__(self, noise_scale, noise_min, noise_max, steps, alpha_sparity, beta_sparity, postive_gain_degree, sparse_temp, beta_fixed=True):
        super(SparityDiffusion, self).__init__()
        self.alpha_sparity = alpha_sparity
        self.beta_sparity = beta_sparity
        self.postive_gain_degree = postive_gain_degree
        self.sparse_temp = sparse_temp
        self.open_noise_adaptive = True
        
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
            t = torch.tensor([i] * x_t.shape[0]).to(x_start.device)
            model_mean, model_log_variance = self.p_mean_variance(model, x_t, t)
            if sampling_noise:
                noise = torch.randn_like(x_t)
                nonzero_mask = ((t!=0).float().view(-1, *([1]*(len(x_t.shape)-1))))
                x_t = model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise
            else:
                x_t = model_mean
        return x_t
    
    def q_sample(self, x_start, t, noise=None):
        if self.open_noise_adaptive:
            batch_size = x_start.shape[0]
            item_size = x_start.shape[1]
            batch_noise_adaptive_penalty_factor = 1 - (x_start.sum() / (batch_size * item_size))
            noise_coe = self.alpha_sparity * (1 + batch_noise_adaptive_penalty_factor) * torch.exp(-1.0 * self.beta_sparity * t.float())
            ones_tensor = torch.ones_like(x_start)
            batch_postive_position_mask_matirx = torch.where(x_start == 0, ones_tensor - x_start, self.postive_gain_degree * x_start)
            noise_coe = noise_coe.unsqueeze(1)
            noise_coe = noise_coe * batch_postive_position_mask_matirx
        else:
            noise_coe = torch.ones_like(x_start)
        
        if noise is None:
            noise = torch.randn_like(x_start)
        noise = noise * noise_coe
        
        return (self._extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                self._extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)
    
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
        
        model_mean = (self._extract_into_tensor(self.posterior_mean_coef1, t, x.shape) * model_output +
                     self._extract_into_tensor(self.posterior_mean_coef2, t, x.shape) * x)
        
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
        
        mse = self.mean_flat((noise - model_output) ** 2)
        weight = self.SNR(ts - 1) - self.SNR(ts)
        weight = torch.where((ts == 0), 1.0, weight)
        
        diff_loss = weight * mse
        
        usr_model_embeds = torch.mm(model_output, model_feats)
        usr_id_embeds = torch.mm(x_start, itmEmbeds)
        gc_loss = self.mean_flat((usr_model_embeds - usr_id_embeds) ** 2)
        
        model_feat_embedding = torch.multiply(itmEmbeds, model_feats)
        model_feat_embedding_origin = torch.mm(x_start, model_feat_embedding)
        model_feat_embedding_diffusion = torch.mm(model_output, model_feat_embedding)
        
        contra_loss = self.infoNCE_loss(model_feat_embedding_origin, model_feat_embedding_diffusion, self.sparse_temp)
        
        return diff_loss, gc_loss, contra_loss
    
    def infoNCE_loss(self, view1, view2, temperature):
        view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
        pos_score = torch.sum((view1 * view2), dim=-1)
        pos_score = torch.exp(pos_score / temperature)
        
        neg_score = (view1 @ view2.T) / temperature
        neg_score = torch.exp(neg_score).sum(dim=1)
        contrast_loss = -1 * torch.log(pos_score / neg_score).mean()
        
        return contrast_loss
    
    def mean_flat(self, tensor):
        return tensor.mean(dim=list(range(1, len(tensor.shape))))
    
    def SNR(self, t):
        self.alphas_cumprod = self.alphas_cumprod.to(t.device)
        return self.alphas_cumprod[t] / (1 - self.alphas_cumprod[t])
