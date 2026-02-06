# coding: utf-8
# Desc: Core code of the GenRecV1.
# Author: OrangeAI Research Team
# Time: 2025-03-24
# paper: "Flip is Better than Noise: Unbiased Interest Generation for Multimedia Recommendation" (GenRec-V1, ACM MM 2025) 

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import scipy.sparse as sp
from common.abstract_recommender import GeneralRecommender

# 生成质量四指标（与 GenRecv2 一致，便于跨模型对比）
from models.genrecv2 import sliced_wasserstein_distance, fid_embedding, trajectory_linearity

class GenRecV1(GeneralRecommender):
	def __init__(self, config, dataset):
		super(GenRecV1, self).__init__(config, dataset)
		self.config = config

		# Config parameters
		self.latdim = config['embedding_size']
		self.n_layers = config['n_layers']
		self.keep_rate = config['keep_rate']
		self.sparse_temp = config['sparse_temp']
		self.temp = config['temperature']
		self.ssl_reg1 = config['ssl_reg1']
		self.ssl_reg2 = config['ssl_reg2']
		self.gen_topk = config['gen_topk']
		self.rebuild_k = config['rebuild_k']
		self.d_emb_size = config['d_emb_size']
		self.nhead = config['nhead']
		self.num_layers = config['num_layers']
		self.lr = config['learning_rate']
		self.steps = config['steps']
		self.flip_temp = config['flip_temp']
		self.bayesian_samplinge_schedule = config['bayesian_samplinge_schedule']
		self.sampling_steps = config['sampling_steps']
		# modality switcher
		self.visual_modality = config['visual_modality']
		self.text_modality = config['text_modality']
		self.audio_modality = config['audio_modality']
		# Load features
		self.image_embedding = self.v_feat
		self.text_embedding = self.t_feat
		if self.audio_modality:
			self.audio_embedding = self.a_feat 
		else:
			self.audio_embedding = None 
	
		
		self.sparse = True
		
		# Pre-calculate graph
		self.norm_adj = self.get_norm_adj_mat(dataset.inter_matrix(form='coo').astype(np.float32)).to(self.device)
		self.R = self._get_user_item_matrix(dataset.inter_matrix(form='coo').astype(np.float32)).to(self.device)

		# Model components
		self.edgeDropper = SpAdjDropEdge(self.keep_rate)
		
		self.origin_weight = nn.Parameter(torch.ones(1))
		self.generation_weight = nn.Parameter(torch.ones(1))
		
		self.img_weight = nn.Parameter(torch.ones(1))
		self.txt_weight = nn.Parameter(torch.ones(1))
		self.aud_weight = nn.Parameter(torch.ones(1))
		
		nn.init.normal_(self.img_weight, mean=1.0, std=0.1)
		nn.init.normal_(self.txt_weight, mean=1.0, std=0.1)
		nn.init.normal_(self.aud_weight, mean=1.0, std=0.1)

		# User & Item Embeddings
		self.user_embedding = nn.Embedding(self.n_users, self.latdim)
		self.item_id_embedding = nn.Embedding(self.n_items, self.latdim)
		nn.init.xavier_uniform_(self.user_embedding.weight)
		nn.init.xavier_uniform_(self.item_id_embedding.weight)
		
		self.fusion_weight = nn.Parameter(torch.ones(3))
		self.res_scale = nn.Parameter(torch.ones(1))
		
		# Modality Projections
		self._init_modal_projections()
		
		# Attention Fusion
		self.caculate_common = nn.Sequential(
			nn.Linear(self.latdim, self.latdim),
			nn.BatchNorm1d(self.latdim),
			nn.Tanh(),
			nn.Linear(self.latdim, 1, bias=False)
		)
		for layer in self.caculate_common:
			if isinstance(layer, nn.Linear):
				nn.init.xavier_uniform_(layer.weight)
				
		self.gate_image_modal = self._build_gate()
		self.gate_text_modal = self._build_gate()
		self.gate_audio_modal = self._build_gate()

		# Diffusion Model
		self.diffusion_model = FlipInterestDiffusion(config=config, steps=self.steps, base_temp=self.flip_temp)
		
		# Denoise Model
		# Input dim is image feature dim (e.g. 4096 or 128)
		# GenRecV1 uses image_embedding.shape[0] as in_dims/out_dims ?? 
		# Wait, in Main.py: out_dims = self.image_embedding.shape[0] -> This is n_items!
		# It seems GenRecV1 performs diffusion on the User-Item interaction vector (size n_items).
		self.denoise_in_dims = self.n_items
		self.denoise_out_dims = self.n_items
		
		self.denoise_model_image = ModalDenoiseTransformer(
			in_dims=self.denoise_in_dims,
			out_dims=self.denoise_out_dims,
			emb_size=self.d_emb_size,
			nhead=self.nhead,
			num_layers=self.num_layers
		).to(self.device)
		
		# Placeholders for generated matrices
		self.image_UI_matrix = None
		self.image_II_matrix = None
		self.text_II_matrix = None
		
		# Initialize II matrices
		# This usually happens once. We can do it in init or trainer.
		# But GeneralRecommender usually doesn't do heavy computation in init.
		# We'll leave it to trainer or a setup method.


	def _get_user_item_matrix(self, interaction_matrix):
		i = torch.LongTensor(np.array([interaction_matrix.row, interaction_matrix.col]))
		data = torch.FloatTensor(interaction_matrix.data)
		return torch.sparse.FloatTensor(i, data, torch.Size((self.n_users, self.n_items)))

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


	def _build_gate(self):
		gate = nn.Sequential(
			nn.Linear(self.latdim, self.latdim),
			nn.BatchNorm1d(self.latdim),
			nn.Sigmoid()
		)
		for layer in gate:
			if isinstance(layer, nn.Linear):
				nn.init.xavier_uniform_(layer.weight)
		return gate

	def _init_modal_projections(self):
		# Image
		if self.image_embedding is not None:
			self.image_residual_project = nn.Sequential(
				nn.Linear(self.image_embedding.shape[1], self.latdim),
				nn.BatchNorm1d(self.latdim),
				nn.LeakyReLU(negative_slope=0.2),
				nn.Dropout(0.1)
			)
			self.image_modal_project = nn.Sequential(
				nn.Linear(self.latdim, self.latdim),
				nn.BatchNorm1d(self.latdim),
				nn.LeakyReLU(negative_slope=0.2),
				nn.Dropout(0.1)
			)
			for layer in self.image_residual_project:
				if isinstance(layer, nn.Linear): nn.init.xavier_uniform_(layer.weight)
			for layer in self.image_modal_project:
				if isinstance(layer, nn.Linear): nn.init.xavier_uniform_(layer.weight)

		# Text
		if self.text_embedding is not None:
			self.text_residual_project = nn.Sequential(
				nn.Linear(self.text_embedding.shape[1], self.latdim),
				nn.BatchNorm1d(self.latdim),
				nn.LeakyReLU(negative_slope=0.2),
				nn.Dropout(0.1)
			)
			self.text_modal_project = nn.Sequential(
				nn.Linear(self.latdim, self.latdim),
				nn.BatchNorm1d(self.latdim),
				nn.LeakyReLU(negative_slope=0.2),
				nn.Dropout(0.1)
			)
			for layer in self.text_residual_project:
				if isinstance(layer, nn.Linear): nn.init.xavier_uniform_(layer.weight)
			for layer in self.text_modal_project:
				if isinstance(layer, nn.Linear): nn.init.xavier_uniform_(layer.weight)

		# Audio 
		if self.audio_embedding is not None:
			self.audio_residual_project = nn.Sequential(
				nn.Linear(in_features=self.audio_embedding.shape[1], out_features=self.latdim),
				nn.BatchNorm1d(self.latdim),
				nn.LeakyReLU(negative_slope=0.2),
				nn.Dropout(0.1)
			)
			self.audio_modal_project = nn.Sequential(
				nn.Linear(in_features=self.latdim, out_features=self.latdim),
				nn.BatchNorm1d(self.latdim),
				nn.LeakyReLU(negative_slope=0.2),
				nn.Dropout(0.1)
			)
			for layer in self.audio_residual_project:
				if isinstance(layer, nn.Linear): nn.init.xavier_uniform_(layer.weight)
			for layer in self.audio_modal_project:
				if isinstance(layer, nn.Linear): nn.init.xavier_uniform_(layer.weight)


	def getImageFeats(self):
		if self.image_embedding is not None:
			x = self.image_residual_project(self.image_embedding)
			image_modal_feature = self.image_modal_project(x)
			image_modal_feature = self.res_scale * x + image_modal_feature
			return image_modal_feature
		return None

	def getTextFeats(self):
		if self.text_embedding is not None:
			x = self.text_residual_project(self.text_embedding)
			text_modal_feature = self.text_modal_project(x)
			text_modal_feature = self.res_scale * x + text_modal_feature
			return text_modal_feature
		return None

	def getAudioFeats(self):
		if self.audio_embedding is not None:
			x = self.audio_residual_project(self.audio_embedding)
			audio_modal_feature = self.audio_modal_project(x)
			audio_modal_feature = self.res_scale * x + audio_modal_feature
			return audio_modal_feature
		return None
	
	def getItemEmbeds(self):
		return self.item_id_embedding.weight
	
	def getUserEmbeds(self):
		return self.user_embedding.weight

	def user_item_GCN(self, adj):
		cat_embedding = torch.cat([self.user_embedding.weight, self.item_id_embedding.weight], dim=0)
		all_embeddings = [cat_embedding]
		for i in range(self.n_layers):
			temp_embeddings2 = torch.sparse.mm(adj, cat_embedding)
			cat_embedding = temp_embeddings2
			all_embeddings += [cat_embedding]
		all_embeddings = torch.stack(all_embeddings, dim=1)
		all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
		return all_embeddings

	def item_item_GCN(self, R, original_ui_adj, diffusion_ii_image_adj, diffusion_ii_text_adj, diffusion_ii_audio_adj=None):
		image_modal_feature = self.getImageFeats()
		image_item_id_embedding = torch.multiply(self.item_id_embedding.weight, self.gate_image_modal(image_modal_feature))

		text_modal_feature = self.getTextFeats()
		text_item_id_embedding = torch.multiply(self.item_id_embedding.weight, self.gate_text_modal(text_modal_feature))

		if self.audio_modality:
			audio_modal_feature = self.getAudioFeats()
			audio_item_id_embedding = torch.multiply(self.item_id_embedding.weight, self.gate_audio_modal(audio_modal_feature))

		if self.sparse:
			for _ in range(self.n_layers):
				image_item_id_embedding = torch.sparse.mm(diffusion_ii_image_adj, image_item_id_embedding)
		else:
			for _ in range(self.n_layers):
				image_item_id_embedding = torch.mm(diffusion_ii_image_adj, image_item_id_embedding)

		image_user_embedding = torch.sparse.mm(R, image_item_id_embedding) 
		image_ui_embedding = torch.cat([image_user_embedding, image_item_id_embedding], dim=0)

		if self.sparse:
			for _ in range(self.n_layers):
				text_item_id_embedding = torch.sparse.mm(diffusion_ii_text_adj, text_item_id_embedding)
		else:
			for _ in range(self.n_layers):
				text_item_id_embedding = torch.mm(diffusion_ii_text_adj, text_item_id_embedding)
		text_user_embedding = torch.sparse.mm(R, text_item_id_embedding) 
		text_ui_embedding = torch.cat([text_user_embedding, text_item_id_embedding], dim=0)

		if self.audio_modality:
			if self.sparse:
				for _ in range(self.n_layers):
					audio_item_id_embedding = torch.sparse.mm(diffusion_ii_audio_adj, audio_item_id_embedding)
			else:
				for _ in range(self.n_layers):
					audio_item_id_embedding = torch.mm(diffusion_ii_audio_adj, audio_item_id_embedding)
			audio_user_embedding = torch.sparse.mm(R, audio_item_id_embedding) 
			audio_ui_embedding = torch.cat([text_user_embedding, audio_item_id_embedding], dim=0)
		
		return (image_ui_embedding, text_ui_embedding, audio_ui_embedding) if self.audio_modality else (image_ui_embedding, text_ui_embedding)
	

	def gate_attention_fusion(self, image_ui_embedding, text_ui_embedding, audio_ui_embedding=None):

		if self.audio_modality:

			attention_common = torch.cat([self.caculate_common(image_ui_embedding), self.caculate_common(text_ui_embedding), self.caculate_common(audio_ui_embedding)], dim=-1)
			weight_common = self.softmax(attention_common)
			common_embedding = weight_common[:, 0].unsqueeze(dim=1) * image_ui_embedding + weight_common[:, 1].unsqueeze(dim=1) * text_ui_embedding + weight_common[:, 2].unsqueeze(dim=1) * audio_ui_embedding
			sepcial_image_ui_embedding = image_ui_embedding - common_embedding
			special_text_ui_embedding  = text_ui_embedding - common_embedding
			special_audio_ui_embedding = audio_ui_embedding - common_embedding

			return sepcial_image_ui_embedding, special_text_ui_embedding, special_audio_ui_embedding, common_embedding

		attention_common = torch.cat([self.caculate_common(image_ui_embedding), self.caculate_common(text_ui_embedding)], dim=-1)
		weight_common = F.softmax(attention_common, dim=-1)
		common_embedding = weight_common[:, 0].unsqueeze(dim=1) * image_ui_embedding + weight_common[:, 1].unsqueeze(dim=1) * text_ui_embedding 
		sepcial_image_ui_embedding = image_ui_embedding - common_embedding
		special_text_ui_embedding  = text_ui_embedding - common_embedding

		return sepcial_image_ui_embedding, special_text_ui_embedding, common_embedding

	def forward(self, R, original_ui_adj, diffusion_ui_image_adj, diffusion_ii_image_adj, diffusion_ii_text_adj):
		# User-Item GCN
		content_embedding1 = self.user_item_GCN(original_ui_adj)
		content_embedding2 = self.user_item_GCN(diffusion_ui_image_adj)
		
		weights = F.softmax(torch.stack([self.origin_weight, self.generation_weight]), dim=0)
		content_embedding = weights[0] * content_embedding1 + weights[1] * content_embedding2
		
		# Item-Item GCN
		image_ui_embedding, text_ui_embedding = self.item_item_GCN(R, original_ui_adj, diffusion_ii_image_adj, diffusion_ii_text_adj, diffusion_ii_audio_adj=None)
		
		# Fusion
		sepcial_image_ui_embedding, special_text_ui_embedding, common_embedding = self.gate_attention_fusion(image_ui_embedding, text_ui_embedding, audio_ui_embedding=None)
		
		image_prefer_embedding = self.gate_image_modal(content_embedding) 
		text_prefer_embedding = self.gate_text_modal(content_embedding) 
		
		sepcial_image_ui_embedding = torch.multiply(image_prefer_embedding, sepcial_image_ui_embedding)
		special_text_ui_embedding = torch.multiply(text_prefer_embedding, special_text_ui_embedding)

		side_embedding = (sepcial_image_ui_embedding + special_text_ui_embedding + common_embedding) / 4
		all_embedding = content_embedding
		
		return all_embedding, side_embedding

	def calculate_loss(self, interaction):
		# Assuming interaction is [users, pos_items, neg_items] from Trainer
		# Or batch interaction object.
		users = interaction[0]
		pos_items = interaction[1]
		neg_items = interaction[2]
		
		# Check if matrices are ready (set by Trainer)
		if self.image_UI_matrix is None:
			return torch.tensor(0.0, requires_grad=True).to(self.device)

		# R is needed for Item-Item GCN. R is usually the full sparse matrix.
		# Trainer should set self.R
		
		content_Emebeds, side_Embeds = self.forward(self.R, self.norm_adj, self.image_UI_matrix, self.image_II_matrix, self.text_II_matrix)
		
		usrEmbeds, itmEmbeds = torch.split(content_Emebeds, [self.n_users, self.n_items], dim=0)
		
		ancEmbeds = usrEmbeds[users]
		posEmbeds = itmEmbeds[pos_items]
		negEmbeds = itmEmbeds[neg_items]
		
		# BPR Loss
		pos_scores = torch.sum(torch.mul(ancEmbeds, posEmbeds), dim=-1)
		neg_scores = torch.sum(torch.mul(ancEmbeds, negEmbeds), dim=-1)
		bpr_loss = -1 * torch.mean(F.logsigmoid(pos_scores - neg_scores))
		
		# Reg Loss
		reg_loss = self.reg_loss() * self.config['reg_weight']
		
		# Contrastive Loss
		side_embeds_users, side_embeds_items = torch.split(side_Embeds, [self.n_users, self.n_items], dim=0)
		content_embeds_user, content_embeds_items = torch.split(content_Emebeds, [self.n_users, self.n_items], dim=0)

		# item-item contrastive loss
		clLoss1 = self.infoNCE_loss(side_embeds_items[pos_items], content_embeds_items[pos_items], self.temp) + \
				  self.infoNCE_loss(side_embeds_users[users], content_embeds_user[users], self.temp)
		
		# user-item contrastive loss
		clLoss2 = self.infoNCE_loss(usrEmbeds[users], content_embeds_items[pos_items], self.temp) + \
				  self.infoNCE_loss(usrEmbeds[users], side_embeds_items[pos_items], self.temp)

		clLoss = clLoss1 * self.ssl_reg1 + clLoss2 * self.ssl_reg2
		
		return bpr_loss + reg_loss + clLoss

	def reg_loss(self):
		ret = 0
		ret += self.user_embedding.weight.norm(2).square()
		ret += self.item_id_embedding.weight.norm(2).square()
		return ret

	def infoNCE_loss(self, view1, view2, temperature):
		view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
		pos_score = torch.sum((view1 * view2), dim=-1)
		pos_score = torch.exp(pos_score / temperature)
		neg_score = (view1 @ view2.T) / temperature
		neg_score = torch.exp(neg_score).sum(dim=1)
		contrast_loss = -1 * torch.log(pos_score / neg_score).mean()
		return contrast_loss


	def full_sort_predict(self, interaction):
			user = interaction[0]
			if self.image_UI_matrix is None:
				# Fallback
				return torch.zeros(len(user), self.n_items).to(self.device)
			
			content_Emebeds, _ = self.forward(self.R, self.norm_adj, self.image_UI_matrix, self.image_II_matrix, self.text_II_matrix)
			usrEmbeds, itmEmbeds = torch.split(content_Emebeds, [self.n_users, self.n_items], dim=0)
			
			score_mat_ui = torch.matmul(usrEmbeds[user], itmEmbeds.transpose(0, 1))
			return score_mat_ui

	@torch.no_grad()
	def _get_origin_and_generated_embeds(self):
		"""原始 GCN 兴趣表征 vs 扩散生成图上的 GCN 兴趣表征，用于验证时计算生成质量指标。"""
		if self.image_UI_matrix is None:
			return None, None
		# origin = GCN(原始UI图), generated = GCN(扩散生成的UI图)
		content_embedding1 = self.user_item_GCN(self.norm_adj)
		content_embedding2 = self.user_item_GCN(self.image_UI_matrix)
		return content_embedding1, content_embedding2

	def compute_gen_quality_metrics(self, n_proj=32, fid_eps=1e-6):
		"""
		计算生成兴趣表征质量指标：SWD、平均余弦、MSE、FID。
		原始 = GCN(norm_adj)，生成 = GCN(image_UI_matrix)。与 GenRecv2 同四指标便于对比。
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
		轨迹线性度 (Trajectory Linearity)：Flip 扩散逆过程轨迹的直线性，与 GenRecv2 同指标便于对比。
		使用 image 模态 denoise；按用户分批跑逆扩散，避免 (T+1, n_users, n_items) 整轨 OOM。
		"""
		steps = n_steps if n_steps is not None else self.sampling_steps
		steps = min(steps, self.steps)
		if steps < 1:
			return {}
		# 按用户分批，避免 CUDA OOM（全量 x_start 与 trajectory 过大）
		batch_size = 512
		x_start_full = self.R.to_dense().to(self.device).float()
		n_users = x_start_full.shape[0]
		linearities = []
		trajectory_for_viz = None
		for start in range(0, n_users, batch_size):
			end = min(start + batch_size, n_users)
			x_start = x_start_full[start:end]
			_, trajectory = self.diffusion_model.p_sample_with_trajectory(
				self.denoise_model_image, x_start, steps=steps, bayesian_samplinge_schedule=True
			)
			lin = trajectory_linearity(trajectory)
			linearities.append(lin.cpu().item())
			if return_trajectory_for_viz and trajectory_for_viz is None:
				trajectory_for_viz = trajectory.cpu()
		linearity_mean = sum(linearities) / len(linearities) if linearities else 0.0
		result = {"trajectory_linearity": float(linearity_mean)}
		if return_trajectory_for_viz and trajectory_for_viz is not None:
			result["trajectory"] = trajectory_for_viz
		return result


def denoise_norm(emb1, weight=0.1):
	'''
		embedding denoise function with SVD
	'''
	# nuclear norm denoising
	# print("weight:", weight, "weight.item:", weight.item())
	# weight = weight.cuda()
	nuclear_norm_emb1= torch.linalg.svdvals(emb1).sum()
	emb1_norm = emb1 - weight * nuclear_norm_emb1

	return emb1_norm


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


class FlipInterestDiffusion(nn.Module):
	def __init__(self, config, steps=5, base_temp=1.0):
		super(FlipInterestDiffusion, self).__init__()
		self.eps = 1e-8
		self.steps = steps
		self.base_temp = base_temp

		# hyper params
		self.sparse_temp = config['sparse_temp']
		self.ssl_gen1 = config['ssl_gen1']	
		self.ssl_gen2 = config['ssl_gen2']
		self.ssl_gen3 = config['ssl_gen3']

		# modality switcher
		self.visual_modality = config['visual_modality']
		self.text_modality = config['text_modality']
		self.audio_modality = config['audio_modality']



	def _compute_sparsity(self, x):
		return (x == 0).float().mean()

	def _auto_schedule_params(self, x_start):
		sparsity = self._compute_sparsity(x_start)
		gamma_start = 0.1 * (1 - sparsity) + 0.001
		gamma_end = gamma_start * 0.1
		epsilon_start = 0.005 * sparsity + 0.0001
		epsilon_end = epsilon_start * 0.1
		return gamma_start, gamma_end, epsilon_start, epsilon_end

	def get_cum(self, x_start):
		gamma_start, gamma_end, epsilon_start, epsilon_end = self._auto_schedule_params(x_start)
		gamma = torch.linspace(gamma_start, gamma_end, self.steps)
		epsilon = torch.linspace(epsilon_start, epsilon_end, self.steps)
		epsilon = torch.clamp(epsilon, max=0.01)
		gamma_cum = 1 - torch.cumprod(1 - gamma, dim=0)
		epsilon_cum = 1 - torch.cumprod(1 - epsilon, dim=0)
		return gamma_cum.to(x_start.device), epsilon_cum.to(x_start.device)

	@staticmethod
	def generate_custom_noise(x, temp_scale=1.0, mode='randn'):
		if mode == 'randn':
			mean = x.float().mean()
			var = x.float().var(unbiased=True)
			std = torch.sqrt(var + 1e-8) * temp_scale
			noise = torch.randn_like(x.float())
			return noise * std + mean
		if mode == 'rand':
			noise = torch.rand_like(x.float())
			return noise

	def q_sample(self, x_start, t, temp_scale=1.0):
		gamma_cum, epsilon_cum = self.get_cum(x_start)
		self.alpha_bar0_t = self._extract_into_tensor(gamma_cum, t, x_start.shape)
		self.alpha_bar1_t = self._extract_into_tensor(epsilon_cum, t, x_start.shape)
		noise = self.generate_custom_noise(x_start, temp_scale, mode='rand')

		flip_prob = torch.where(
			x_start == 0,
			torch.sigmoid((self.alpha_bar0_t - noise) * self.base_temp),
			torch.sigmoid((self.alpha_bar1_t - noise) * self.base_temp)
		)
		flip_mask = torch.bernoulli(flip_prob)
		x_t = x_start.clone()
		x_t[flip_mask.bool()] = 1 - x_t[flip_mask.bool()]
		return x_t

	def p_sample(self, model, x_start, steps, bayesian_samplinge_schedule=True):
		batch_size = x_start.shape[0]
		if steps == 0:
			x_t = x_start
		else:
			t = torch.tensor([steps - 1] * batch_size).to(x_start.device)
			x_t = self.q_sample(x_start, t)
		
		indices = list(range(self.steps))[::-1]
		for i in indices:
			t = torch.tensor([i] * x_t.shape[0]).to(x_start.device)
			logits, probs = self.p_interest_shift_probs(model, x_t, t)
			if bayesian_samplinge_schedule == True and i > 0:
				prev_alpha_bar0_t = self._extract_into_tensor(self.alpha_bar0_t, t-1, x_start.shape)
				prev_alpha_bar1_t = self._extract_into_tensor(self.alpha_bar1_t, t-1, x_start.shape)
				p0 = probs * (1 - prev_alpha_bar0_t) + (1 - probs) * prev_alpha_bar1_t
				p1 = probs * prev_alpha_bar0_t + (1 - probs) * (1 - prev_alpha_bar1_t)
				x_t = torch.bernoulli(p1 /(p0 + p1))
			else:
				x_t = torch.bernoulli(probs)
		return x_t, probs

	def p_sample_with_trajectory(self, model, x_start, steps, bayesian_samplinge_schedule=True):
		"""Run reverse diffusion and return (x_0, trajectory (T+1, B, D)) for trajectory_linearity."""
		batch_size = x_start.shape[0]
		if steps == 0:
			trajectory = x_start.unsqueeze(0).float()
			return x_start, trajectory
		t = torch.tensor([steps - 1] * batch_size).to(x_start.device)
		x_t = self.q_sample(x_start, t)
		trajectory_list = [x_t.detach().clone().float()]
		indices = list(range(self.steps))[::-1]
		for i in indices:
			t = torch.tensor([i] * x_t.shape[0]).to(x_start.device)
			logits, probs = self.p_interest_shift_probs(model, x_t, t)
			if bayesian_samplinge_schedule == True and i > 0:
				prev_alpha_bar0_t = self._extract_into_tensor(self.alpha_bar0_t, t - 1, x_start.shape)
				prev_alpha_bar1_t = self._extract_into_tensor(self.alpha_bar1_t, t - 1, x_start.shape)
				p0 = probs * (1 - prev_alpha_bar0_t) + (1 - probs) * prev_alpha_bar1_t
				p1 = probs * prev_alpha_bar0_t + (1 - probs) * (1 - prev_alpha_bar1_t)
				x_t = torch.bernoulli(p1 / (p0 + p1))
			else:
				x_t = torch.bernoulli(probs)
			trajectory_list.append(x_t.detach().clone().float())
		trajectory = torch.stack(trajectory_list, dim=0)
		return x_t, trajectory

	def training_losses(self, model, x_start, itmEmbeds, batch_index, model_feats, text_feats=None, audio_feats=None):
		pos_weight = torch.sum(1 - x_start) / (torch.sum(x_start) + 1e-8)
		batch_size = x_start.size(0)
		t = torch.randint(0, self.steps, (batch_size,)).long().to(x_start.device)
		x_t = self.q_sample(x_start, t)
		logits, probs = self.p_interest_shift_probs(model, x_t, t)
		
		# Focal Loss
		gamma = 2.0
		alpha = 0.25
		p = torch.sigmoid(logits)
		p = torch.clamp(p, min=1e-7, max=1-1e-7)
		adaptive_alpha = alpha * pos_weight.detach()
		pos_mask = x_start.float()
		neg_mask = 1 - pos_mask
		
		# Stable Focal Loss
		p_safe = torch.clamp(p, 1e-7, 1-1e-7)
		pos_loss = -adaptive_alpha * (1 - p_safe).pow(gamma) * pos_mask * torch.log(p_safe)
		neg_loss = -(1 - adaptive_alpha) * p_safe.pow(gamma) * neg_mask * torch.log(1 - p_safe)
		
		focal_loss = (pos_loss + neg_loss).sum() / (pos_mask.sum() + neg_mask.sum() + 1e-8)
		bce_loss = F.binary_cross_entropy_with_logits(
			logits, x_start.float(), 
			pos_weight=pos_weight.to(x_start.device) 
		)
		# Contrastive Loss
		gen_output, _ = self.p_sample(model=model, x_start=x_start, steps=self.steps, bayesian_samplinge_schedule=True)
		
		model_feat_embedding = torch.multiply(itmEmbeds, model_feats)
		model_feat_embedding_origin = torch.mm(x_start, model_feat_embedding)
		model_feat_embedding_diffusion = torch.mm(gen_output, model_feat_embedding)
		cl_loss = self.infoNCE_loss(model_feat_embedding_origin, model_feat_embedding_diffusion, self.sparse_temp)
		
		text_model_feat_embedding = torch.multiply(itmEmbeds, text_feats)
		text_model_feat_embedding_origin = torch.mm(x_start, text_model_feat_embedding)
		text_model_feat_embedding_diffusion = torch.mm(gen_output, text_model_feat_embedding)
		cl_loss_text = self.infoNCE_loss(text_model_feat_embedding_origin, text_model_feat_embedding_diffusion, self.sparse_temp)

		if self.audio_modality:
			audio_model_feat_embedding =  torch.multiply(itmEmbeds, audio_feats)
			audio_model_feat_embedding_origin = torch.mm(x_start, audio_model_feat_embedding)
			audio_model_feat_embedding_diffusion = torch.mm(gen_output, audio_model_feat_embedding)
			cl_loss_audio = self.infoNCE_loss(audio_model_feat_embedding_origin, audio_model_feat_embedding_diffusion, self.sparse_temp)

		kl_loss = self._calc_kl_divergence(x_start, x_t, t, probs)
		curriculum_weight = torch.clamp(t.float() / self.steps, 0, 0.5)
		kl_loss = (curriculum_weight * kl_loss).mean()
		

		if self.audio_modality:
			total_loss = focal_loss +  kl_loss + self.ssl_gen1 * cl_loss   + self.ssl_gen2 * cl_loss_text + self.ssl_gen3 * cl_loss_audio 
		else:
			# total_loss = focal_loss +  kl_loss + self.ssl_gen1 * cl_loss  + self.ssl_gen2 * cl_loss_text 
			# total_loss = bce_loss +  kl_loss + 0.01 * cl_loss
			total_loss = focal_loss + kl_loss + self.ssl_gen1 * cl_loss + self.ssl_gen2 * cl_loss_text

		return total_loss

	def _calc_kl_divergence(self, x0, xt, t, probs):
		post_probs = self._true_posterior(x0, xt, t).detach()
		post_probs = torch.clamp(post_probs, self.eps, 1-self.eps)
		probs = torch.clamp(probs.detach(), self.eps, 1-self.eps)
		
		# Stable KL
		kl = post_probs * (torch.log(post_probs + 1e-10) - torch.log(probs + 1e-10))
		kl += (1 - post_probs) * (torch.log(1 - post_probs + 1e-10) - torch.log(1 - probs + 1e-10))
		return kl.mean(dim=1)

	def _true_posterior(self, x0, xt, t):
		alpha0 = self._extract_into_tensor(self.alpha_bar0_t, t, x0.shape)
		alpha1 = self._extract_into_tensor(self.alpha_bar1_t, t, x0.shape)
		case0 = (x0 == 0).float() * (1 - alpha0)
		case1 = (x0 == 1).float() * alpha1
		numerator = case0 + case1
		denom0 = (x0 == 0).float() * (1 - alpha0 + alpha1)
		denom1 = (x0 == 1).float() * (alpha0 + 1 - alpha1)
		denominator = denom0 + denom1
		return numerator / (denominator + self.eps)

	def p_interest_shift_probs(self, model, x_t, t):
		logits = model(x_t, t)
		probs = torch.sigmoid(logits)
		return logits, probs

	def _extract_into_tensor(self, arr, timesteps, broadcast_shape):
		arr = arr.to(timesteps.device)
		res = arr[timesteps].float()
		while len(res.shape) < len(broadcast_shape):
			res = res[..., None]
		return res.expand(broadcast_shape)

	def infoNCE_loss(self, view1, view2, temperature):
		view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
		pos_score = torch.sum((view1 * view2), dim=-1)
		pos_score = torch.exp(pos_score / temperature)
		neg_score = (view1 @ view2.T) / temperature
		neg_score = torch.exp(neg_score).sum(dim=1)
		contrast_loss = -1 * torch.log(pos_score / neg_score).mean()
		return contrast_loss

class ModalDenoiseTransformer(nn.Module):
	def __init__(self, in_dims, out_dims, emb_size, nhead=8, num_layers=6, dim_feedforward=512, dropout=0.2):
		super().__init__()
		self.in_dims = in_dims
		self.out_dims = out_dims
		self.time_emb_dim = emb_size
		
		self.time_emb = nn.Sequential(
			nn.Linear(emb_size, 4*emb_size),
			nn.SiLU(),
			nn.Linear(4*emb_size, emb_size)
		)
		self.emb_layer = nn.Linear(self.time_emb_dim, self.time_emb_dim)
		self.input_proj = nn.Linear(in_dims + emb_size, dim_feedforward)
		
		decoder_layer = nn.TransformerDecoderLayer(
			d_model=dim_feedforward,
			nhead=nhead,
			dim_feedforward=dim_feedforward,
			dropout=dropout,
			batch_first=True)
		self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

		self.output_proj = nn.Sequential(
			nn.Linear(dim_feedforward, dim_feedforward//2),
			nn.LayerNorm(dim_feedforward//2),
			nn.GELU(),
			nn.Linear(dim_feedforward//2, out_dims))

		self.adaLN_modulation = nn.Sequential(
			nn.SiLU(),
			nn.Linear(emb_size, 2*dim_feedforward))

		self.apply(self._init_weights)

	def _init_weights(self, module):
		if isinstance(module, nn.Linear):
			nn.init.xavier_uniform_(module.weight)
			if module.bias is not None:
				nn.init.constant_(module.bias, 0.01)

	def forward(self, x, timesteps):
		freqs = torch.exp(-math.log(10000) * torch.arange(start=0, end=self.time_emb_dim//2, dtype=torch.float32) / (self.time_emb_dim//2)).to(x.device)
		temp = timesteps[:, None].float() * freqs[None]
		time_emb = torch.cat([torch.cos(temp), torch.sin(temp)], dim=-1)
		if self.time_emb_dim % 2:
			time_emb = torch.cat([time_emb, torch.zeros_like(time_emb[:, :1])], dim=-1)
		time_emb = self.emb_layer(time_emb.to(x.device))

		h = torch.cat([x, time_emb], dim=-1)
		h = self.input_proj(h)
		h = h.unsqueeze(1)
		shift, scale = self.adaLN_modulation(time_emb).chunk(2, dim=1)
		h = h * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
	
		memory = torch.zeros_like(h) 
		out = self.transformer_decoder(tgt=h, memory=memory)

		out = out.squeeze(1)
		out = self.output_proj(out)
		return out
