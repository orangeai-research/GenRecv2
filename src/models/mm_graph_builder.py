# coding: utf-8
"""
模块一: 图像模态和文本模态的 KNN 图构建（与 backbone 共用）
- 构建/加载 image_adj、text_adj
- 边集取交、边权加权得到 item-item 图 ii_adj
- 与 user-item 交互合并得到 norm_adj、R
供 BACKBONE、RFBM3 等模型复用。
"""

import os
import numpy as np
import scipy.sparse as sp
import torch

from utils.utils import build_sim, build_knn_normalized_graph


def build_image_text_knn_adjs(
    dataset_path,
    knn_k,
    sparse,
    device,
    image_embedding_weight,
    text_embedding_weight,
):
    """
    构建或加载图像/文本 KNN 归一化邻接（与 backbone 模块一一致）。
    Args:
        dataset_path: 数据集目录
        knn_k: KNN 的 K
        sparse: 是否稀疏
        device: torch device
        image_embedding_weight: (n_items, dim) 或 None
        text_embedding_weight: (n_items, dim) 或 None
    Returns:
        image_adj: torch sparse (n_items, n_items) 或 None
        text_adj: torch sparse (n_items, n_items) 或 None
    """
    image_adj_file = os.path.join(dataset_path, "image_adj_{}_{}.pt".format(knn_k, sparse))
    text_adj_file = os.path.join(dataset_path, "text_adj_{}_{}.pt".format(knn_k, sparse))

    image_adj, text_adj = None, None

    if image_embedding_weight is not None:
        if os.path.exists(image_adj_file):
            image_adj = torch.load(image_adj_file)
        else:
            image_adj = build_sim(image_embedding_weight.detach())
            image_adj = build_knn_normalized_graph(
                image_adj, topk=knn_k, is_sparse=sparse, norm_type="sym"
            )
            torch.save(image_adj, image_adj_file)
        image_adj = image_adj.to(device)

    if text_embedding_weight is not None:
        if os.path.exists(text_adj_file):
            text_adj = torch.load(text_adj_file)
        else:
            text_adj = build_sim(text_embedding_weight.detach())
            text_adj = build_knn_normalized_graph(
                text_adj, topk=knn_k, is_sparse=sparse, norm_type="sym"
            )
            torch.save(text_adj, text_adj_file)
        text_adj = text_adj.to(device)

    return image_adj, text_adj


def get_ii_adj_intersection_attention(image_adj, text_adj, alpha, n_items):
    """
    边集取交、边权加权：仅保留图像与文本 KNN 都有的边，边权 alpha*w_img + (1-alpha)*w_txt。
    Args:
        image_adj: torch sparse (n_items, n_items)
        text_adj: torch sparse (n_items, n_items)
        alpha: mm_image_weight
        n_items: 物品数
    Returns:
        ii_adj: scipy.sparse.coo_matrix (n_items, n_items), float
    """
    def _adj_to_edge_dict(adj):
        adj = adj.coalesce()
        inds = adj.indices().cpu().numpy()
        vals = adj.values().cpu().numpy()
        out = {}
        for k in range(inds.shape[1]):
            r, c = int(inds[0, k]), int(inds[1, k])
            try:
                w = float(vals[k])
                out[(r, c)] = 0.0 if w != w else w
            except (TypeError, ValueError):
                out[(r, c)] = 0.0
        return out

    img_edges = _adj_to_edge_dict(image_adj)
    txt_edges = _adj_to_edge_dict(text_adj)

    rows, cols, values = [], [], []
    for i in range(n_items):
        img_nbrs = {j for (r, j) in img_edges if r == i}
        txt_nbrs = {j for (r, j) in txt_edges if r == i}
        inter = (img_nbrs & txt_nbrs) - {i}
        for j in inter:
            ji, jj = int(i), int(j)
            w_img = img_edges.get((ji, jj), 0.0) or 0.0
            w_txt = txt_edges.get((ji, jj), 0.0) or 0.0
            w = alpha * w_img + (1.0 - alpha) * w_txt
            rows.append(i)
            cols.append(j)
            values.append(w)

    if len(rows) == 0:
        return sp.coo_matrix((n_items, n_items), dtype=np.float32)
    return sp.coo_matrix(
        (np.array(values, dtype=np.float32), (np.array(rows), np.array(cols))),
        shape=(n_items, n_items),
        dtype=np.float32,
    )




def get_adj_mat_with_ii(interaction_matrix, item_adj, n_users, n_items):
    """
    User-Item 交互 + Item-Item 图合并为整图并对称归一化。
    Args:
        interaction_matrix: scipy coo 或 lil，user-item 交互
        item_adj: scipy sparse (n_items, n_items)，item-item 边
        n_users, n_items: 数量
    Returns:
        norm_adj: 归一化后的整图 (scipy csr)，shape (n_users+n_items, n_users+n_items)
        R: 归一化后的 user-item 块 (scipy)，shape (n_users, n_items)
    """
    adj_mat = sp.dok_matrix((n_users + n_items, n_users + n_items), dtype=np.float32)
    adj_mat = adj_mat.tolil()

    R = interaction_matrix.tolil()
    adj_mat[:n_users, n_users:] = R
    adj_mat[n_users:, :n_users] = R.T
    adj_mat[n_users:, n_users:] = item_adj

    adj_mat = adj_mat.todok()

    def normalized_adj_single(adj):
        rowsum = np.array(adj.sum(1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.0
        d_mat_inv = sp.diags(d_inv)
        norm_adj = d_mat_inv.dot(adj)
        norm_adj = norm_adj.dot(d_mat_inv)
        return norm_adj.tocoo()

    norm_adj_mat = normalized_adj_single(adj_mat)
    norm_adj_mat = norm_adj_mat.tolil()
    R_block = norm_adj_mat[:n_users, n_users:]

    return norm_adj_mat.tocsr(), R_block


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Scipy sparse -> torch sparse FloatTensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
