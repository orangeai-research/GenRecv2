
import torch
import torch.nn as nn
import torch.nn.functional as F

class RectifiedFlowAligner(nn.Module):
    '''
    Desc@: Rectified Flow for Cross-Modal Alignment
    Args:
        dim: dimension of the input and output
        hidden_dim: dimension of the hidden layer
    Returns:
        output: the output of the Rectified Flow
    Version: v1 
    '''
    def __init__(self, dim, hidden_dim=None):
        super(RectifiedFlowAligner, self).__init__()
        if hidden_dim is None:
            hidden_dim = dim
        self.net = nn.Sequential(
            nn.Linear(dim + 1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, dim)
        )
    
    def forward(self, x, t):
        xt = torch.cat([x, t], dim=-1)
        return self.net(xt)

    def compute_loss(self, x0, x1):
        batch_size = x0.size(0)
        t = torch.rand(batch_size, 1, device=x0.device)
        z_t = t * x1 + (1 - t) * x0
        v_target = x1 - x0 # 目标传输向量v_target：图像到文本的真实差值（最优传输方向）跨模态语义gap
        v_pred = self.forward(z_t, t) # 模型预测的 t 时刻传输方向
        loss = F.mse_loss(v_pred, v_target)
        return loss


class RectifiedFlow_Cross_Modality_Alignment(nn.Module):
    '''
    Version 2
    Rectified Flow for Cross-Modal Alignment with Feature Projection
    Args:
        dim: dimension of the input and output (target dimension, e.g. 64)
        input_dim1: dimension of input modality 1 (e.g. Image 4096)
        input_dim2: dimension of input modality 2 (e.g. Text 384)
        hidden_dim: dimension of the hidden layer
    Returns:
        output: the output of the Rectified Flow
    '''
    def __init__(self, dim, input_dim1=None, input_dim2=None, hidden_dim=None):
        super(RectifiedFlow_Cross_Modality_Alignment, self).__init__()
        
        # 定义投影层，如果输入维度不等于目标维度
        self.proj1 = nn.Identity()
        if input_dim1 is not None and input_dim1 != dim:
            self.proj1 = nn.Sequential(
                nn.Linear(input_dim1, dim),
                nn.Tanh()
            )
            
        self.proj2 = nn.Identity()
        if input_dim2 is not None and input_dim2 != dim:
            self.proj2 = nn.Sequential(
                nn.Linear(input_dim2, dim),
                nn.Tanh()
            )

        if hidden_dim is None:
            hidden_dim = dim
            
        self.net = nn.Sequential(
            nn.Linear(dim + 1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, dim)
        )
    
    def forward(self, x, t):
        xt = torch.cat([x, t], dim=-1)
        return self.net(xt)

    def compute_loss(self, x0, x1):
        # x0: Modality 1 (e.g. Image)
        # x1: Modality 2 (e.g. Text)
        
        # 1. 对齐维度
        x0 = self.proj1(x0)
        x1 = self.proj2(x1)
        
        batch_size = x0.size(0)
        t = torch.rand(batch_size, 1, device=x0.device)
        z_t = t * x1 + (1 - t) * x0
        v_target = x1 - x0 # 目标传输向量v_target
        v_pred = self.forward(z_t, t) # 模型预测的 t 时刻传输方向
        loss = F.mse_loss(v_pred, v_target)
        return loss