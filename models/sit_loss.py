import torch
import torch.nn as nn
import torch.nn.functional as F


def pairwise_distances(x, y=None):
    if y is None:
        y = x
    x_norm = (x ** 2).sum(1).view(-1, 1)
    y_norm = (y ** 2).sum(1).view(1, -1)
    dist = x_norm + y_norm - 2.0 * torch.mm(x, torch.transpose(y, 0, 1))
    return torch.clamp(dist, 0.0, float('inf'))


def get_kernel(X, sigma=None):
    dist = pairwise_distances(X)
    if sigma is None:
        sigma = torch.median(dist[dist > 0])
        if sigma == 0:
            sigma = 1.0
    return torch.exp(-dist / (2.0 * sigma))


class HSIC_Loss(nn.Module):
    """
    希尔伯特-施密特独立性准则 (HSIC) 解耦损失。
    通过惩罚不同模态间的核关联度，强制特征空间正交化，消除信息冗余。
    """
    def forward(self, Z_A, Z_B):
        n = Z_A.size(0)
        if n <= 1:
            return torch.tensor(0.0, device=Z_A.device, requires_grad=Z_A.requires_grad)

        K = get_kernel(Z_A)
        L = get_kernel(Z_B)

        H = torch.eye(n, device=Z_A.device) - (1.0 / n) * torch.ones(n, n, device=Z_A.device)
        Kc = torch.mm(H, torch.mm(K, H))
        Lc = torch.mm(H, torch.mm(L, H))

        hsic = torch.trace(torch.mm(Kc, Lc)) / ((n - 1) ** 2)
        return torch.clamp(hsic, min=0.0, max=10.0)


class SurvNCE_Loss(nn.Module):
    """
    生存期约束的多模态互信息对比损失 (Survival-Aware InfoNCE)。
    强制模型将生存周期高度相似的病人的模态特征在跨模态流形中拉近。
    """
    def __init__(self, temperature=0.1, sigma_min=1e-3):
        super(SurvNCE_Loss, self).__init__()
        self.temp = temperature
        self.sigma_min = sigma_min
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, z_i, z_j, time, event):
        B = z_i.size(0)
        if B <= 1:
            return torch.tensor(0.0, device=z_i.device, requires_grad=z_i.requires_grad)

        z_i_norm = F.normalize(z_i, dim=1)
        z_j_norm = F.normalize(z_j, dim=1)
        sim_matrix = torch.matmul(z_i_norm, z_j_norm.T) / self.temp

        # 时间相似度权重
        time_diff = torch.abs(time.view(-1, 1) - time.view(1, -1))
        sigma_time = torch.std(time) + self.sigma_min
        sigma_time = torch.clamp(sigma_time, min=self.sigma_min)
        surv_weight = torch.exp(-(time_diff ** 2) / (2 * sigma_time ** 2))
        surv_weight = surv_weight / (surv_weight.sum(dim=1, keepdim=True) + 1e-8)

        # 对角线置零（避免自身配对）
        mask = torch.eye(B, device=z_i.device).bool()
        sim_matrix = sim_matrix.masked_fill(mask, float('-inf'))

        log_softmax = F.log_softmax(sim_matrix, dim=1)
        loss = -(surv_weight * log_softmax).sum(dim=1).mean()

        if torch.isnan(loss) or torch.isinf(loss):
            return torch.tensor(0.0, device=z_i.device, requires_grad=True)

        return torch.clamp(loss, min=0.0, max=50.0)


class SlicedWassersteinLoss(nn.Module):
    """
    基于最优传输的切片 Wasserstein 距离。
    用于衡量生成特征与真实特征在全局分布上的差异。
    """
    def __init__(self, num_projections=50):
        super(SlicedWassersteinLoss, self).__init__()
        self.num_projections = num_projections

    def forward(self, x, y):
        if x is None or y is None:
            return torch.tensor(0.0, device=x.device if x is not None else (y.device if y is not None else 'cpu'),
                               requires_grad=True)
        B, D = x.shape
        projections = torch.randn(D, self.num_projections, device=x.device)
        projections = F.normalize(projections, p=2, dim=0)

        x_proj = torch.matmul(x, projections)
        y_proj = torch.matmul(y, projections)

        x_proj_sorted, _ = torch.sort(x_proj, dim=0)
        y_proj_sorted, _ = torch.sort(y_proj, dim=0)

        wasserstein_dist = torch.mean((x_proj_sorted - y_proj_sorted) ** 2)
        return torch.clamp(wasserstein_dist, min=0.0, max=100.0)
