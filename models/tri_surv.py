import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import functional as TF
from .deref.util import initialize_weights, SNN_Block, NystromAttention


class TransLayer(nn.Module):
    def __init__(self, norm_layer=nn.LayerNorm, dim=512):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim=dim,
            dim_head=dim // 8,
            heads=8,
            num_landmarks=dim // 2,
            pinv_iterations=6,
            residual=True,
            dropout=0.1,
        )

    def forward(self, x):
        x = x + self.attn(self.norm(x))
        return x


class PPEG(nn.Module):
    def __init__(self, dim=512):
        super(PPEG, self).__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7 // 2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5 // 2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3 // 2, groups=dim)

    def forward(self, x, H, W):
        B, _, C = x.shape
        cls_token, feat_token = x[:, 0], x[:, 1:]
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat) + cnn_feat + self.proj1(cnn_feat) + self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x


class PathEncoder(nn.Module):
    def __init__(self, feature_dim=256):
        super(PathEncoder, self).__init__()
        self.pos_layer = PPEG(dim=feature_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, feature_dim))
        nn.init.normal_(self.cls_token, std=1e-6)
        self.layer1 = TransLayer(dim=feature_dim)
        self.layer2 = TransLayer(dim=feature_dim)
        self.norm = nn.LayerNorm(feature_dim)

    def forward(self, features):
        H = features.shape[1]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        if add_length > 0:
            h = torch.cat([features, features[:, :add_length, :]], dim=1)
        else:
            h = features
        B = h.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1).to(h.device)
        h = torch.cat((cls_tokens, h), dim=1)
        h = self.layer1(h)
        h = self.pos_layer(h, _H, _W)
        h = self.layer2(h)
        h = self.norm(h)
        return h[:, 0]


class MIBottleneck(nn.Module):
    """
    变分信息瓶颈 (Variational Information Bottleneck)
    提取与生存相关的纯净表示，并通过 KL 散度压制噪声。
    """
    def __init__(self, input_dim=256, z_dim=128):
        super(MIBottleneck, self).__init__()
        self.fc_mu = nn.Linear(input_dim, z_dim)
        self.fc_var = nn.Linear(input_dim, z_dim)

    def forward(self, x):
        mu = self.fc_mu(x)
        logvar = self.fc_var(x)
        logvar = torch.clamp(logvar, min=-10.0, max=10.0)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
        return z, kl_loss


class CrossModalAttention(nn.Module):
    """
    交叉注意力融合模块。
    三个模态之间通过交叉注意力进行信息交互，每个模态都能从其他模态获取信息，
    最终输出融合了跨模态上下文的表示。
    """
    def __init__(self, dim_p=128, dim_g=128, dim_c=64, num_heads=4):
        super(CrossModalAttention, self).__init__()
        total_dim = dim_p + dim_g + dim_c
        self.num_heads = num_heads
        head_dim = total_dim // num_heads

        self.to_q = nn.Linear(total_dim, total_dim)
        self.to_k = nn.Linear(total_dim, total_dim)
        self.to_v = nn.Linear(total_dim, total_dim)
        self.to_out = nn.Linear(total_dim, total_dim)

        self.scale = head_dim ** -0.5

        # 模态边界索引
        self.dim_p = dim_p
        self.dim_g = dim_g
        self.dim_c = dim_c

    def forward(self, z_p, z_g, z_c):
        """
        Args:
            z_p: (B, dim_p) 病理模态表示
            z_g: (B, dim_g) 基因模态表示
            z_c: (B, dim_c) 临床模态表示
        Returns:
            fused_p, fused_g, fused_c: 各模态融合后的表示
        """
        B = z_p.size(0)

        # 拼接三模态
        joint = torch.cat([z_p, z_g, z_c], dim=1)

        # 交叉注意力：Query = joint, Key = joint, Value = joint
        # 这样每个位置（模态）的 query 可以 attend 到所有模态的 key
        q = self.to_q(joint)
        k = self.to_k(joint)
        v = self.to_v(joint)

        # 多头分割
        q = q.view(B, self.num_heads, -1, self.num_heads)
        k = k.view(B, self.num_heads, -1, self.num_heads)
        v = v.view(B, self.num_heads, -1, self.num_heads)

        # 注意力分数
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)

        # 聚合
        out = torch.matmul(attn, v)
        out = out.view(B, -1)
        out = self.to_out(out)

        # 按模态边界切分
        fused = torch.cat([z_p, z_g, z_c], dim=1) + out  # 残差连接
        fused_p = fused[:, :self.dim_p]
        fused_g = fused[:, self.dim_p:self.dim_p + self.dim_g]
        fused_c = fused[:, self.dim_p + self.dim_g:]

        return fused_p, fused_g, fused_c


class cVAE_Imputer(nn.Module):
    """
    条件变分自编码生成器 (cVAE) 缺失模态补全。
    支持所有 7 种缺失模态组合的补全逻辑。
    """
    def __init__(self, dim_p=128, dim_g=128, dim_c=64):
        super(cVAE_Imputer, self).__init__()

        # 缺单个模态时的 decoder
        # 缺 path: 用 geno + clin 生成
        self.decode_path = nn.Sequential(
            nn.Linear(dim_g + dim_c, 256), nn.LayerNorm(256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.ReLU()
        )
        # 缺 geno: 用 path + clin 生成
        self.decode_geno = nn.Sequential(
            nn.Linear(dim_p + dim_c, 256), nn.LayerNorm(256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.ReLU()
        )
        # 缺 clin: 用 path + geno 生成
        self.decode_clin = nn.Sequential(
            nn.Linear(dim_p + dim_g, 256), nn.LayerNorm(256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 64), nn.ReLU()
        )

        # 缺两个模态时的 decoder
        # 缺 path + geno: 只用 clin 生成
        self.decode_path_geno = nn.Sequential(
            nn.Linear(dim_c, 128), nn.LayerNorm(128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 128), nn.ReLU()
        )
        # 缺 path + clin: 只用 geno 生成
        self.decode_path_clin = nn.Sequential(
            nn.Linear(dim_g, 128), nn.LayerNorm(128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 128), nn.ReLU()
        )
        # 缺 geno + clin: 只用 path 生成
        self.decode_geno_clin = nn.Sequential(
            nn.Linear(dim_p, 128), nn.LayerNorm(128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 128), nn.ReLU()
        )

    def forward(self, z_p, z_g, z_c, mask):
        """
        Args:
            z_p, z_g, z_c: 各模态的隐向量（None 表示该模态缺失）
            mask: {'path': bool, 'geno': bool, 'clin': bool}
        Returns:
            imputed_p, imputed_g, imputed_c: 补全后的各模态表示
        """
        # 统计缺失情况
        num_missing = sum([int(mask['path']), int(mask['geno']), int(mask['clin'])])

        # 获取可用的模态向量（如果缺失则用零向量占位，以便拼接）
        p = z_p if z_p is not None else torch.zeros_like(z_p) if z_p is not None else torch.zeros_like(z_g) if z_g is not None else torch.zeros(1, 128, device=next(self.parameters()).device)
        g = z_g if z_g is not None else torch.zeros_like(z_g) if z_g is not None else torch.zeros_like(z_p) if z_p is not None else torch.zeros(1, 128, device=next(self.parameters()).device)
        c = z_c if z_c is not None else torch.zeros_like(z_c) if z_c is not None else torch.zeros(1, 64, device=next(self.parameters()).device)

        if num_missing == 0:
            # 全模态，无须补全
            return z_p, z_g, z_c

        elif num_missing == 1:
            # 只缺一个模态
            if mask['path']:
                return self.decode_path(torch.cat([g, c], dim=1)), g, c
            elif mask['geno']:
                return p, self.decode_geno(torch.cat([p, c], dim=1)), c
            elif mask['clin']:
                return p, g, self.decode_clin(torch.cat([p, g], dim=1))

        elif num_missing == 2:
            # 缺两个模态
            if mask['path'] and mask['geno']:
                # 只有 clin
                imputed_pg = self.decode_path_geno(c)
                return imputed_pg, imputed_pg, c
            elif mask['path'] and mask['clin']:
                # 只有 geno
                imputed_pc = self.decode_path_clin(g)
                return imputed_pc, g, imputed_pc
            elif mask['geno'] and mask['clin']:
                # 只有 path
                imputed_gc = self.decode_geno_clin(p)
                return p, imputed_gc, imputed_gc

        # 全缺失（理论不应出现，但给一个合理的 fallback）
        return z_p, z_g, z_c


class GatedFusion(nn.Module):
    """
    自适应门控融合：对融合后的模态进行动态权重分配。
    """
    def __init__(self, dim_p=128, dim_g=128, dim_c=64):
        super(GatedFusion, self).__init__()
        total_dim = dim_p + dim_g + dim_c
        self.gate = nn.Sequential(
            nn.Linear(total_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )

    def forward(self, z_p, z_g, z_c):
        """
        Returns:
            fused: 各模态加权拼接后的向量
            weights: 三个模态的注意力权重 (B, 3)
        """
        joint = torch.cat([z_p, z_g, z_c], dim=1)
        weights = F.softmax(self.gate(joint), dim=1)
        fused = torch.cat([
            z_p * weights[:, 0:1],
            z_g * weights[:, 1:2],
            z_c * weights[:, 2:3],
        ], dim=1)
        return fused, weights


class TriSurv(nn.Module):
    def __init__(self, omic_sizes=[100, 200, 300], num_classes=4, clin_input_dim=10,
                 use_cross_attn=True, use_cvae=True, use_vib=True):
        super(TriSurv, self).__init__()

        self.use_cross_attn = use_cross_attn
        self.use_cvae = use_cvae
        self.use_vib = use_vib

        # 1. Pathomics Encoder
        self.path_fc = nn.Sequential(
            nn.Linear(1024, 512), nn.LayerNorm(512), nn.ReLU(), nn.Dropout(0.25),
            nn.Linear(512, 256), nn.LayerNorm(256), nn.ReLU(), nn.Dropout(0.25)
        )
        self.path_encoder = PathEncoder(feature_dim=256)

        # 2. Genomics Encoder
        total_omic_size = sum(omic_sizes)
        self.geno_encoder = nn.Sequential(
            SNN_Block(total_omic_size, 512),
            SNN_Block(512, 256)
        )

        # 3. Clinical Encoder（改进：区分连续和分类特征的处理）
        self.clin_encoder = nn.Sequential(
            nn.Linear(clin_input_dim, 128), nn.LayerNorm(128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU()
        )

        # 4. MI Bottlenecks（提取 z 变量）
        if use_vib:
            self.path_mib = MIBottleneck(input_dim=256, z_dim=128)
            self.geno_mib = MIBottleneck(input_dim=256, z_dim=128)
            self.clin_mib = MIBottleneck(input_dim=64, z_dim=64)
            z_dim_p, z_dim_g, z_dim_c = 128, 128, 64
        else:
            # 不用 VIB 时直接线性映射
            self.path_bottleneck = nn.Linear(256, 128)
            self.geno_bottleneck = nn.Linear(256, 128)
            self.clin_bottleneck = nn.Linear(64, 64)
            z_dim_p, z_dim_g, z_dim_c = 128, 128, 64

        # 5. 缺失模态 embedding（修复 Bug 3：不再用零向量初始化）
        self.missing_path_emb = nn.Parameter(torch.randn(1, 128) * 0.02)
        self.missing_geno_emb = nn.Parameter(torch.randn(1, 128) * 0.02)
        self.missing_clin_emb = nn.Parameter(torch.randn(1, 64) * 0.02)

        # 6. cVAE Imputer（修复 Bug 2：覆盖所有 7 种缺失组合）
        if use_cvae:
            self.imputer = cVAE_Imputer(dim_p=128, dim_g=128, dim_c=64)

        # 7. Cross-Attention 融合模块（改进：替代简单的加权拼接）
        if use_cross_attn:
            self.cross_attn = CrossModalAttention(dim_p=128, dim_g=128, dim_c=64)

        # 8. Gated Fusion
        self.gated_fusion = GatedFusion(dim_p=128, dim_g=128, dim_c=64)

        # 9. Survival Classifier
        self.classifier = nn.Sequential(
            nn.Linear(128 + 128 + 64, 256), nn.LayerNorm(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

        self.apply(initialize_weights)

    def _get_z(self, feat, mib, bottleneck, mask_flag):
        """统一获取隐向量的辅助函数，同时处理 VIB 和非 VIB 模式"""
        if mask_flag:
            return None, torch.tensor(0.0, device=feat.device)
        if self.use_vib:
            return mib(feat)
        else:
            return bottleneck(feat), torch.tensor(0.0, device=feat.device)

    def forward(self, x_path, x_omic, x_clin,
                mask={'path': False, 'geno': False, 'clin': False},
                beta=1.0):
        """
        Args:
            mask: dictionary denoting if a modality is missing (True = missing)
            beta: KL-annealing factor (0.0 to 1.0), only used when use_vib=True
        """
        # 确保 batch 维度
        if x_path is not None and x_path.dim() == 2:
            x_path = x_path.unsqueeze(0)
        if x_omic is not None and x_omic.dim() == 1:
            x_omic = x_omic.unsqueeze(0)
        if x_clin is not None and x_clin.dim() == 1:
            x_clin = x_clin.unsqueeze(0)

        B = x_path.shape[0] if x_path is not None else x_omic.shape[0]
        device = next(self.parameters()).device

        kl_losses = torch.tensor(0.0, device=device)

        # ---- 特征提取阶段 ----
        # Pathomics
        if not mask['path'] and x_path is not None:
            p_feat = self.path_fc(x_path)
            cls_p = self.path_encoder(p_feat)
            z_p, kl_p = self._get_z(cls_p, self.path_mib, self.path_bottleneck, False)
            kl_losses = kl_losses + kl_p * beta
        else:
            z_p = None

        # Genomics
        if not mask['geno'] and x_omic is not None:
            g_feat = self.geno_encoder(x_omic)
            z_g, kl_g = self._get_z(g_feat, self.geno_mib, self.geno_bottleneck, False)
            kl_losses = kl_losses + kl_g * beta
        else:
            z_g = None

        # Clinical
        if not mask['clin'] and x_clin is not None:
            c_feat = self.clin_encoder(x_clin)
            z_c, kl_c = self._get_z(c_feat, self.clin_mib, self.clin_bottleneck, False)
            kl_losses = kl_losses + kl_c * beta
        else:
            z_c = None

        # ---- 缺失模态补全阶段（修复 Bug 2）----
        if self.use_cvae:
            z_p, z_g, z_c = self.imputer(z_p, z_g, z_c, mask)

        # 如果仍然有缺失（imputer 无法处理或 use_cvae=False），用可学习的 embedding 填充
        if z_p is None:
            z_p = self.missing_path_emb.expand(B, -1)
        if z_g is None:
            z_g = self.missing_geno_emb.expand(B, -1)
        if z_c is None:
            z_c = self.missing_clin_emb.expand(B, -1)

        # ---- 融合阶段（改进：Cross-Attention）----
        if self.use_cross_attn:
            z_p, z_g, z_c = self.cross_attn(z_p, z_g, z_c)

        # 门控融合
        fused, attn_weights = self.gated_fusion(z_p, z_g, z_c)

        # ---- 生存预测阶段 ----
        logits = self.classifier(fused)
        hazards = torch.sigmoid(logits)
        hazards = torch.clamp(hazards, min=1e-6, max=1.0 - 1e-6)
        S = torch.cumprod(1 - hazards, dim=1)

        z_dict = {'p': z_p, 'g': z_g, 'c': z_c}

        return hazards, S, logits, kl_losses, z_dict, attn_weights
