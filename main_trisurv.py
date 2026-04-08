import os
import csv
import time
import random
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from sksurv.metrics import concordance_index_censored

from datasets.dataset_survival import Generic_MIL_Survival_Dataset
from utils.options import parse_args
from utils.util import get_split_loader, set_seed

from models.tri_surv import TriSurv
from models.sit_loss import HSIC_Loss, SurvNCE_Loss, SlicedWassersteinLoss


class NLLSurvLoss(nn.Module):
    """负对数似然生存损失。"""
    def __init__(self, alpha=0.15):
        super().__init__()
        self.alpha = alpha

    def forward(self, hazards, S, Y, c, alpha=None):
        Y = Y.view(-1, 1)
        c = c.view(-1, 1)
        if alpha is None:
            alpha = self.alpha

        y_c = torch.gather(S, 1, Y).view(-1, 1)
        y_c_prev = torch.gather(S, 1, (Y - 1).clamp(min=0)).view(-1, 1)
        y_c_prev[Y == 0] = 1.0

        loss = -(1 - c) * torch.log(y_c_prev - y_c + 1e-7) - c * torch.log(y_c + 1e-7)
        return torch.mean(loss)


class EarlyStopping:
    """早停机制：监控验证集指标，连续 patience 个 epoch 未提升则停止。"""
    def __init__(self, patience=10, mode='max'):
        self.patience = patience
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0

    def __call__(self, score, epoch):
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == 'max':
            improved = score > self.best_score
        else:
            improved = score < self.best_score

        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        return False


class TriSurvEngine:
    """Tri-Surv 训练引擎。"""
    def __init__(self, args, fold_dir, fold):
        self.args = args
        self.fold_dir = fold_dir
        self.fold = fold
        self.best_c_index = 0.0
        self.best_epoch = 0
        self.accumulation_steps = args.accumulation_steps
        self.loss_cox_fn = NLLSurvLoss()
        self.loss_hsic_fn = HSIC_Loss()
        self.loss_surv_nce_fn = SurvNCE_Loss(temperature=0.1)
        self.loss_ot_fn = SlicedWassersteinLoss(num_projections=50)

        # 早停
        self.early_stopping = EarlyStopping(patience=args.patience, mode='max')

        # 梯度累积计数器
        self.accum_count = 0

    def train_epoch(self, model, loader, optimizer, epoch, writer, global_step):
        model.train()
        total_loss = 0.0
        total_l_cox = 0.0
        total_l_hsic = 0.0
        total_l_nce = 0.0
        total_l_kl = 0.0
        n_batches = 0

        # Beta annealing：前 20% 的 epoch 线性增长
        beta = min(1.0, epoch / (self.args.num_epoch * 0.2 + 1e-9))

        pbar = tqdm(loader, desc=f"Fold {self.fold} Epoch {epoch}")
        self.accum_count = 0
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(pbar):
            path_feat, geno_feat, clin_feat, label, time_val, c = batch

            path_feat = path_feat.cuda() if path_feat.dim() > 1 else None
            geno_feat = geno_feat.cuda()
            clin_feat = clin_feat.cuda()
            time_val = torch.tensor(time_val, dtype=torch.float32).cuda()
            label = label.cuda()
            c = c.cuda()

            # ---- 缺失模态模拟（训练时随机丢弃）----
            mask = {'path': False, 'geno': False, 'clin': False}
            rv = random.random()
            if rv < 0.10:
                mask['path'] = True
            elif rv < 0.20:
                mask['geno'] = True

            # ---- 前向传播 ----
            hazards, S, logits, kl_term, z_dict, attn_weights = model(
                path_feat, geno_feat, clin_feat, mask, beta=beta
            )

            # ---- 损失计算 ----
            l_cox = self.loss_cox_fn(hazards, S, label, c)
            l_kl = kl_term

            # HSIC 解耦损失
            l_hsic = torch.tensor(0.0, device=l_cox.device)
            z_p, z_g, z_c = z_dict['p'], z_dict['g'], z_dict['c']
            if z_p is not None and z_g is not None:
                l_hsic += self.loss_hsic_fn(z_p, z_g)
            if z_g is not None and z_c is not None:
                l_hsic += self.loss_hsic_fn(z_g, z_c)

            # SurvNCE 对比损失
            l_nce = torch.tensor(0.0, device=l_cox.device)
            if mask['path'] and z_g is not None:
                l_nce += self.loss_surv_nce_fn(path_feat.mean(1).cuda(), z_g, time_val, c)
            elif mask['geno'] and z_p is not None:
                l_nce += self.loss_surv_nce_fn(geno_feat.mean(1).cuda() if geno_feat.dim() > 1 else geno_feat, z_p, time_val, c)

            # 总损失
            loss = (l_cox
                    + self.args.loss_kl * l_kl
                    + self.args.loss_hsic * l_hsic
                    + self.args.loss_nce * l_nce)

            # ---- 梯度累积 ----
            loss = loss / self.accumulation_steps
            loss.backward()

            total_loss += loss.item() * self.accumulation_steps
            total_l_cox += l_cox.item()
            total_l_hsic += l_hsic.item()
            total_l_nce += l_nce.item()
            total_l_kl += l_kl.item()
            n_batches += 1

            self.accum_count += 1
            if self.accum_count >= self.accumulation_steps:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=self.args.clip_grad)
                optimizer.step()
                optimizer.zero_grad()
                self.accum_count = 0

                # TensorBoard 记录
                if writer is not None and (global_step % self.args.log_interval == 0):
                    writer.add_scalar(f'fold_{self.fold}/loss_total', total_loss / n_batches, global_step)
                    writer.add_scalar(f'fold_{self.fold}/loss_cox', total_l_cox / n_batches, global_step)
                    writer.add_scalar(f'fold_{self.fold}/loss_hsic', total_l_hsic / n_batches, global_step)
                    writer.add_scalar(f'fold_{self.fold}/loss_nce', total_l_nce / n_batches, global_step)
                    writer.add_scalar(f'fold_{self.fold}/loss_kl', total_l_kl / n_batches, global_step)
                    writer.add_scalar(f'fold_{self.fold}/lr', optimizer.param_groups[0]['lr'], global_step)

                global_step += 1

            pbar.set_postfix({
                'loss': f"{loss.item() * self.accumulation_steps:.4f}",
                'l_cox': f"{l_cox.item():.4f}",
                'beta': f"{beta:.2f}"
            })

        return total_loss / n_batches, global_step

    def validate(self, model, loader, epoch, writer=None, tag='val'):
        """完整模态验证 + 缺失模态鲁棒性测试。"""
        model.eval()
        all_results = {}  # 存储各种模态组合的结果

        # ---- 测试完整模态 ----
        risks_full, times_full, censors_full = self._collect_risks(
            model, loader, mask={'path': False, 'geno': False, 'clin': False}
        )
        c_full = self._compute_cindex(times_full, censors_full, risks_full)
        all_results['full'] = c_full

        # ---- 测试缺失 path ----
        risks_path, times_p, censors_p = self._collect_risks(
            model, loader, mask={'path': True, 'geno': False, 'clin': False}
        )
        c_path = self._compute_cindex(times_p, censors_p, risks_path)
        all_results['missing_path'] = c_path

        # ---- 测试缺失 geno ----
        risks_geno, times_g, censors_g = self._collect_risks(
            model, loader, mask={'path': False, 'geno': True, 'clin': False}
        )
        c_geno = self._compute_cindex(times_g, censors_g, risks_geno)
        all_results['missing_geno'] = c_geno

        # ---- 测试缺失 clin ----
        risks_clin, times_c, censors_c = self._collect_risks(
            model, loader, mask={'path': False, 'geno': False, 'clin': True}
        )
        c_clin = self._compute_cindex(times_c, censors_c, risks_clin)
        all_results['missing_clin'] = c_clin

        if writer is not None:
            for key, val in all_results.items():
                writer.add_scalar(f'fold_{self.fold}/{tag}_{key}', val, epoch)

        return all_results['full'], all_results  # 返回全模态 C-index 和所有结果

    def _collect_risks(self, model, loader, mask):
        risks, times, censors = [], [], []
        with torch.no_grad():
            for batch in loader:
                path_feat, geno_feat, clin_feat, _, time_val, c = batch
                path_feat = path_feat.cuda() if path_feat.dim() > 1 else None
                geno_feat = geno_feat.cuda()
                clin_feat = clin_feat.cuda()

                _, S, _, _, _, _ = model(path_feat, geno_feat, clin_feat, mask=mask)
                risk = -torch.sum(S, dim=1).cpu().numpy()
                risks.extend(risk)
                times.extend(time_val)
                censors.extend(c.cpu().numpy())
        return np.array(risks), np.array(times), np.array(censors)

    def _compute_cindex(self, times, censors, risks):
        import numpy as np
        risks_arr = np.array(risks)
        if np.isnan(risks_arr).any():
            print(f"[Warning] NaNs detected. Replacing with 0.0")
            risks_arr = np.nan_to_num(risks_arr)
        if len(np.unique(risks_arr)) <= 1:
            return 0.5
        return concordance_index_censored(
            (1 - np.array(censors)).astype(bool),
            np.array(times),
            risks_arr
        )[0]


def _build_model(args, train_ds):
    """根据 args 或 ablation 配置构建模型。"""
    use_vib = True
    use_cvae = True
    use_cross_attn = True

    if args.ablation:
        if args.ablation == 'no_vib':
            use_vib = False
        elif args.ablation == 'no_cvae':
            use_cvae = False
        elif args.ablation == 'no_cross_attn':
            use_cross_attn = False
        elif args.ablation == 'path_only':
            return None  # 特殊处理

    clin_input_dim = len(train_ds.metadata)
    omic_sizes = getattr(train_ds, 'omic_sizes', [100, 200, 300])

    model = TriSurv(
        omic_sizes=omic_sizes,
        num_classes=4,
        clin_input_dim=clin_input_dim,
        use_cross_attn=use_cross_attn,
        use_cvae=use_cvae,
        use_vib=use_vib
    ).cuda()
    return model


def main():
    import numpy as np
    args = parse_args()
    set_seed(args.seed)

    # 打印 ablation 信息
    if args.ablation:
        print(f"[INFO] Running ablation: {args.ablation}")

    timestamp = time.strftime("%Y-%m-%d-%H-%M")
    ablation_tag = f"_{args.ablation}" if args.ablation else ""
    results_dir = os.path.join(
        args.save_dir,
        f"{args.dataset}{ablation_tag}_{timestamp}"
    )
    os.makedirs(results_dir, exist_ok=True)

    # TensorBoard writer
    writer = None
    if args.log_data:
        writer = SummaryWriter(log_dir=results_dir)

    fold_scores = []
    all_fold_results = []
    global_step = 0

    for fold in range(5):
        fold_dir = os.path.join(results_dir, f"fold_{fold}")
        os.makedirs(fold_dir, exist_ok=True)
        engine = TriSurvEngine(args, fold_dir, fold)

        # ---- 数据构建 ----
        dataset = Generic_MIL_Survival_Dataset(
            csv_path=f"./csv/{args.dataset}_all_clean/{args.dataset}_all_clean/{args.dataset}_all_clean.csv",
            modal=args.modal,
            OOM=args.OOM,
            apply_sig=True,
            data_dir=args.data_root_dir,
            n_bins=4
        )

        split_dir = os.path.join("./splits", args.which_splits, args.dataset)
        train_ds, val_ds = dataset.return_splits(
            from_id=False,
            csv_path=f"{split_dir}/splits_{fold}.csv"
        )
        train_loader = get_split_loader(
            train_ds, training=True, weighted=args.weighted_sample,
            modal=args.modal, batch_size=args.batch_size
        )
        val_loader = get_split_loader(
            val_ds, training=False, modal=args.modal, batch_size=args.batch_size
        )

        # ---- 模型构建 ----
        model = _build_model(args, train_ds)
        if model is None:
            # path_only 特殊处理：只传 path 特征
            model = TriSurv(
                omic_sizes=[1], num_classes=4,
                clin_input_dim=1,
                use_cross_attn=False, use_cvae=False, use_vib=False
            ).cuda()

        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )

        # 学习率调度
        if args.scheduler == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.num_epoch, eta_min=1e-6
            )
        else:
            scheduler = None

        # ---- Resume 支持 ----
        if args.resume:
            resume_path = os.path.join(fold_dir, 'model_last.pth')
            if os.path.exists(resume_path):
                checkpoint = torch.load(resume_path)
                model.load_state_dict(checkpoint['model'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                if scheduler and 'scheduler' in checkpoint:
                    scheduler.load_state_dict(checkpoint['scheduler'])
                engine.best_c_index = checkpoint.get('best_c_index', 0.0)
                engine.best_epoch = checkpoint.get('best_epoch', 0)
                global_step = checkpoint.get('global_step', 0)
                print(f"[Fold {fold}] Resumed from {resume_path}")

        best_val_c = 0.0

        # ---- 训练循环 ----
        for epoch in range(args.num_epoch):
            # Cosine LR schedule
            if scheduler:
                scheduler.step()

            train_loss, global_step = engine.train_epoch(
                model, train_loader, optimizer, epoch, writer, global_step
            )

            c_val, val_results = engine.validate(
                model, val_loader, epoch, writer, tag='val'
            )

            print(f"Fold {fold} Epoch {epoch} | "
                  f"Val C-index (full): {c_val:.4f} | "
                  f"loss: {train_loss:.4f} | "
                  f"lr: {optimizer.param_groups[0]['lr']:.2e}")

            # 打印缺失模态结果
            for k, v in val_results.items():
                if k != 'full':
                    print(f"  {k}: {v:.4f}")

            # 保存 last checkpoint
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict() if scheduler else None,
                'best_c_index': engine.best_c_index,
                'best_epoch': engine.best_epoch,
                'global_step': global_step,
                'epoch': epoch,
            }, os.path.join(fold_dir, 'model_last.pth'))

            # 保存 best model
            if c_val > best_val_c:
                best_val_c = c_val
                engine.best_c_index = c_val
                engine.best_epoch = epoch
                torch.save(model.state_dict(), os.path.join(fold_dir, 'model_best.pth'))

            # 早停检查
            if engine.early_stopping(c_val, epoch):
                print(f"Fold {fold}: Early stopping triggered at epoch {epoch}. "
                      f"Best was epoch {engine.best_epoch} with C-index {engine.best_c_index:.4f}")
                break

        fold_scores.append(engine.best_c_index)
        all_fold_results.append(val_results)
        print(f"Fold {fold} Finished. Best Score: {engine.best_c_index:.4f} "
              f"at Epoch {engine.best_epoch}")

    # ---- 汇总报告 ----
    print(f"\n{'='*50}")
    print(f"Final CV results (full modality): Mean {np.mean(fold_scores):.4f} "
          f"Std {np.std(fold_scores):.4f}")
    print(f"{'='*50}")

    # 写入报告
    report_path = os.path.join(results_dir, 'final_report.csv')
    with open(report_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["Fold", "Best C-index (full)"] + [k for k in all_fold_results[0].keys()])
        for i, (score, results) in enumerate(zip(fold_scores, all_fold_results)):
            w.writerow([i, score] + [results[k] for k in results.keys()])
        w.writerow(["Mean", np.mean(fold_scores), ""])
        w.writerow(["Std", np.std(fold_scores), ""])

    # 保存完整配置
    config_path = os.path.join(results_dir, 'config.txt')
    with open(config_path, 'w') as f:
        for k, v in vars(args).items():
            f.write(f"{k}: {v}\n")

    if writer:
        writer.close()

    print(f"\nResults saved to: {results_dir}")
    return fold_scores


if __name__ == "__main__":
    main()
