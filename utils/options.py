import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Configurations for Tri-Surv Survival Analysis on TCGA Data.")

    # Path & Data
    parser.add_argument(
        "--data_root_dir", type=str, default="/data3/share/TCGA/BLCA_feature",
        help="Data directory to WSI features (extracted via CLAM)"
    )
    parser.add_argument("--seed", type=int, default=1, help="Random seed (default: 1)")
    parser.add_argument(
        "--which_splits", type=str, default="5foldcv",
        help="Which splits folder to use in ./splits/ (Default: ./splits/5foldcv)"
    )
    parser.add_argument(
        "--dataset", type=str, default="tcga_blca",
        help='Which cancer type within ./splits/<which_dataset> to use. (Default: tcga_blca)'
    )
    parser.add_argument("--OOM", type=int, default=0,
        help="Randomly sample patches to avoid OOM error")

    # Model
    parser.add_argument(
        "--model", type=str, default="TriSurv",
        help="Type of model (Default: TriSurv)"
    )
    parser.add_argument(
        "--modal", type=str,
        choices=["omic", "path", "pathomic", "cluster", "coattn", "trimodal"],
        default="trimodal",
        help="Specifies which modalities to use."
    )
    parser.add_argument(
        "--ablation", type=str, default=None,
        choices=["no_clinical", "no_vib", "no_cvae", "no_cross_attn",
                 "no_hsic", "no_nce", "path_only", "omic_only", "full"],
        help="Ablation mode. If set, overrides model components."
    )
    parser.add_argument("--alpha", type=float, default=1,
        help="hyper-parameter of loss function")

    # Training
    parser.add_argument("--num_epoch", type=int, default=30,
        help="Maximum number of epochs (default: 30)")
    parser.add_argument("--batch_size", type=int, default=1,
        help="Batch size (default: 1, due to varying bag sizes)")
    parser.add_argument("--accumulation_steps", type=int, default=8,
        help="Gradient accumulation steps. Effective batch_size = batch_size * accumulation_steps (default: 8)")
    parser.add_argument("--patience", type=int, default=10,
        help="Early stopping patience. Stop if no improvement for N epochs (default: 10)")
    parser.add_argument("--lr", type=float, default=2e-4,
        help="Learning rate (default: 0.0002)")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
        help="Weight decay (default: 0.0001)")
    parser.add_argument("--optimizer", type=str,
        choices=["SGD", "Adam", "AdamW", "RAdam"],
        default="Adam")
    parser.add_argument("--scheduler", type=str,
        choices=["None", "exp", "step", "plateau", "cosine"],
        default="cosine")
    parser.add_argument("--loss", type=str, default="nll_surv",
        help="Survival loss function (default: nll_surv)")
    parser.add_argument("--weighted_sample", action="store_true", default=True,
        help="Enable weighted sampling for class imbalance")
    parser.add_argument("--clip_grad", type=float, default=1.0,
        help="Gradient clipping norm (default: 1.0)")

    # Loss weights
    parser.add_argument("--loss_kl", type=float, default=0.01,
        help="Weight for KL divergence loss (VIB) (default: 0.01)")
    parser.add_argument("--loss_hsic", type=float, default=0.05,
        help="Weight for HSIC decoupling loss (default: 0.05)")
    parser.add_argument("--loss_nce", type=float, default=0.1,
        help="Weight for SurvNCE loss (default: 0.1)")

    # Checkpoint & Logging
    parser.add_argument("--log_data", action="store_true", default=True,
        help="Enable TensorBoard logging")
    parser.add_argument("--log_interval", type=int, default=10,
        help="Log every N batches (default: 10)")
    parser.add_argument("--evaluate", action="store_true",
        help="Evaluate model on test set")
    parser.add_argument("--resume", type=str, default="",
        help="Path to checkpoint to resume from (default: none)")
    parser.add_argument("--save_dir", type=str, default="results_trisurv",
        help="Directory to save results (default: results_trisurv)")

    # GPU
    parser.add_argument("--gpu", type=int, default=0,
        help="GPU device ID (default: 0)")

    args = parser.parse_args()
    return args
