import os
import torch
import numpy as np
from sksurv.metrics import concordance_index_censored
from tqdm import tqdm

from datasets.dataset_survival import Generic_MIL_Survival_Dataset
from utils.options import parse_args
from utils.util import get_split_loader
from models.tri_surv import TriSurv

def evaluate_robustness(model, loader, mode='full'):
    model.eval()
    risks, times, censors = [], [], []
    
    # Define masks for different robustness scenarios
    mask = {'path': False, 'geno': False, 'clin': False}
    if mode == 'only_path':
        mask = {'path': False, 'geno': True, 'clin': True}
    elif mode == 'only_geno':
        mask = {'path': True, 'geno': False, 'clin': True}
    elif mode == 'no_path':
        mask = {'path': True, 'geno': False, 'clin': False}
    elif mode == 'no_geno':
        mask = {'path': False, 'geno': True, 'clin': False}
    
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Testing {mode}"):
            path_feat, geno_feat, clin_feat, _, time_val, c = batch
            path_feat = path_feat.cuda() if path_feat.dim() > 1 else None
            geno_feat = geno_feat.cuda(); clin_feat = clin_feat.cuda()
            
            # Forward with specific mask
            _, S, _, _, _, _ = model(path_feat, geno_feat, clin_feat, mask=mask)
            risk = -torch.sum(S, dim=1).cpu().numpy()
            
            risks.extend(risk)
            times.extend(time_val)
            censors.extend(c.cpu().numpy())
            
    c_index = concordance_index_censored((1-np.array(censors)).astype(bool), np.array(times), np.array(risks))[0]
    return c_index

def main():
    args = parse_args()
    # Path to the best model from a specific fold (User needs to provide this)
    # Example: model_path = "./results_trisurv/tcga_blca_.../fold_0/model_best.pth"
    model_path = args.resume if args.resume else None
    
    if not model_path or not os.path.exists(model_path):
        print("Error: Please provide a valid model path using --resume [path]")
        return

    # Build validation dataset (Fold 0 as example)
    dataset = Generic_MIL_Survival_Dataset(
        csv_path=f"./csv/{args.dataset}_all_clean/{args.dataset}_all_clean/{args.dataset}_all_clean.csv",
        modal='trimodal', data_dir=args.data_root_dir, n_bins=4)
    split_dir = os.path.join("./splits", args.which_splits, args.dataset)
    _, val_ds = dataset.return_splits(from_id=False, csv_path=f"{split_dir}/splits_0.csv")
    val_loader = get_split_loader(val_ds, training=False, modal='trimodal', batch_size=args.batch_size)
    
    # Load Model
    model = TriSurv(omic_sizes=val_ds.omic_sizes, num_classes=4, clin_input_dim=len(val_ds.metadata)).cuda()
    model.load_state_dict(torch.load(model_path))
    print(f"Loaded model from {model_path}")
    
    # Run Robustness Tests
    scenarios = ['full', 'only_path', 'only_geno', 'no_path', 'no_geno']
    results = {}
    for s in scenarios:
        score = evaluate_robustness(model, val_loader, mode=s)
        results[s] = score
        print(f"Scenario: {s:10} | C-index: {score:.4f}")
        
    print("\n--- Robustness Summary ---")
    for s, score in results.items():
        drop = results['full'] - score
        print(f"{s:10}: {score:.4f} (Drop: {drop:.4f})")

if __name__ == "__main__":
    main()
