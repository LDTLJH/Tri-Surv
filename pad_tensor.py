import torch, os
fp = "D:/TCGA_DATA/TCGA_BLCA/backup_768_dims/TCGA-2F-A9KO-01Z-00-DX1.195576CF-B739-4BD9-B15B-4A70AE287D3E.pt"
dest = "D:/TCGA_DATA/TCGA_BLCA/resnet50/TCGA-2F-A9KO-01Z-00-DX1.195576CF-B739-4BD9-B15B-4A70AE287D3E.pt"
print(f"Loading {fp}...")
t = torch.load(fp, map_location='cpu')
print(f"Original shape: {t.shape}")
t_padded = torch.nn.functional.pad(t, (0, 1024 - 768))
print(f"Padded shape: {t_padded.shape}")
os.makedirs(os.path.dirname(dest), exist_ok=True)
torch.save(t_padded, dest)
print(f"Successfully saved to {dest}!")
