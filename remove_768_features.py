import os
import torch
import glob
import shutil

def check_dimensions(data_dir):
    pt_files = glob.glob(os.path.join(data_dir, "**", "*.pt"), recursive=True)
    
    dim_768_files = []
    dim_1024_files = []
    other_dims_files = []
    
    print(f"Found {len(pt_files)} .pt files. Scanning dimensions...")
    
    for pt_file in pt_files:
        try:
            # We load the tensor on CPU to avoid GPU OOM
            data = torch.load(pt_file, map_location='cpu')
            if isinstance(data, torch.Tensor):
                feature_dim = data.shape[-1]
                if feature_dim == 768:
                    dim_768_files.append(pt_file)
                elif feature_dim == 1024:
                    dim_1024_files.append(pt_file)
                else:
                    other_dims_files.append((pt_file, feature_dim))
        except Exception as e:
            print(f"Failed to load {pt_file}: {e}")

    print("-" * 50)
    print(f"Total files scanned: {len(pt_files)}")
    print(f"1024-dimensional files (Normal): {len(dim_1024_files)}")
    print(f"768-dimensional files (Anomalies): {len(dim_768_files)}")
    if other_dims_files:
        print(f"Other dimensional files: {len(other_dims_files)}")
    print("-" * 50)
    
    if len(dim_768_files) > 0:
        # Instead of completely removing them permanently, we move them to a backup folder 
        # so you have a chance to recover them if needed.
        backup_dir = os.path.join(data_dir, "backup_768_dims")
        os.makedirs(backup_dir, exist_ok=True)
        
        print(f"Moving {len(dim_768_files)} files to {backup_dir} ...")
        for f in dim_768_files:
            filename = os.path.basename(f)
            dest = os.path.join(backup_dir, filename)
            shutil.move(f, dest)
            print(f"Moved: {filename}")
        print("Done handling abnormal features!")
    else:
        print("No 768-dim files found!")

if __name__ == '__main__':
    # Set your data root dir
    dir_path = "D:/TCGA_DATA/TCGA_BLCA"
    if os.path.exists(dir_path):
        check_dimensions(dir_path)
    else:
        print(f"Directory {dir_path} does not exist. Please check your path.")
