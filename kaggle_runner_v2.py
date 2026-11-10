#!/usr/bin/env python3
"""
kaggle_runner_v2.py — Automated Kaggle Training Runner for VS#-15M
Optimized for Tesla T4 / P100 GPUs on Kaggle Notebooks.
- Automatic dataset compilation using Node.js build_dataset.js
- Resumes training from latest Hugging Face checkpoint (Epoch 150+ -> 200+)
- Regularized AdamW + Cosine Annealing + Mixed Precision FP16
- Compresses output to model_weights.json.gz and synchronizes back to Hugging Face
"""

import os
import sys
import subprocess
import urllib.request

def run_cmd(cmd, cwd=None):
    print(f"[*] Executing: {cmd}")
    res = subprocess.run(cmd, shell=True, cwd=cwd)
    if res.returncode != 0:
        print(f"[!] Warning/Error executing: {cmd} (exit code: {res.returncode})")
    return res.returncode

def main():
    print("===========================================================")
    print("🚀 Starting VS#-15M Kaggle Training Pipeline v2")
    print("===========================================================")
    
    # 1. Verify CUDA
    try:
        import torch
        print(f"[*] PyTorch version: {torch.__version__}")
        print(f"[*] CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"[*] Device Name: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("[!] PyTorch not found. Installing...")
        run_cmd("pip install -q torch torchvision torchaudio")

    # 2. Recompile master dataset
    print("[*] Recompiling master training dataset...")
    run_cmd("node build_dataset.js")

    # 3. Download current baseline checkpoint if not present
    if not os.path.exists("model_weights.json"):
        hf_url = "https://huggingface.co/alexon-v8/VS-Sharp/resolve/main/model_weights.json"
        print(f"[*] Downloading latest checkpoint from: {hf_url}")
        try:
            urllib.request.urlretrieve(hf_url, "model_weights.json")
            print("[*] Checkpoint downloaded successfully.")
        except Exception as e:
            print(f"[!] Failed downloading checkpoint ({e}). Starting fresh.")

    # 4. Token configuration
    hf_token = os.environ.get("HF_TOKEN", "")
    if not hf_token:
        try:
            from kaggle_secrets import UserSecretsClient
            user_secrets = UserSecretsClient()
            hf_token = user_secrets.get_secret("HF_TOKEN")
        except Exception:
            pass

    # 5. Execute GPU training
    print("[*] Launching train_gpu.py with Mixed Precision & Anti-Memorization regularizers...")
    epochs = os.environ.get("VS_EPOCHS", "100")
    batch_size = os.environ.get("VS_BATCH", "128")
    lr = os.environ.get("VS_LR", "0.0003")
    
    train_args = [
        "python train_gpu.py",
        f"--epochs {epochs}",
        f"--batch-size {batch_size}",
        f"--lr {lr}",
        "--dropout 0.15",
        "--label-smoothing 0.1",
        "--save-every 10",
        "--resume"
    ]
    if hf_token:
        train_args.append(f'--hf-token "{hf_token}"')
        train_args.append('--hf-repo "alexon-v8/VS-Sharp"')

    cmd = " ".join(train_args)
    run_cmd(cmd)

    # 6. Copy artifacts to Kaggle Output
    kaggle_out = "/kaggle/working"
    if os.path.exists(kaggle_out):
        print("[*] Copying trained weights to Kaggle Output directory...")
        run_cmd(f"cp model_weights.json.gz {kaggle_out}/ 2>/dev/null || true")
        run_cmd(f"cp epoch_eval_history.json {kaggle_out}/ 2>/dev/null || true")
        print("✅ Artifacts ready for download in Kaggle Output panel.")

if __name__ == "__main__":
    main()
