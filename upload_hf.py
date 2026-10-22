#!/usr/bin/env python3
"""
Upload trained VS# model weights to Hugging Face Hub for free, unlimited cloud storage and direct online URL access.
Works in Kaggle, Google Colab, or local environments.
"""

import os
import sys
import argparse

def upload():
    parser = argparse.ArgumentParser(description="Upload VS# model weights to Hugging Face Hub")
    parser.add_argument("--repo-id", type=str, required=True, help="Hugging Face repo ID, e.g., 'your-username/vs-sharp-15m'")
    parser.add_argument("--token", type=str, required=True, help="Hugging Face Write Token (from huggingface.co/settings/tokens)")
    parser.add_argument("--file", type=str, default=None, help="Path to weights file (defaults to model_weights.json.gz or model_weights.json)")
    args = parser.parse_args()

    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        print("[!] huggingface_hub is not installed. Installing now...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"])
        from huggingface_hub import HfApi, create_repo

    target_file = args.file
    if not target_file:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        gz_path = os.path.join(base_dir, "model_weights.json.gz")
        json_path = os.path.join(base_dir, "model_weights.json")

        # Also check /kaggle/working/ if running in Kaggle
        kaggle_gz = "/kaggle/working/VS-Sharp/model_weights.json.gz"
        kaggle_json = "/kaggle/working/VS-Sharp/model_weights.json"

        if os.path.exists(gz_path):
            target_file = gz_path
        elif os.path.exists(json_path):
            target_file = json_path
        elif os.path.exists(kaggle_gz):
            target_file = kaggle_gz
        elif os.path.exists(kaggle_json):
            target_file = kaggle_json
        else:
            print("[X] Error: Could not find model_weights.json.gz or model_weights.json!")
            sys.exit(1)

    file_size_mb = os.path.getsize(target_file) / (1024 * 1024)
    file_name = os.path.basename(target_file)
    print(f"[*] Target file : {target_file} ({file_size_mb:.2f} MB)")
    print(f"[*] Destination : https://huggingface.co/{args.repo_id}")

    # 1. Create model repository if it doesn't already exist
    print("[*] Ensuring repository exists on Hugging Face...")
    create_repo(args.repo_id, repo_type="model", exist_ok=True, token=args.token)

    # 2. Upload file
    print(f"[*] Uploading {file_name} to Hugging Face Hub...")
    api = HfApi()
    api.upload_file(
        path_or_fileobj=target_file,
        path_in_repo=file_name,
        repo_id=args.repo_id,
        repo_type="model",
        token=args.token
    )

    # Also upload eval history if present
    eval_file = os.path.join(os.path.dirname(target_file), "epoch_eval_history.json")
    if os.path.exists(eval_file):
        try:
            api.upload_file(
                path_or_fileobj=eval_file,
                path_in_repo="epoch_eval_history.json",
                repo_id=args.repo_id,
                repo_type="model",
                token=args.token
            )
            print("[*] Uploaded epoch_eval_history.json as well!")
        except Exception:
            pass

    direct_url = f"https://huggingface.co/{args.repo_id}/resolve/main/{file_name}"
    print("=" * 68)
    print("✅ Successfully uploaded model weights to Hugging Face!")
    print(f"🔗 Repository URL : https://huggingface.co/{args.repo_id}")
    print(f"🌐 Direct Raw URL : {direct_url}")
    print("=" * 68)
    print(f"\nYou can set MODEL_WEIGHTS_URL={direct_url} on any server to access weights instantly!")

if __name__ == "__main__":
    upload()
