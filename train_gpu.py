#!/usr/bin/env python3
"""
VS#-15M Advanced GPU Trainer (PyTorch / CUDA)
Optimized for Tesla T4 / P100 GPUs (Kaggle & Colab).
Features:
- 15,005,577 Parameter Deep Neural Architecture
- Resumes automatically from Hugging Face or local checkpoint
- Mixed Precision (FP16 AMP) for 2x faster GPU Tensor Core throughput
- AdamW + Cosine Annealing Learning Rate Scheduler
- Gradient Clipping for training stability
- Top-p (Nucleus) + Temperature sampling to eliminate garbled tokens
- Automatic export to model_weights.json and model_weights.json.gz (<45 MB)
- Automated push back to Hugging Face Hub
"""

import os
import re
import sys
import json
import time
import math
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# --- ARCHITECTURE CONSTANTS (TARGET: 15 MILLION PARAMETERS) ---
TARGET_PARAMS = 15000000
CONTEXT_WINDOW = 48
EMBED_DIM = 256
HIDDEN_SIZE_2 = 512

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "knowledge", "training_data.json")
WEIGHTS_FILE = os.path.join(BASE_DIR, "model_weights.json")
EVAL_FILE = os.path.join(BASE_DIR, "epoch_eval_history.json")
STOP_FILE = os.path.join(BASE_DIR, "stop_training.txt")

# Default Hugging Face weights URL to resume from
HF_WEIGHTS_URL = "https://huggingface.co/alexon-v8/VS-Sharp/resolve/main/model_weights.json"

# --- TOKENIZER (IDENTICAL TO SERVER.JS AND BUILD_DATASET.JS) ---
TOKEN_REGEX = re.compile(r'(\r?\n|[a-zA-Z0-9_]+|[^\s\w])')

def tokenize(text):
    tokens = []
    for match in TOKEN_REGEX.finditer(text):
        t = match.group(0)
        if t in ('\r\n', '\n'):
            tokens.append('\n')
        else:
            tokens.append(t)
    return tokens

# --- DYNAMIC PARAMETER TARGETING ---
def calculate_hidden_size(vocab_size):
    numerator = TARGET_PARAMS - vocab_size * (EMBED_DIM + HIDDEN_SIZE_2 + 1) - HIDDEN_SIZE_2
    denominator = CONTEXT_WINDOW * EMBED_DIM + 1 + HIDDEN_SIZE_2
    return max(64, round(numerator / denominator))

# --- PYTORCH NEURAL MODEL ---
class VSSharp15M(nn.Module):
    def __init__(self, vocab_size, hidden_size_1):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_window = CONTEXT_WINDOW
        self.embed_dim = EMBED_DIM
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = HIDDEN_SIZE_2

        # 1. Embedding Layer: [batch, CONTEXT_WINDOW] -> [batch, CONTEXT_WINDOW, EMBED_DIM]
        self.embedding = nn.Embedding(vocab_size, EMBED_DIM)

        # 2. Hidden Layer 1: [batch, CONTEXT_WINDOW * EMBED_DIM] -> [batch, H1]
        self.fc1 = nn.Linear(CONTEXT_WINDOW * EMBED_DIM, hidden_size_1)
        self.act1 = nn.Tanh()

        # 3. Hidden Layer 2: [batch, H1] -> [batch, H2]
        self.fc2 = nn.Linear(hidden_size_1, HIDDEN_SIZE_2)
        self.act2 = nn.Tanh()

        # 4. Output Logits: [batch, H2] -> [batch, vocab_size]
        self.fc_out = nn.Linear(HIDDEN_SIZE_2, vocab_size)

    def forward(self, x):
        embeds = self.embedding(x)
        flat = embeds.view(embeds.size(0), -1)
        h1 = self.act1(self.fc1(flat))
        h2 = self.act2(self.fc2(h1))
        logits = self.fc_out(h2)
        return logits

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())

# --- DATASET LOADER ---
class SequenceDataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = torch.tensor(inputs, dtype=torch.long)
        self.targets = torch.tensor(targets, dtype=torch.long)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

# --- TOP-P NUCLEUS SAMPLING (PREVENTS GARBLED REPETITION) ---
def sample_token(logits, temperature=0.7, top_p=0.9):
    logits = logits / max(temperature, 1e-5)
    probs = torch.softmax(logits, dim=-1)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False

    indices_to_remove = sorted_indices[sorted_indices_to_remove]
    probs[indices_to_remove] = 0.0
    sum_probs = probs.sum()
    if sum_probs > 0:
        probs = probs / sum_probs
        return torch.multinomial(probs, 1).item()
    return torch.argmax(logits).item()

# --- TEXT GENERATION TESTER ---
def generate_text(model, vocab, vocab_map, prompt, device, max_tokens=55, temperature=0.7, top_p=0.9):
    model.eval()
    pad_idx = vocab_map.get('<pad>', 0)
    start_idx = vocab_map.get('<start>', 1)
    sep_idx = vocab_map.get('<sep>', 2)
    end_idx = vocab_map.get('<end>', 3)
    unk_idx = vocab_map.get('<unk>', 4)

    prompt_tokens = tokenize(prompt)
    seq = [start_idx]
    for t in prompt_tokens:
        seq.append(vocab_map.get(t, unk_idx))
    seq.append(sep_idx)

    generated = []

    with torch.no_grad():
        for _ in range(max_tokens):
            ctx = []
            for c in range(CONTEXT_WINDOW, 0, -1):
                idx = len(seq) - c
                ctx.append(pad_idx if idx < 0 else seq[idx])

            x_in = torch.tensor([ctx], dtype=torch.long, device=device)
            logits = model(x_in)[0]

            next_idx = sample_token(logits, temperature=temperature, top_p=top_p)
            if next_idx == end_idx:
                break

            seq.append(next_idx)
            generated.append(vocab[next_idx])

    result = " ".join(generated)
    result = re.sub(r' \n ', '\n', result)
    result = re.sub(r' \n', '\n', result)
    result = re.sub(r'\n ', '\n', result)
    return result

# --- EXPORT WEIGHTS FOR SERVER.JS ---
def export_weights(model, vocab, config, epoch, loss, acc):
    model.eval()
    with torch.no_grad():
        E = model.embedding.weight.cpu().numpy().tolist()
        W1 = model.fc1.weight.t().cpu().numpy().tolist()
        b1 = model.fc1.bias.cpu().numpy().tolist()
        W2 = model.fc2.weight.t().cpu().numpy().tolist()
        b2 = model.fc2.bias.cpu().numpy().tolist()
        W3 = model.fc_out.weight.t().cpu().numpy().tolist()
        b3 = model.fc_out.bias.cpu().numpy().tolist()

    data = {
        "vocab": vocab,
        "config": config,
        "weights": {
            "E": E,
            "W1": W1,
            "b1": b1,
            "W2": W2,
            "b2": b2,
            "W3": W3,
            "b3": b3
        },
        "epoch": epoch,
        "loss": float(loss),
        "accuracy": float(acc),
        "updatedAt": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    }

    # 1. Uncompressed JSON
    tmp_file = WEIGHTS_FILE + ".tmp"
    with open(tmp_file, "w", encoding="utf-8") as f:
        json.dump(data, f)
    os.replace(tmp_file, WEIGHTS_FILE)

    # 2. GZIP Compressed JSON (< 45 MB, ready for git / fast download)
    import gzip
    gz_tmp = WEIGHTS_FILE + ".gz.tmp"
    with gzip.open(gz_tmp, "wt", encoding="utf-8") as f:
        json.dump(data, f)
    os.replace(gz_tmp, WEIGHTS_FILE + ".gz")

    gz_mb = os.path.getsize(WEIGHTS_FILE + ".gz") / (1024 * 1024)
    print(f"[Export] Saved Epoch {epoch} weights to model_weights.json & model_weights.json.gz ({gz_mb:.1f} MB)")

# --- DOWNLOAD FROM HUGGING FACE IF NOT FOUND LOCALLY ---
def ensure_weights_available():
    if os.path.exists(WEIGHTS_FILE) or os.path.exists(WEIGHTS_FILE + ".gz"):
        return True
    try:
        print(f"[*] Downloading previous weights from Hugging Face ({HF_WEIGHTS_URL})...")
        import urllib.request
        urllib.request.urlretrieve(HF_WEIGHTS_URL, WEIGHTS_FILE)
        print(f"[+] Download complete: {WEIGHTS_FILE} ({os.path.getsize(WEIGHTS_FILE)/(1024*1024):.1f} MB)")
        return True
    except Exception as e:
        print(f"[!] Note: Could not auto-download from Hugging Face ({e}).")
        return False

# --- MAIN TRAINING LOOP ---
def main():
    parser = argparse.ArgumentParser(description="Advanced VS#-15M Trainer (PyTorch / CUDA)")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs to train")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size (default 128)")
    parser.add_argument("--lr", type=float, default=0.001, help="Initial learning rate")
    parser.add_argument("--samples-per-epoch", type=int, default=15000, help="Samples per epoch")
    parser.add_argument("--reset", action="store_true", help="Reset weights to 0")
    parser.add_argument("--hf-token", type=str, default=None, help="Hugging Face Write Token for auto-upload")
    parser.add_argument("--hf-repo", type=str, default="alexon-v8/VS-Sharp", help="Hugging Face model repo ID")
    args = parser.parse_args()

    print("=" * 68)
    print("🚀 VS#-15M Deep Neural Language Model Trainer (PyTorch GPU)")
    print("=" * 68)

    # 1. Device Setup & Mixed Precision
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = torch.cuda.is_available()
    print(f"[Device] Compute backend: {device}")
    if use_cuda:
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"[GPU] {gpu_name} ({vram:.2f} GB VRAM) | Mixed Precision FP16 Enabled ⚡")

    # 2. Check / Compile Dataset
    if not os.path.exists(DATA_FILE):
        print("[!] Dataset missing. Running build_dataset.js...")
        os.system("node build_dataset.js")

    with open(DATA_FILE, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    print(f"[Dataset] Loaded {len(raw_data):,} prompt-response pairs from {DATA_FILE}")

    # 3. Vocabulary
    special_tokens = ['<pad>', '<start>', '<sep>', '<end>', '<unk>']
    token_counts = {}
    for pair in raw_data:
        p_toks = tokenize(pair.get("prompt", ""))
        r_toks = tokenize(pair.get("response", ""))
        for t in p_toks: token_counts[t] = token_counts.get(t, 0) + 1
        for t in r_toks: token_counts[t] = token_counts.get(t, 0) + 1

    sorted_tokens = sorted(token_counts.keys(), key=lambda t: token_counts[t], reverse=True)
    vocab = special_tokens + sorted_tokens
    vocab_map = {t: idx for idx, t in enumerate(vocab)}
    vocab_size = len(vocab)
    print(f"[Vocab] Built vocabulary of {vocab_size} unique tokens.")

    # 4. Model Construction
    hidden_size_1 = calculate_hidden_size(vocab_size)
    model = VSSharp15M(vocab_size, hidden_size_1).to(device)
    actual_params = model.count_parameters()

    config = {
        "modelName": "VS#-15M",
        "parameterCount": actual_params,
        "parameterDisplay": f"{actual_params / 1e6:.2f} Million ({round(actual_params / 1e6)}M)",
        "architecture": "VS#-15M Deep Neural Language Model & Polyglot Engine",
        "vocabSize": vocab_size,
        "contextWindow": CONTEXT_WINDOW,
        "embedDim": EMBED_DIM,
        "hiddenSize": hidden_size_1,
        "hiddenSize2": HIDDEN_SIZE_2
    }

    print(f"[Model] Target Parameters : {TARGET_PARAMS:,}")
    print(f"[Model] Actual Parameters : {actual_params:,} ({config['parameterDisplay']})")
    print(f"[Model] Context Window    : {CONTEXT_WINDOW} tokens")
    print(f"[Model] Hidden Layer 1    : {hidden_size_1} units")
    print(f"[Model] Hidden Layer 2    : {HIDDEN_SIZE_2} units")

    # 5. Sliding Windows
    pad_idx = vocab_map['<pad>']
    start_idx = vocab_map['<start>']
    sep_idx = vocab_map['<sep>']
    end_idx = vocab_map['<end>']
    unk_idx = vocab_map['<unk>']

    all_inputs = []
    all_targets = []

    for pair in raw_data:
        p_toks = [vocab_map.get(t, unk_idx) for t in tokenize(pair.get("prompt", ""))]
        r_toks = [vocab_map.get(t, unk_idx) for t in tokenize(pair.get("response", ""))]
        seq = [start_idx] + p_toks + [sep_idx] + r_toks + [end_idx]
        prompt_len = len(p_toks) + 2

        for i in range(prompt_len, len(seq)):
            ctx = []
            for c in range(CONTEXT_WINDOW, 0, -1):
                pos = i - c
                ctx.append(pad_idx if pos < 0 else seq[pos])
            all_inputs.append(ctx)
            all_targets.append(seq[i])

    total_samples = len(all_inputs)
    print(f"[Dataset] Generated {total_samples:,} sliding context sequences.")

    # 6. Check / Resume Weights
    starting_epoch = 1
    if not args.reset:
        ensure_weights_available()
        target_load = WEIGHTS_FILE if os.path.exists(WEIGHTS_FILE) else (WEIGHTS_FILE + ".gz")
        if os.path.exists(target_load):
            try:
                print(f"[Setup] Loading checkpoint from {target_load}...")
                if target_load.endswith('.gz'):
                    import gzip
                    with gzip.open(target_load, 'rt', encoding='utf-8') as f:
                        prev_data = json.load(f)
                else:
                    with open(target_load, 'r', encoding='utf-8') as f:
                        prev_data = json.load(f)

                w = prev_data.get("weights", {})
                if "E" in w and "W1" in w:
                    model.embedding.weight.data.copy_(torch.tensor(w["E"], dtype=torch.float32))
                    model.fc1.weight.data.copy_(torch.tensor(w["W1"], dtype=torch.float32).t())
                    model.fc1.bias.data.copy_(torch.tensor(w["b1"], dtype=torch.float32))
                    model.fc2.weight.data.copy_(torch.tensor(w["W2"], dtype=torch.float32).t())
                    model.fc2.bias.data.copy_(torch.tensor(w["b2"], dtype=torch.float32))
                    model.fc_out.weight.data.copy_(torch.tensor(w["W3"], dtype=torch.float32).t())
                    model.fc_out.bias.data.copy_(torch.tensor(w["b3"], dtype=torch.float32))
                    starting_epoch = prev_data.get("epoch", 0) + 1
                    print(f"🌟 [Setup] Successfully loaded weights! Resuming from Epoch {starting_epoch}!")
            except Exception as e:
                print(f"[!] Warning loading weights ({e}). Starting fresh.")

    # 7. Optimizer, Scheduler, Scaler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    scaler = torch.cuda.amp.GradScaler(enabled=use_cuda)
    criterion = nn.CrossEntropyLoss()

    eval_suites = [
        {"category": "Conversation", "prompt": "hello"},
        {"category": "VerScript", "prompt": "write a hello world program in verscript"},
        {"category": "Inject Python", "prompt": "how to run python in verscript"},
        {"category": "Inject JS", "prompt": "write javascript async fetch function"},
        {"category": "Inject C", "prompt": "write c code for binary search"},
        {"category": "VerScript Math", "prompt": "how to use Math.abs in verscript"}
    ]

    eval_history = []
    if os.path.exists(EVAL_FILE):
        try:
            with open(EVAL_FILE, "r", encoding="utf-8") as f:
                eval_history = json.load(f)
        except Exception:
            pass

    end_epoch = starting_epoch + args.epochs - 1
    print(f"\n[Training] Commencing GPU Training for Epochs {starting_epoch} -> {end_epoch}...")

    # 8. Main Loop
    for epoch in range(starting_epoch, end_epoch + 1):
        if os.path.exists(STOP_FILE):
            print(f"[Stop] Detected {STOP_FILE}. Halting cleanly.")
            os.remove(STOP_FILE)
            break

        model.train()
        t0 = time.time()

        sample_count = min(total_samples, args.samples_per_epoch)
        indices = np.random.choice(total_samples, size=sample_count, replace=False)
        epoch_inputs = [all_inputs[i] for i in indices]
        epoch_targets = [all_targets[i] for i in indices]

        dataset = SequenceDataset(epoch_inputs, epoch_targets)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

        total_loss = 0.0
        total_correct = 0
        total_tokens = 0

        for b_idx, (batch_x, batch_y) in enumerate(loader):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=use_cuda):
                logits = model(batch_x)
                loss = criterion(logits, batch_y)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item() * batch_x.size(0)
            preds = torch.argmax(logits, dim=-1)
            total_correct += (preds == batch_y).sum().item()
            total_tokens += batch_x.size(0)

            if (b_idx + 1) % 20 == 0 or (b_idx + 1) == len(loader):
                cur_loss = total_loss / total_tokens
                cur_acc = (total_correct / total_tokens) * 100
                print(f"  [Epoch {epoch}] Batch {b_idx + 1}/{len(loader)} | Loss: {cur_loss:.4f} | Acc: {cur_acc:.1f}%")

        scheduler.step()
        epoch_loss = total_loss / max(total_tokens, 1)
        epoch_acc = (total_correct / max(total_tokens, 1)) * 100
        elapsed = time.time() - t0

        print(f"\n[Epoch {epoch} Finished] Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.1f}% | Time: {elapsed:.2f}s ({total_tokens/elapsed:.0f} tok/s) | LR: {scheduler.get_last_lr()[0]:.6f}")

        # Save weights checkpoint after EVERY epoch
        export_weights(model, vocab, config, epoch, epoch_loss, epoch_acc / 100.0)

        # Run LLM Generation Evaluation Tests
        print(f"\n--- [Epoch {epoch}] Live Neural Generation Test Review ---")
        epoch_review = {
            "epoch": epoch,
            "loss": epoch_loss,
            "accuracy": epoch_acc / 100.0,
            "timestamp": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            "tests": []
        }

        for test in eval_suites:
            out_text = generate_text(model, vocab, vocab_map, test["prompt"], device, max_tokens=55, temperature=0.7, top_p=0.9)
            epoch_review["tests"].append({
                "category": test["category"],
                "prompt": test["prompt"],
                "response": out_text
            })
            print(f"  [{test['category']}] Prompt: \"{test['prompt']}\"")
            print(f"  Generated:\n{out_text[:180]}{'...' if len(out_text) > 180 else ''}\n")

        eval_history.append(epoch_review)
        with open(EVAL_FILE, "w", encoding="utf-8") as f:
            json.dump(eval_history, f, indent=2)

    # 9. Auto-upload to Hugging Face if token provided
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    if hf_token and args.hf_repo:
        print("\n[*] Automatically pushing latest trained weights to Hugging Face Hub...")
        try:
            from huggingface_hub import HfApi, create_repo
            create_repo(args.hf_repo, repo_type="model", exist_ok=True, token=hf_token)
            api = HfApi()
            api.upload_file(
                path_or_fileobj=WEIGHTS_FILE,
                path_in_repo="model_weights.json",
                repo_id=args.hf_repo,
                repo_type="model",
                token=hf_token
            )
            api.upload_file(
                path_or_fileobj=WEIGHTS_FILE + ".gz",
                path_in_repo="model_weights.json.gz",
                repo_id=args.hf_repo,
                repo_type="model",
                token=hf_token
            )
            print(f"✅ Successfully updated model on Hugging Face: https://huggingface.co/{args.hf_repo}")
        except Exception as e:
            print(f"[!] Warning auto-uploading to Hugging Face ({e})")

    print("=" * 68)
    print("✅ Training Loop Finished! Latest weights in model_weights.json")
    print("=" * 68)

if __name__ == "__main__":
    main()
