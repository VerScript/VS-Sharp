#!/usr/bin/env python3
"""
VS#-15M Advanced GPU Trainer (PyTorch / CUDA)
Optimized for Tesla T4 / P100 GPUs (Kaggle & Colab).
Features:
- 15,005,577 Parameter Deep Neural Architecture
- Resumes automatically from Hugging Face or local checkpoint (Epoch 150 -> 151+)
- Warm-Start Vocabulary Alignment (Strictly preserves 100% of pre-trained weights)
- Anti-Memorization Regularization: Dropout (p=0.15) + Label Smoothing (0.1) + AdamW Weight Decay (1e-2)
- Mixed Precision (FP16 AMP) for 2x faster GPU Tensor Core throughput
- Cosine Annealing Learning Rate Scheduler for fine-tuning without catastrophic forgetting
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

# --- PYTORCH NEURAL MODEL WITH DROPOUT (ANTI-OVERFITTING) ---
class VSSharp15M(nn.Module):
    def __init__(self, vocab_size, hidden_size_1, dropout_p=0.15):
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
        self.drop1 = nn.Dropout(p=dropout_p)

        # 3. Hidden Layer 2: [batch, H1] -> [batch, H2]
        self.fc2 = nn.Linear(hidden_size_1, HIDDEN_SIZE_2)
        self.act2 = nn.Tanh()
        self.drop2 = nn.Dropout(p=dropout_p)

        # 4. Output Logits: [batch, H2] -> [batch, vocab_size]
        self.fc_out = nn.Linear(HIDDEN_SIZE_2, vocab_size)

    def forward(self, x):
        embeds = self.embedding(x)
        flat = embeds.view(embeds.size(0), -1)
        h1 = self.drop1(self.act1(self.fc1(flat)))
        h2 = self.drop2(self.act2(self.fc2(h1)))
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
def sample_token(logits, temperature=0.75, top_p=0.9):
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
def generate_text(model, vocab, vocab_map, prompt, device, max_tokens=60, temperature=0.75, top_p=0.9):
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

    # Reconstruct readable string
    out = ""
    for token in generated:
        if token == '\n':
            out += '\n'
        elif len(out) == 0 or out[-1] in ('\n', ' ', '(', '[', '{', '.', '`', ':', '═'):
            out += token
        elif token in ('.', ',', ':', ';', ')', ']', '}', '?', '!'):
            out += token
        else:
            out += ' ' + token
    return out.strip()

# --- EXPORT WEIGHTS FUNCTION ---
def export_weights(model, vocab, config, epoch, loss, accuracy):
    data = {
        "vocab": vocab,
        "config": config,
        "weights": {
            "E": model.embedding.weight.data.cpu().numpy().tolist(),
            "W1": model.fc1.weight.data.cpu().numpy().T.tolist(),
            "b1": model.fc1.bias.data.cpu().numpy().tolist(),
            "W2": model.fc2.weight.data.cpu().numpy().T.tolist(),
            "b2": model.fc2.bias.data.cpu().numpy().tolist(),
            "W3": model.fc_out.weight.data.cpu().numpy().T.tolist(),
            "b3": model.fc_out.bias.data.cpu().numpy().tolist()
        },
        "epoch": epoch,
        "loss": float(loss),
        "accuracy": float(accuracy),
        "updatedAt": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    }

    # 1. Standard JSON file
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
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs to train (default 50)")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size (default 128)")
    parser.add_argument("--lr", type=float, default=0.00015, help="Fine-tuning learning rate (default 0.00015)")
    parser.add_argument("--samples-per-epoch", type=int, default=20000, help="Samples per epoch (default 20000)")
    parser.add_argument("--dropout", type=float, default=0.15, help="Dropout rate to break memorization (default 0.15)")
    parser.add_argument("--label-smoothing", type=float, default=0.1, help="Label smoothing to avoid 99.8% overconfidence (default 0.1)")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="AdamW weight decay regularization (default 0.01)")
    parser.add_argument("--reset", action="store_true", help="Reset weights to 0 (default: False)")
    parser.add_argument("--hf-token", type=str, default=None, help="Hugging Face Write Token for auto-upload")
    parser.add_argument("--hf-repo", type=str, default="alexon-v8/VS-Sharp", help="Hugging Face model repo ID")
    args = parser.parse_args()

    print("=" * 68)
    print("🚀 VS#-15M Deep Neural Language Model Trainer (PyTorch GPU)")
    print("   Regularized Continuous Training & Warm-Start Architecture")
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

    # 3. Ingest Checkpoint First (Guarantees Vocab Alignment & Preserves Weights)
    prev_data = None
    target_load = WEIGHTS_FILE if os.path.exists(WEIGHTS_FILE) else (WEIGHTS_FILE + ".gz")
    if not os.path.exists(target_load) and not args.reset:
        ensure_weights_available()
        target_load = WEIGHTS_FILE if os.path.exists(WEIGHTS_FILE) else (WEIGHTS_FILE + ".gz")

    if os.path.exists(target_load) and not args.reset:
        try:
            print(f"[Setup] Reading checkpoint metadata from {target_load}...")
            if target_load.endswith('.gz'):
                import gzip
                with gzip.open(target_load, 'rt', encoding='utf-8') as f:
                    prev_data = json.load(f)
            else:
                with open(target_load, 'r', encoding='utf-8') as f:
                    prev_data = json.load(f)
        except Exception as e:
            print(f"[!] Warning reading checkpoint: {e}")

    # 4. Checkpoint-Aware Vocabulary Alignment
    special_tokens = ['<pad>', '<start>', '<sep>', '<end>', '<unk>']
    if prev_data and "vocab" in prev_data and len(prev_data["vocab"]) > 0:
        base_vocab = list(prev_data["vocab"])
        print(f"🌟 [Vocab Alignment] Preserving all {len(base_vocab):,} base tokens from checkpoint.")
    else:
        base_vocab = list(special_tokens)

    existing_set = set(base_vocab)
    new_tokens = []
    token_counts = {}
    for pair in raw_data:
        p_toks = tokenize(pair.get("prompt", ""))
        r_toks = tokenize(pair.get("response", ""))
        for t in p_toks + r_toks:
            if t not in existing_set:
                token_counts[t] = token_counts.get(t, 0) + 1

    for t, _ in sorted(token_counts.items(), key=lambda x: x[1], reverse=True):
        new_tokens.append(t)
        existing_set.add(t)

    vocab = base_vocab + new_tokens
    vocab_map = {t: idx for idx, t in enumerate(vocab)}
    vocab_size = len(vocab)
    old_vocab_size = len(base_vocab) if prev_data else 0
    print(f"[Vocab] Final vocabulary: {vocab_size} tokens ({old_vocab_size} preserved from checkpoint + {len(new_tokens)} newly mapped).")

    # 5. Architecture Construction
    if prev_data and "config" in prev_data and "hiddenSize" in prev_data["config"]:
        hidden_size_1 = prev_data["config"]["hiddenSize"]
    else:
        hidden_size_1 = calculate_hidden_size(vocab_size)

    model = VSSharp15M(vocab_size, hidden_size_1, dropout_p=args.dropout).to(device)
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
    print(f"[Model] Regularization    : Dropout p={args.dropout} | Label Smoothing={args.label_smoothing} | Weight Decay={args.weight_decay}")

    # 6. Weight Transfer (Warm-Start without Losing Epoch 150 Progress)
    starting_epoch = 1
    if prev_data and "weights" in prev_data:
        w = prev_data["weights"]
        try:
            # Transfer Hidden Layers 1 & 2 (Identical shapes)
            model.fc1.weight.data.copy_(torch.tensor(w["W1"], dtype=torch.float32).t())
            model.fc1.bias.data.copy_(torch.tensor(w["b1"], dtype=torch.float32))
            model.fc2.weight.data.copy_(torch.tensor(w["W2"], dtype=torch.float32).t())
            model.fc2.bias.data.copy_(torch.tensor(w["b2"], dtype=torch.float32))

            # Transfer Embedding
            old_e = torch.tensor(w["E"], dtype=torch.float32)
            model.embedding.weight.data[:old_e.size(0), :].copy_(old_e)
            if vocab_size > old_e.size(0):
                e_mean = old_e.mean()
                e_std = old_e.std()
                model.embedding.weight.data[old_e.size(0):, :].normal_(e_mean.item(), e_std.item())

            # Transfer Output Logits
            w3_t = torch.tensor(w["W3"], dtype=torch.float32).t()
            b3 = torch.tensor(w["b3"], dtype=torch.float32)
            model.fc_out.weight.data[:w3_t.size(0), :].copy_(w3_t)
            model.fc_out.bias.data[:b3.size(0)].copy_(b3)

            starting_epoch = prev_data.get("epoch", 150) + 1
            print(f"🌟 [Warm-Start] SUCCESS: 100% of Epoch {prev_data.get('epoch', 150)} weights preserved!")
            print(f"🚀 Resuming continuous regularized training from Epoch {starting_epoch}!")
        except Exception as e:
            print(f"[!] Warning transferring weights ({e}). Starting fresh.")

    # 7. Sliding Windows Generation
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
    print(f"[Dataset] Generated {total_samples:,} sliding context sequences across {len(raw_data):,} pairs.")

    # 8. Regularized Optimizer, Cosine Scheduler, Scaler, and Smoothed Loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler(enabled=use_cuda)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    eval_suites = [
        {"category": "Conversation", "prompt": "hello"},
        {"category": "VerScript Syntax", "prompt": "write a hello world program in verscript"},
        {"category": "Algorithm", "prompt": "write a function to check if a number is prime in verscript"},
        {"category": "Syntax Repair", "prompt": "how to fix count = 10 in verscript"},
        {"category": "Inject Python", "prompt": "how to run python in verscript"},
        {"category": "Inject JS", "prompt": "write javascript async fetch function"},
        {"category": "Inject C", "prompt": "write c code for binary search"},
        {"category": "VerScript Stdlib", "prompt": "how to use Math.sqrt in verscript"}
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
    print(f"[*] Training Strategy: Anti-Overfitting Active (Label Smoothing {args.label_smoothing}, Dropout {args.dropout})")

    # 9. Main Loop
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
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

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

            if (b_idx + 1) % 25 == 0 or (b_idx + 1) == len(loader):
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

        # Run Live LLM Generation Evaluation Tests
        print(f"\n--- [Epoch {epoch}] Live Neural Generation Test Review ---")
        epoch_review = {
            "epoch": epoch,
            "loss": epoch_loss,
            "accuracy": epoch_acc / 100.0,
            "timestamp": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            "tests": []
        }

        for test in eval_suites:
            out_text = generate_text(model, vocab, vocab_map, test["prompt"], device, max_tokens=60, temperature=0.75, top_p=0.9)
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

    # 10. Auto-upload to Hugging Face Hub if token provided
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    if hf_token and args.hf_repo:
        print("\n[*] Automatically pushing updated trained weights to Hugging Face Hub...")
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
