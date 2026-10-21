#!/usr/bin/env python3
"""
VS#-15M High-Performance GPU Trainer (PyTorch / CUDA)
Optimized for Google Colab Tesla T4 GPU.
Trains a 15,000,000 parameter neural network on VerScript and Polyglot Injection Languages.
Exports directly to model_weights.json compatible with VS-Sharp's server.js.
"""

import os
import re
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
        # x: [batch, context_window]
        embeds = self.embedding(x) # [batch, context_window, embed_dim]
        flat = embeds.view(embeds.size(0), -1) # [batch, context_window * embed_dim]
        h1 = self.act1(self.fc1(flat)) # [batch, hidden_size_1]
        h2 = self.act2(self.fc2(h1))   # [batch, hidden_size_2]
        logits = self.fc_out(h2)       # [batch, vocab_size]
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

# --- TEXT GENERATION TESTER ---
def generate_text(model, vocab, vocab_map, prompt, device, max_tokens=50, temperature=0.75):
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
            logits = model(x_in)
            logits = logits[0] / max(temperature, 1e-5)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()

            next_idx = np.random.choice(len(probs), p=probs)
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
        # E: [vocab_size, embed_dim]
        E = model.embedding.weight.cpu().numpy().tolist()

        # W1: In server.js, xFlat is [1, C * E] and W1 is [C * E, H1]
        # In PyTorch fc1.weight is [H1, C * E]. We transpose it (.T)
        W1 = model.fc1.weight.t().cpu().numpy().tolist()
        b1 = model.fc1.bias.cpu().numpy().tolist()

        # W2: [H1, H2] -> PyTorch is [H2, H1], so transpose
        W2 = model.fc2.weight.t().cpu().numpy().tolist()
        b2 = model.fc2.bias.cpu().numpy().tolist()

        # W3: [H2, vocab_size] -> PyTorch is [vocab_size, H2], so transpose
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

    tmp_file = WEIGHTS_FILE + ".tmp"
    with open(tmp_file, "w", encoding="utf-8") as f:
        json.dump(data, f)
    os.replace(tmp_file, WEIGHTS_FILE)
    print(f"[Export] Saved Epoch {epoch} weights ({config['parameterDisplay']}) to model_weights.json")

# --- MAIN TRAINING LOOP ---
def main():
    parser = argparse.ArgumentParser(description="Train VS#-15M on GPU (Google Colab / CUDA)")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs to train")
    parser.add_argument("--batch-size", type=int, default=128, help="Mini-batch size (default 128 for T4 GPU)")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate (Adam)")
    parser.add_argument("--samples-per-epoch", type=int, default=10000, help="Samples trained per epoch")
    parser.add_argument("--reset", action="store_true", help="Reset weights to epoch 0")
    args = parser.parse_args()

    print("=" * 68)
    print("🚀 VS#-15M Deep Neural Language Model Trainer (PyTorch GPU)")
    print("=" * 68)

    # 1. Device Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Using compute device: {device}")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"[GPU] Detected {gpu_name} with {vram:.2f} GB VRAM")

    # 2. Dataset Loading
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"Training dataset not found at {DATA_FILE}. Run 'node build_dataset.js' first.")

    with open(DATA_FILE, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    print(f"[Dataset] Loaded {len(raw_data)} prompt-response pairs from {DATA_FILE}")

    # 3. Vocabulary Building
    special_tokens = ['<pad>', '<start>', '<sep>', '<end>', '<unk>']
    token_counts = {}
    for pair in raw_data:
        p_toks = tokenize(pair.get("prompt", ""))
        r_toks = tokenize(pair.get("response", ""))
        for t in p_toks:
            token_counts[t] = token_counts.get(t, 0) + 1
        for t in r_toks:
            token_counts[t] = token_counts.get(t, 0) + 1

    sorted_tokens = sorted(token_counts.keys(), key=lambda t: token_counts[t], reverse=True)
    vocab = special_tokens + sorted_tokens
    vocab_map = {t: idx for idx, t in enumerate(vocab)}
    vocab_size = len(vocab)
    print(f"[Vocab] Built vocabulary of {vocab_size} unique tokens.")

    # 4. Model Instantiation & Parameter Calculation
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
    print(f"[Model] Hidden Layer 1    : {hidden_size_1} units (tanh)")
    print(f"[Model] Hidden Layer 2    : {HIDDEN_SIZE_2} units (tanh)")

    # 5. Build Training Sliding Windows
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

    # 6. Optimizer & Loss
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    eval_suites = [
        {"category": "Conversation", "prompt": "hello"},
        {"category": "VerScript", "prompt": "write a hello world program in verscript"},
        {"category": "Inject Python", "prompt": "how to run python in verscript"},
        {"category": "Inject JS", "prompt": "write javascript async fetch function"},
        {"category": "Inject C", "prompt": "write c code for binary search"}
    ]

    eval_history = []
    if os.path.exists(EVAL_FILE):
        try:
            with open(EVAL_FILE, "r", encoding="utf-8") as f:
                eval_history = json.load(f)
        except Exception:
            pass

    starting_epoch = 1
    # Check if weights exist and not reset
    if not args.reset and os.path.exists(WEIGHTS_FILE):
        try:
            print("[Setup] Found existing model_weights.json. Loading trained weights...")
            with open(WEIGHTS_FILE, "r", encoding="utf-8") as f:
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
                print(f"[Setup] Successfully loaded weights. Resuming from Epoch {starting_epoch}!")
        except Exception as e:
            print(f"[Setup] Could not load previous weights directly ({e}). Starting fresh.")

    end_epoch = starting_epoch + args.epochs - 1
    print(f"\n[Training] Commencing GPU Training for Epochs {starting_epoch} -> {end_epoch} (Batch Size: {args.batch_size})...")

    # 7. Training Loop
    for epoch in range(starting_epoch, end_epoch + 1):
        if os.path.exists(STOP_FILE):
            print(f"[Stop] Detected {STOP_FILE}. Halting training cleanly.")
            os.remove(STOP_FILE)
            break

        model.train()
        t0 = time.time()

        # Sample sequences for this epoch
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
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_x.size(0)
            preds = torch.argmax(logits, dim=-1)
            total_correct += (preds == batch_y).sum().item()
            total_tokens += batch_x.size(0)

            if (b_idx + 1) % 15 == 0 or (b_idx + 1) == len(loader):
                cur_loss = total_loss / total_tokens
                cur_acc = (total_correct / total_tokens) * 100
                print(f"  [Epoch {epoch}] Batch {b_idx + 1}/{len(loader)} | Loss: {cur_loss:.4f} | Acc: {cur_acc:.1f}%")

        epoch_loss = total_loss / max(total_tokens, 1)
        epoch_acc = (total_correct / max(total_tokens, 1)) * 100
        elapsed = time.time() - t0

        print(f"\n[Epoch {epoch} Finished] Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.1f}% | Duration: {elapsed:.2f}s ({total_tokens/elapsed:.0f} tokens/s)")

        # Save weights checkpoint after EVERY epoch
        export_weights(model, vocab, config, epoch, epoch_loss, epoch_acc / 100.0)

        # Run LLM Evaluation Tests after EVERY epoch
        print(f"\n--- [Epoch {epoch}] LLM Generation Test Review ---")
        epoch_review = {
            "epoch": epoch,
            "loss": epoch_loss,
            "accuracy": epoch_acc / 100.0,
            "timestamp": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            "tests": []
        }

        for test in eval_suites:
            out_text = generate_text(model, vocab, vocab_map, test["prompt"], device, max_tokens=45)
            epoch_review["tests"].append({
                "category": test["category"],
                "prompt": test["prompt"],
                "response": out_text
            })
            print(f"  [{test['category']}] Prompt: \"{test['prompt']}\"")
            print(f"  Generated: {out_text[:140]}{'...' if len(out_text) > 140 else ''}\n")

        eval_history.append(epoch_review)
        with open(EVAL_FILE, "w", encoding="utf-8") as f:
            json.dump(eval_history, f, indent=2)

    print("=" * 68)
    print("✅ GPU Training Complete! Exported latest weights to model_weights.json")
    print("=" * 68)

if __name__ == "__main__":
    main()
