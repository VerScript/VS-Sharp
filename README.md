# VS-Sharp (VS#) 🧠⚡

Official Neural Language Model & Intelligent Runtime Engine for **VerScript** and Polyglot Development.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/VerScript/VS-Sharp/blob/main/colab_train.ipynb)

---

## 🌟 Overview
**VS#** is an autonomous AI coding assistant and neural language engine purpose-built for the **VerScript** programming ecosystem. It provides:
- **15 Million Parameters (VS#-15M)**: Deep neural architecture dynamically configured to 15,000,000 weights.
- **Polyglot Code Generation**: Writes pure VerScript as well as foreign code accessible via the `inject` directive (`inject python`, `inject javascript`, `inject c`, `inject html`).
- **Conversational Reasoning**: Understands natural language programming queries, walkthroughs, explanations, and debugging.
- **Zero Canned Answers**: Pure neural autoregressive generation — never relies on hardcoded or pre-written responses.
- **Direct IDE Buffer Injection**: Emits structured JSON action payloads (`{ action: { type: "edit", code: "..." } }`) for connected IDEs and playgrounds.

---

## ⚡ Fast GPU Training with Google Colab (Tesla T4)

Training 15 million parameters on CPU is computationally intensive. We provide a fully automated, one-click GPU training pipeline optimized for **Google Colab Tesla T4 GPUs**.

### Quickstart on Colab:
1. Click the **[Open In Colab](https://colab.research.google.com/github/VerScript/VS-Sharp/blob/main/colab_train.ipynb)** badge.
2. Select **Runtime > Change runtime type > T4 GPU**.
3. Run all cells or execute:
   ```bash
   python train_gpu.py --epochs 50 --batch-size 128 --samples-per-epoch 12000
   ```
4. Download the resulting `model_weights.json` or push directly to GitHub.

---

## 🖥️ Local Training & Development (Node.js)

### 1. Install Dependencies
```bash
npm install
```

### 2. Compile Multi-Repository Dataset
Gathers all syntax, tests, standard libraries, and polyglot injection examples across all VerScript organization repositories:
```bash
node build_dataset.js
```

### 3. Run CPU Trainer
```bash
node train.js 10
```
- Tracks live batch-by-batch loss and accuracy (`onBatchEnd`).
- Runs multi-domain LLM evaluation tests after every epoch.
- Automatically saves checkpoints to `model_weights.json`.

---

## 🚀 Running the Server

### Mount into PolyServer
```javascript
const { mountRoutes } = require('./VS-Sharp/server');
mountRoutes(app, '/vs-sharp');
```

### Standalone Server
```bash
node server.js
```
Runs on `http://localhost:3000`.

### Endpoints
- `POST /api/chat`: Generates neural responses and code editing action payloads.
- `GET /status`: Inspects model architecture, parameter count (15M), and backend status.
- `POST /run`: Sandboxed execution of VerScript code via the native compiler binary.
