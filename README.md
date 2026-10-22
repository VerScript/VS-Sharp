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

## ⚡ Fast Cloud GPU Training (Kaggle & Colab)

Training 15 million parameters on CPU is computationally intensive (~40 min/epoch). With cloud GPUs (Tesla T4 or P100), training takes **only ~3 to 5 seconds per epoch**!

### Option A: Kaggle Notebooks (30 Hours/Week FREE GPU)
Kaggle provides **30 hours per week of completely free GPU compute** with **Tesla T4 x2** or **Tesla P100** (no credit limits!):

1. Go to [kaggle.com](https://www.kaggle.com) and click **Create > New Notebook**.
2. Click **File > Import Notebook** and upload [`kaggle_train.ipynb`](https://github.com/VerScript/VS-Sharp/blob/main/kaggle_train.ipynb) (or select **Upload from URL**).
3. In the right-hand panel under **Settings**:
   - **Accelerator**: Select **GPU T4 x2** or **GPU P100**.
   - **Internet**: Toggle to **Internet on** (required to clone from GitHub).
4. Run all cells (`Shift + Enter`).
5. After training completes, look at the right sidebar under **Data > Output** to download `model_weights.json.gz` with a single click!

---

### Option B: Google Colab (Tesla T4)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/VerScript/VS-Sharp/blob/main/colab_train.ipynb)

1. Open [`colab_train.ipynb`](https://colab.research.google.com/github/VerScript/VS-Sharp/blob/main/colab_train.ipynb) in Colab.
2. Select **Runtime > Change runtime type > T4 GPU**.
3. Run all cells or execute:
   ```bash
   python train_gpu.py --epochs 50 --batch-size 128 --samples-per-epoch 15000
   ```
4. Download `model_weights.json.gz` via Section 7 or push directly to GitHub via Section 8.

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
