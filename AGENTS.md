# AGENTS.md — VS-Sharp

Instructions and context for autonomous AI coding agents (such as Google Jules).

## 🚀 Dev Environment & Commands
- **Install Dependencies**: `npm install`
- **Mount Routes**: Exported via `module.exports = { mountRoutes }` to be consumed by PolyServer.
- **Run Standalone (Local Test)**: `node server.js`

## 🧠 Neural Language Model (LLM) & Training
- **Training Data**: Located in `knowledge/training_data.json` as prompt-response pairs.
- **Background Trainer**: Run `node train.js` to start training the neural network from scratch.
- **Stop Command**: To stop training safely, run the PowerShell script `.\stop-training.ps1` or write a `stop_training.txt` file in the directory.
- **Shared Weights State**: Saves checkpoints dynamically to `model_weights.json`.
- **Model Architecture**: Multi-Layer Perceptron (MLP) built completely from scratch using standard JavaScript (no TensorFlow/PyTorch).
  - `E` (Embeddings): size `[vocab_size, EMBED_DIM (32)]`
  - `W1` (Hidden weights): size `[CONTEXT_WINDOW (8) * EMBED_DIM (32), HIDDEN_SIZE (64)]`
  - `b1` (Hidden biases): size `[HIDDEN_SIZE (64)]`
  - `W2` (Output weights): size `[HIDDEN_SIZE (64), vocab_size]`
  - `b2` (Output biases): size `[vocab_size]`

## ⚙️ Path Resolution Rules
- The C interpreter binary `verscript` is executed by the `/run` endpoint.
- **Binary Resolution**: Always resolve the binary path using:
  `const srcBin = path.join(process.cwd(), 'verscript_src', 'verscript');`
  `const rootBin = path.join(process.cwd(), 'verscript');`
  `const localBin = path.join(__dirname, 'verscript');`
  `const VERSCRIPT_BIN = fs.existsSync(srcBin) ? srcBin : (fs.existsSync(rootBin) ? rootBin : localBin);`
- Always verify read/write access and permissions before running.

## 🛡️ Coding Guidelines & Rules
- **No Third-Party ML Frameworks**: Absolutely no TensorFlow, PyTorch, or external API wrappers (OpenAI, Gemini) allowed. All network layers, backpropagation, and tokenizer must remain custom JS logic.
- **Direct IDE Codespace Writing**: Ensure that if the AI assistant generates a response containing a ````verscript` code block, it compiles a payload in the JSON response:
  ```json
  {
    "response": "Text explanation...",
    "action": {
      "type": "edit",
      "code": "extracted_code_block"
    }
  }
  ```
- **Tokenizer Consistency**: Keep tokenizer matching logic (`tokenize()`) identical between `train.js` and `server.js` to prevent index mismatch errors.
