# AGENTS.md — VS-Sharp

Instructions and context for autonomous AI coding agents.

## 🚀 Dev Environment & Commands
- **Install Dependencies**: `npm install`
- **Mount Routes**: Exported via `module.exports = { mountRoutes }` to be consumed by PolyServer.
- **Run Standalone (Local Test)**: `node server.js`
- **Dataset Generation**: `node build_dataset.js` (Ingests all VerScript organization repos and polyglot languages).
- **Run Trainer**: `node train.js` (Trains the 15M parameter neural language model).

## 🧠 Neural Language Model (LLM) & Training
- **Model Architecture**: **VS#-15M** (15 Million Parameters) deep neural network language model.
  - Vocabulary: Dynamic across all VerScript org repos and polyglot codes.
  - `Context Window`: 48 tokens.
  - `Embedding Dimension (E)`: 256.
  - `Hidden Layer 1 (W1)`: Dynamically calculated to target exactly 15,000,000 parameters.
  - `Hidden Layer 2 (W2)`: 512 units (tanh).
  - `Output Projection (W3)`: Logits over vocabulary size.
- **Training Data**: Located in `knowledge/training_data.json` containing prompt-response pairs from:
  - VerScript standard libraries, tests, and syntax examples.
  - Polyglot languages accessible via the `inject` keyword (Python, JavaScript, C, HTML/CSS).
  - Conversational speech, debugging, and line-by-line code explanations.
- **Weights State**: Saves checkpoints dynamically to `model_weights.json`.
- **Stop Command**: To stop training safely, create a `stop_training.txt` file or run `.\stop-training.ps1`.

## ⚙️ Path Resolution Rules
- The C interpreter binary `verscript` / `verscript.exe` is executed by the `/run` endpoint.
- Binary resolution searches:
  1. `process.cwd()/verscript`
  2. `__dirname/verscript`
  3. `../VerScript/verscript`
  4. `../polyserver/verscript`

## 🛡️ Coding Guidelines & Rules
- **Pure Generated Responses Only**: VS# must NEVER return pre-written or canned answers. All conversational and code responses must be produced directly by the neural language model generator, no matter how gibberish or experimental.
- **Action Payloads for IDE**: If the generated response contains a fenced code block (` ``` `), extract it and include it in the `action` payload:
  ```json
  {
    "response": "...",
    "action": {
      "type": "edit",
      "code": "extracted_code_block"
    }
  }
  ```
- **Tokenizer Consistency**: Keep `tokenize()` identical across `train.js`, `build_dataset.js`, and `server.js`.
