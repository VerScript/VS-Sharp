// VS-Sharp — 15M Deep Neural Language Model + VerScript Runner
// Mounts all routes onto an Express app to be consumed by PolyServer or run standalone.

const fs = require('fs');
const path = require('path');
const { execFile } = require('child_process');
const os = require('os');

let tf;
try {
    tf = require('@tensorflow/tfjs-node');
} catch (_) {
    tf = require('@tensorflow/tfjs');
}

const WEIGHTS_FILE = path.join(__dirname, 'model_weights.json');

// --- VS#-15M DEFAULT CONFIGURATION ---
const DEFAULT_CONFIG = {
    modelName: "VS#-15M",
    parameterCount: 15000000,
    parameterDisplay: "15.0 Million (15M)",
    architecture: "VS#-15M Deep Neural Language Model & Polyglot Syntax Engine",
    vocabSize: 6500,
    contextWindow: 48,
    embedDim: 256,
    hiddenSize: 780,
    hiddenSize2: 512,
    quantization: "FP32"
};

// --- TOKENIZER ---
function tokenize(text) {
    const tokens = [];
    const regex = /(\r?\n|[a-zA-Z0-9_]+|[^\s\w])/g;
    let match;
    while ((match = regex.exec(text)) !== null) {
        let t = match[0];
        if (t === '\r\n' || t === '\n') {
            tokens.push('\n');
        } else {
            tokens.push(t);
        }
    }
    return tokens;
}

// --- TENSORFLOW 15M FORWARD PASS ---
function forwardTF(contextIdxs, tfWeights) {
    return tf.tidy(() => {
        const { E, W1, b1, W2, b2, W3, b3 } = tfWeights;

        // 1. Embedding lookup: [C, EMBED_DIM]
        const indices = tf.tensor1d(contextIdxs, 'int32');
        const x = tf.gather(E, indices);

        // 2. Flatten: [1, C * EMBED_DIM]
        const xFlat = x.reshape([1, -1]);

        // 3. Dense Layer 1 (tanh): [1, H1]
        const h1 = tf.tanh(tf.add(tf.matMul(xFlat, W1), b1));

        // 4. Dense Layer 2 (tanh): [1, H2]
        const h2 = tf.tanh(tf.add(tf.matMul(h1, W2), b2));

        // 5. Output Logits: [1, vocabSize]
        const logits = tf.add(tf.matMul(h2, W3), b3);

        // 6. Softmax
        const probs = tf.softmax(logits).squeeze();
        return probs.arraySync();
    });
}

// --- NEURAL AUTOREGRESSIVE TEXT GENERATION ---
function generateLLMResponse(message, weightsData) {
    const { vocab, config, weights } = weightsData;
    const vocabMap = new Map(vocab.map((t, idx) => [t, idx]));

    const padIdx = vocabMap.get('<pad>') ?? 0;
    const startIdx = vocabMap.get('<start>') ?? 1;
    const sepIdx = vocabMap.get('<sep>') ?? 2;
    const endIdx = vocabMap.get('<end>') ?? 3;
    const unkIdx = vocabMap.get('<unk>') ?? 4;

    const getIdx = t => vocabMap.has(t) ? vocabMap.get(t) : unkIdx;

    const contextWindow = config?.contextWindow || 48;
    const embedDim = config?.embedDim || 256;
    const hiddenSize1 = config?.hiddenSize || 780;
    const hiddenSize2 = config?.hiddenSize2 || 512;
    const vocabSize = vocab.length;

    const promptTokens = tokenize(message);
    const sequenceIdxs = [startIdx];
    for (let token of promptTokens) {
        sequenceIdxs.push(getIdx(token));
    }
    sequenceIdxs.push(sepIdx);

    const generatedTokens = [];
    const maxGenLength = 160;

    // Convert weights to tensors
    const tfWeights = tf.tidy(() => {
        const E = tf.tensor2d(weights.E, [vocabSize, embedDim]);
        const W1 = tf.tensor2d(weights.W1, [contextWindow * embedDim, hiddenSize1]);
        const b1 = tf.tensor1d(weights.b1);
        const W2 = tf.tensor2d(weights.W2, [hiddenSize1, hiddenSize2]);
        const b2 = tf.tensor1d(weights.b2);
        const W3 = tf.tensor2d(weights.W3, [hiddenSize2, vocabSize]);
        const b3 = tf.tensor1d(weights.b3);

        return { E, W1, b1, W2, b2, W3, b3 };
    });

    try {
        for (let step = 0; step < maxGenLength; step++) {
            const context = [];
            for (let c = contextWindow; c >= 1; c--) {
                const seqIdx = sequenceIdxs.length - c;
                context.push(seqIdx < 0 ? padIdx : sequenceIdxs[seqIdx]);
            }

            const probs = forwardTF(context, tfWeights);

            // Temperature sampling
            const temp = 0.75;
            const logProbs = probs.map(p => Math.log(p + 1e-10) / temp);
            const maxLog = Math.max(...logProbs);
            const tempExps = logProbs.map(lp => Math.exp(lp - maxLog));
            const tempSum = tempExps.reduce((a, b) => a + b, 0);
            const tempProbs = tempExps.map(te => te / (tempSum || 1e-10));

            const r = Math.random();
            let cumulative = 0;
            let nextIdx = endIdx;
            for (let i = 0; i < tempProbs.length; i++) {
                cumulative += tempProbs[i];
                if (r <= cumulative) {
                    nextIdx = i;
                    break;
                }
            }

            if (nextIdx === endIdx) break;

            sequenceIdxs.push(nextIdx);
            generatedTokens.push(vocab[nextIdx]);
        }
    } finally {
        tf.dispose(tfWeights);
    }

    // Decode generated tokens into clean text
    let responseText = "";
    generatedTokens.forEach((t) => {
        if (t === '\n') {
            responseText += '\n';
        } else {
            if (
                responseText.length > 0 &&
                !responseText.endsWith('\n') &&
                !responseText.endsWith('`') &&
                !responseText.endsWith('(') &&
                !responseText.endsWith('[') &&
                !responseText.endsWith('{') &&
                t !== '.' &&
                t !== ',' &&
                t !== '!' &&
                t !== '?' &&
                t !== ':' &&
                t !== ';' &&
                t !== ')' &&
                t !== ']' &&
                t !== '}'
            ) {
                responseText += ' ';
            }
            responseText += t;
        }
    });

    return responseText.trim();
}

// --- EXTRACT CODE BLOCK FOR IDE BUFFER INJECTION ---
function extractCodeBlock(text) {
    if (!text) return null;
    // Match language-tagged or untagged code blocks
    const match = text.match(/```(?:[a-zA-Z0-9_-]+)?\r?\n([\s\S]*?)```/);
    if (match && match[1].trim()) {
        return match[1].trim();
    }
    return null;
}

// --- RESOLVE VERSCRIPT BINARY ---
function resolveVerScriptBinary() {
    const isWin = os.platform() === 'win32';
    const binName = isWin ? 'verscript.exe' : 'verscript';

    const candidates = [
        path.join(process.cwd(), binName),
        path.join(__dirname, binName),
        path.join(__dirname, '..', 'VerScript', binName),
        path.join(__dirname, '..', 'polyserver', binName)
    ];

    for (const cand of candidates) {
        if (fs.existsSync(cand)) return cand;
    }
    return null;
}

// --- MOUNT ROUTES FOR EXPRESS / POLYSERVER ---
function mountRoutes(app, prefix = '') {
    // 1. Status Endpoint
    app.get(prefix + '/status', (req, res) => {
        let loadedConfig = DEFAULT_CONFIG;
        let epoch = 0;
        let loss = 0;
        let accuracy = 0;

        if (fs.existsSync(WEIGHTS_FILE)) {
            try {
                const w = JSON.parse(fs.readFileSync(WEIGHTS_FILE, 'utf8'));
                if (w.config) loadedConfig = { ...DEFAULT_CONFIG, ...w.config };
                if (w.epoch !== undefined) epoch = w.epoch;
                if (w.loss !== undefined) loss = w.loss;
                if (w.accuracy !== undefined) accuracy = w.accuracy;
            } catch (_) {}
        }

        res.json({
            status: 'online',
            ...loadedConfig,
            epoch,
            loss,
            accuracy,
            backend: tf.getBackend(),
            weightsLoaded: fs.existsSync(WEIGHTS_FILE)
        });
    });

    // 2. Chat API — ALWAYS RETURNS PURE GENERATED NEURAL ANSWER
    app.post(prefix + '/api/chat', (req, res) => {
        const { message, code } = req.body;

        if (!message) {
            return res.status(400).json({ error: 'Message is required' });
        }

        console.log(`[VS#-15M] Input received: "${message}"`);

        // Check if weights file exists
        if (!fs.existsSync(WEIGHTS_FILE)) {
            return res.json({
                response: "VS#-15M model weights are currently initializing or training. Please run 'node train.js' to start learning.",
                action: null
            });
        }

        try {
            const weightsData = JSON.parse(fs.readFileSync(WEIGHTS_FILE, 'utf8'));

            // ALWAYS generate dynamically using the neural language model
            // No hardcoded or pre-written answers are allowed
            const responseText = generateLLMResponse(message, weightsData);

            // Extract code block for IDE integration if model generated code
            const codeBlock = extractCodeBlock(responseText);
            let actionPayload = null;
            if (codeBlock) {
                actionPayload = {
                    type: "edit",
                    code: codeBlock
                };
            }

            res.json({
                response: responseText,
                action: actionPayload
            });

        } catch (err) {
            console.error("[VS#-15M] Error generating response:", err);
            res.status(500).json({ error: 'Internal server error running 15M LLM: ' + err.message });
        }
    });

    // 3. VerScript Code Runner Endpoint
    app.post(prefix + '/run', (req, res) => {
        const { code } = req.body;

        if (typeof code !== 'string') {
            return res.status(400).json({ error: 'Code string is required.' });
        }

        const bin = resolveVerScriptBinary();
        if (!bin) {
            return res.status(500).json({ error: 'VerScript binary not found on server.' });
        }

        try { fs.chmodSync(bin, 0o755); } catch (_) {}

        const tmpFile = path.join(os.tmpdir(), `vrs_${Date.now()}_${Math.random().toString(36).substring(7)}.vrs`);
        fs.writeFileSync(tmpFile, code, 'utf8');

        execFile(bin, [tmpFile], { timeout: 10000, cwd: path.dirname(bin) }, (error, stdout, stderr) => {
            try { fs.unlinkSync(tmpFile); } catch (_) {}

            if (error && error.killed) {
                return res.json({ output: stdout || '', error: 'Execution timed out (10s limit).' });
            }

            res.json({
                output: stdout || '',
                error: stderr || (error && !stdout ? error.message : '') || ''
            });
        });
    });
}

// --- STANDALONE SERVER ---
if (require.main === module) {
    const express = require('express');
    const cors = require('cors');
    const app = express();

    app.use(cors());
    app.use(express.json());

    mountRoutes(app);

    const PORT = process.env.PORT || 3000;
    app.listen(PORT, () => {
        console.log(`[VS#-15M] Server running on port ${PORT}`);
    });
}

module.exports = { mountRoutes, generateLLMResponse, forwardTF, tokenize, DEFAULT_CONFIG };
