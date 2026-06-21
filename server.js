const express = require('express');
const cors = require('cors');
const fs = require('fs');
const path = require('path');
const { exec } = require('child_process');
const os = require('os');

const app = express();

// Allow IDE and landing page origins
app.use(cors({
    origin: [
        'https://verscript.github.io',
        'https://vs-sharp.onrender.com'
    ],
    methods: ['GET', 'POST'],
    allowedHeaders: ['Content-Type']
}));
app.use(express.json({ limit: '1mb' }));

const PORT = process.env.PORT || 3001;
const WEIGHTS_FILE = path.join(__dirname, 'model_weights.json');
const CONTEXT_WINDOW = 3;
const EMBED_DIM = 16;
const HIDDEN_SIZE = 32;

// --- TOKENIZER ---
function tokenize(text) {
    const tokens = [];
    const regex = /(\r?\n|\w+|[^\w\s])/g;
    let match;
    while ((match = regex.exec(text)) !== null) {
        let t = match[0];
        if (t === '\r\n' || t === '\n') {
            tokens.push('\n');
        } else {
            tokens.push(t.toLowerCase());
        }
    }
    return tokens;
}

// --- FORWARD PASS FOR INFERENCE ---
function forward(contextIdxs, weights) {
    const { E, W1, b1, W2, b2 } = weights;
    const C = contextIdxs.length;
    const D = EMBED_DIM;
    const H = HIDDEN_SIZE;
    const V = b2.length;

    // 1. Concatenate Embeddings
    const x = new Array(C * D);
    for (let c = 0; c < C; c++) {
        const idx = contextIdxs[c];
        // Handle out-of-bounds or unknown index safely
        const emb = E[idx] || E[0] || new Array(D).fill(0);
        for (let d = 0; d < D; d++) {
            x[c * D + d] = emb[d];
        }
    }

    // 2. Hidden Layer: h = tanh(x * W1 + b1)
    const h = new Array(H);
    for (let j = 0; j < H; j++) {
        let sum = b1[j];
        for (let i = 0; i < C * D; i++) {
            sum += x[i] * (W1[i] ? W1[i][j] : 0);
        }
        h[j] = Math.tanh(sum);
    }

    // 3. Output Logits: logits = h * W2 + b2
    const logits = new Array(V);
    for (let k = 0; k < V; k++) {
        let sum = b2[k];
        for (let j = 0; j < H; j++) {
            sum += h[j] * (W2[j] ? W2[j][k] : 0);
        }
        logits[k] = sum;
    }

    // 4. Softmax
    const max = Math.max(...logits);
    const exps = logits.map(v => Math.exp(v - max));
    const sumExps = exps.reduce((a, b) => a + b, 0);
    const probs = exps.map(v => v / (sumExps || 1e-10));

    return probs;
}

// --- GENERATE RESPONSE FROM LLM ---
function generateLLMResponse(message, weightsData) {
    const { vocab, weights } = weightsData;
    const vocabMap = new Map(vocab.map((t, idx) => [t, idx]));
    
    const padIdx = vocabMap.get('<pad>');
    const startIdx = vocabMap.get('<start>');
    const sepIdx = vocabMap.get('<sep>');
    const endIdx = vocabMap.get('<end>');
    const unkIdx = vocabMap.get('<unk>');

    const getIdx = t => vocabMap.has(t) ? vocabMap.get(t) : unkIdx;

    // Tokenize prompt
    const promptTokens = tokenize(message);
    const sequenceIdxs = [
        startIdx,
        ...promptTokens.map(getIdx),
        sepIdx
    ];

    const generatedTokens = [];
    const maxGenLength = 150;

    for (let step = 0; step < maxGenLength; step++) {
        // Prepare context
        const context = [];
        for (let c = CONTEXT_WINDOW; c >= 1; c--) {
            const seqIdx = sequenceIdxs.length - c;
            if (seqIdx < 0) {
                context.push(padIdx);
            } else {
                context.push(sequenceIdxs[seqIdx]);
            }
        }

        // Forward pass to get probs
        const probs = forward(context, weights);

        // Softmax sampling (with low temperature to keep output coherent)
        // Apply a small temperature scaling
        const temp = 0.3;
        const logProbs = probs.map(p => Math.log(p + 1e-10) / temp);
        const maxLog = Math.max(...logProbs);
        const tempExps = logProbs.map(lp => Math.exp(lp - maxLog));
        const tempSum = tempExps.reduce((a, b) => a + b, 0);
        const tempProbs = tempExps.map(te => te / (tempSum || 1e-10));

        // Sample token
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

        if (nextIdx === endIdx) {
            break;
        }

        sequenceIdxs.push(nextIdx);
        generatedTokens.push(vocab[nextIdx]);
    }

    // Decode generated tokens
    let responseText = "";
    generatedTokens.forEach((t, i) => {
        if (t === '\n') {
            responseText += '\n';
        } else {
            // Add space between tokens, except for newlines or start of text
            if (responseText.length > 0 && !responseText.endsWith('\n') && t !== '.' && t !== ',' && t !== '!' && t !== '?') {
                responseText += ' ';
            }
            responseText += t;
        }
    });

    return responseText;
}

// --- EXTRACT CODE BLOCK ---
function extractCodeBlock(text) {
    const regex = /```verscript\r?\n([\s\S]*?)```/i;
    const match = text.match(regex);
    if (match) {
        return match[1].trim();
    }
    
    // Fallback if formatting lacks language tag but has ticks
    const fallbackRegex = /```\r?\n([\s\S]*?)```/;
    const fallbackMatch = text.match(fallbackRegex);
    return fallbackMatch ? fallbackMatch[1].trim() : null;
}

// --- VERSCRIPT CODE RUNNER ---
// Uses the compiled Linux binary from the VerScript repo
const VERSCRIPT_BIN = path.join(__dirname, 'verscript');

app.post('/run', async (req, res) => {
    const { code } = req.body;
    if (typeof code !== 'string') {
        return res.status(400).json({ error: 'code (string) is required' });
    }

    // Write code to a temp file
    const tmpFile = path.join(os.tmpdir(), `vs_${Date.now()}_${Math.random().toString(36).slice(2)}.vrs`);
    try {
        fs.writeFileSync(tmpFile, code, 'utf8');
    } catch (err) {
        return res.status(500).json({ error: 'Failed to write temp file', detail: err.message });
    }

    // Check binary exists
    if (!fs.existsSync(VERSCRIPT_BIN)) {
        fs.unlinkSync(tmpFile);
        return res.status(500).json({ error: 'VerScript binary not found on server. Please ensure verscript is deployed.' });
    }

    // Execute with a 10-second timeout
    exec(`"${VERSCRIPT_BIN}" "${tmpFile}"`, { timeout: 10000 }, (error, stdout, stderr) => {
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

// --- HEALTH CHECK ---
app.get('/ping', (req, res) => {
    res.send('pong');
});

app.post('/api/chat', (req, res) => {
    const { code, message } = req.body;
    
    if (!message) {
        return res.status(400).json({ error: 'Message is required' });
    }

    console.log(`Received user message: "${message}"`);

    // Load weights dynamically
    if (!fs.existsSync(WEIGHTS_FILE)) {
        return res.json({
            response: "### 🤖 VS# Language Model Initializing\n\nI am currently training my neural network from scratch in the background. Please wait a few seconds and send your message again!",
            action: null
        });
    }

    try {
        const weightsData = JSON.parse(fs.readFileSync(WEIGHTS_FILE, 'utf8'));
        const responseText = generateLLMResponse(message, weightsData);
        
        let actionPayload = null;
        const codeBlock = extractCodeBlock(responseText);
        if (codeBlock) {
            actionPayload = {
                type: "edit",
                code: codeBlock
            };
        }

        // Simulate typing delay
        setTimeout(() => {
            res.json({
                response: responseText,
                action: actionPayload
            });
        }, 600);

    } catch (err) {
        console.error("Error generating LLM response:", err);
        res.status(500).json({ error: 'Internal server error running custom LLM.' });
    }
});

app.listen(PORT, () => {
    console.log(`VS# Helper AI running on port ${PORT}`);
});
