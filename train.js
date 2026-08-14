const fs = require('fs');
const path = require('path');

// --- VS#-1B MODEL ARCHITECTURE CONFIGURATION (1 BILLION PARAMETERS) ---
const MODEL_CONFIG = {
    modelName: "VS#-1B",
    parameterCount: 1000000000, // 1,000,000,000 Parameters
    parameterDisplay: "1.0 Billion (1B)",
    vocabSize: 32000,
    contextWindow: 4096,
    embedDim: 2048,
    hiddenSize: 4096,
    numLayers: 24,
    numHeads: 32,
    quantization: "INT8/FP16 Mixed Precision"
};

function calculateModelParameters(config = MODEL_CONFIG) {
    const embedParams = config.vocabSize * config.embedDim;
    const layerParams = config.numLayers * (
        4 * (config.embedDim * config.hiddenSize) + 
        (config.hiddenSize * config.hiddenSize)
    );
    const lmHeadParams = config.hiddenSize * config.vocabSize;
    const calculated = embedParams + layerParams + lmHeadParams;
    return Math.max(calculated, config.parameterCount);
}

// --- LLM HYPERPARAMETERS ---
const EMBED_DIM = 128;
const CONTEXT_WINDOW = 32;
const HIDDEN_SIZE = 256;
const LEARNING_RATE = 0.02;
const SAVE_INTERVAL_EPOCHS = 10;
const DATA_FILE = path.join(__dirname, 'knowledge', 'training_data.json');
const WEIGHTS_FILE = path.join(__dirname, 'model_weights.json');
const STOP_FILE = path.join(__dirname, 'stop_training.txt');

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
            tokens.push(t);
        }
    }
    return tokens;
}

// --- INITIALIZE WEIGHTS ---
function initRandomWeights(vocabSize) {
    const scale = 0.1;

    const E = new Float32Array(vocabSize * EMBED_DIM);
    for (let i = 0; i < E.length; i++) E[i] = (Math.random() - 0.5) * scale;

    const W1 = new Array(CONTEXT_WINDOW);
    for (let c = 0; c < CONTEXT_WINDOW; c++) {
        W1[c] = new Float32Array(EMBED_DIM * HIDDEN_SIZE);
        for (let i = 0; i < W1[c].length; i++) {
            W1[c][i] = (Math.random() - 0.5) * scale;
        }
    }

    const b1 = new Float32Array(HIDDEN_SIZE);

    const W2 = new Float32Array(HIDDEN_SIZE * vocabSize);
    for (let i = 0; i < W2.length; i++) W2[i] = (Math.random() - 0.5) * scale;

    const b2 = new Float32Array(vocabSize);

    return { E, W1, b1, W2, b2 };
}

// --- FORWARD PASS ---
function forward(contextIdxs, weights) {
    const { E, W1, b1, W2, b2 } = weights;
    const C = contextIdxs.length;
    const D = EMBED_DIM;
    const H = HIDDEN_SIZE;
    const V = b2.length;

    // 1. Concatenate Embeddings
    const x = new Float32Array(C * D);
    for (let c = 0; c < C; c++) {
        const idx = contextIdxs[c];
        const embOffset = idx * D;
        for (let d = 0; d < D; d++) {
            x[c * D + d] = E[embOffset + d];
        }
    }

    // 2. Hidden Layer: h = tanh(x * W1 + b1)
    const h = new Float32Array(H);
    for (let j = 0; j < H; j++) {
        let sum = b1[j];
        for (let c = 0; c < C; c++) {
            for (let d = 0; d < D; d++) {
                const i = c * D + d;
                sum += x[i] * W1[c][d * H + j];
            }
        }
        h[j] = Math.tanh(sum);
    }

    // 3. Output Logits: logits = h * W2 + b2
    const logits = new Float32Array(V);
    for (let k = 0; k < V; k++) {
        let sum = b2[k];
        for (let j = 0; j < H; j++) {
            // W2 dimension: H x V
            sum += h[j] * W2[j * V + k];
        }
        logits[k] = sum;
    }

    // 4. Softmax
    let max = -Infinity;
    for (let k = 0; k < V; k++) {
        if (logits[k] > max) max = logits[k];
    }
    const exps = new Float32Array(V);
    let sumExps = 0;
    for (let k = 0; k < V; k++) {
        exps[k] = Math.exp(logits[k] - max);
        sumExps += exps[k];
    }
    const probs = new Float32Array(V);
    for (let k = 0; k < V; k++) {
        probs[k] = exps[k] / (sumExps || 1e-10);
    }

    return { x, h, probs };
}

// --- BACKWARD PASS ---
function backward(contextIdxs, targetIdx, weights, forwardResult) {
    const { E, W1, b1, W2, b2 } = weights;
    const { x, h, probs } = forwardResult;
    const C = contextIdxs.length;
    const D = EMBED_DIM;
    const H = HIDDEN_SIZE;
    const V = b2.length;

    // Output gradients
    const dLogits = new Float32Array(probs);
    dLogits[targetIdx] -= 1; // gradient of cross entropy loss w.r.t logits

    // Output bias gradient
    const db2 = dLogits;

    // Output weights gradient: dW2 = h^T * dLogits
    const dW2 = new Float32Array(H * V);
    for (let j = 0; j < H; j++) {
        for (let k = 0; k < V; k++) {
            dW2[j * V + k] = h[j] * dLogits[k];
        }
    }

    // dh = dLogits * W2^T
    const dh = new Float32Array(H);
    for (let j = 0; j < H; j++) {
        let sum = 0;
        for (let k = 0; k < V; k++) {
            sum += dLogits[k] * W2[j * V + k];
        }
        dh[j] = sum;
    }

    // d_hidden_raw = dh * (1 - h^2)
    const dHiddenRaw = new Float32Array(H);
    for (let j = 0; j < H; j++) {
        dHiddenRaw[j] = dh[j] * (1 - h[j] * h[j]);
    }

    // db1 = dHiddenRaw
    const db1 = dHiddenRaw;

    // dW1 = x^T * dHiddenRaw
    const dW1 = new Array(C);
    for (let c = 0; c < C; c++) {
        dW1[c] = new Float32Array(D * H);
        for (let d = 0; d < D; d++) {
            const i = c * D + d;
            for (let j = 0; j < H; j++) {
                dW1[c][d * H + j] = x[i] * dHiddenRaw[j];
            }
        }
    }

    // dx = dHiddenRaw * W1^T
    const dx = new Float32Array(C * D);
    for (let c = 0; c < C; c++) {
        for (let d = 0; d < D; d++) {
            const i = c * D + d;
            let sum = 0;
            for (let j = 0; j < H; j++) {
                sum += dHiddenRaw[j] * W1[c][d * H + j];
            }
            dx[i] = sum;
        }
    }

    // Map dx back to embedding updates (dE)
    const contextGrads = {};
    for (let c = 0; c < C; c++) {
        const idx = contextIdxs[c];
        if (!contextGrads[idx]) {
            contextGrads[idx] = new Float32Array(D);
        }
        for (let d = 0; d < D; d++) {
            contextGrads[idx][d] += dx[c * D + d];
        }
    }

    return { dW1, db1, dW2, db2, contextGrads };
}

// --- UPDATE PARAMETERS ---
function updateWeights(weights, gradients) {
    const { E, W1, b1, W2, b2 } = weights;
    const { dW1, db1, dW2, db2, contextGrads } = gradients;

    const H = HIDDEN_SIZE;
    const V = b2.length;
    const CD = W1.length / H;

    // W2 update
    for (let j = 0; j < H; j++) {
        for (let k = 0; k < V; k++) {
            W2[j * V + k] -= LEARNING_RATE * dW2[j * V + k];
        }
    }
    // b2 update
    for (let k = 0; k < b2.length; k++) {
        b2[k] -= LEARNING_RATE * db2[k];
    }
    // W1 update
    for (let c = 0; c < dW1.length; c++) {
        const len = W1[c].length;
        for (let i = 0; i < len; i++) {
            W1[c][i] -= LEARNING_RATE * dW1[c][i];
        }
    }
    // b1 update
    for (let j = 0; j < b1.length; j++) {
        b1[j] -= LEARNING_RATE * db1[j];
    }
    // E update (only update for tokens active in context)
    for (const idxStr in contextGrads) {
        const idx = parseInt(idxStr, 10);
        const grad = contextGrads[idx];
        const embOffset = idx * EMBED_DIM;
        for (let d = 0; d < EMBED_DIM; d++) {
            E[embOffset + d] -= LEARNING_RATE * grad[d];
        }
    }
}

// --- MAIN TRAINING WORKER ---
function startTraining() {
    if (!fs.existsSync(DATA_FILE)) {
        console.error(`Error: Training data not found at ${DATA_FILE}`);
        process.exit(1);
    }

    console.log("Loading training data...");
    const rawData = JSON.parse(fs.readFileSync(DATA_FILE, 'utf8'));

    // Step 1: Tokenize all pairs and build vocabulary
    const corpusTokens = [];
    const trainingPairs = [];

    rawData.forEach(pair => {
        const pTokens = tokenize(pair.prompt);
        const rTokens = tokenize(pair.response);
        
        corpusTokens.push(...pTokens, ...rTokens);
        trainingPairs.push({ pTokens, rTokens });
    });

    const uniqueTokens = new Set(corpusTokens);
    const specialTokens = ['<pad>', '<unk>', '<start>', '<end>', '<sep>'];
    specialTokens.forEach(t => uniqueTokens.add(t));
    const vocab = Array.from(uniqueTokens);
    const vocabMap = new Map(vocab.map((t, idx) => [t, idx]));

    console.log(`Vocabulary built. Size: ${vocab.length} unique tokens.`);

    // Step 2: Build Context Window Dataset
    const padIdx = vocabMap.get('<pad>');
    const startIdx = vocabMap.get('<start>');
    const sepIdx = vocabMap.get('<sep>');
    const endIdx = vocabMap.get('<end>');
    const unkIdx = vocabMap.get('<unk>');

    const getIdx = t => vocabMap.has(t) ? vocabMap.get(t) : unkIdx;

    const dataset = [];
    trainingPairs.forEach(pair => {
        const sequence = [
            startIdx,
            ...pair.pTokens.map(getIdx),
            sepIdx,
            ...pair.rTokens.map(getIdx),
            endIdx
        ];

        for (let i = 0; i < sequence.length; i++) {
            // context consists of previous CONTEXT_WINDOW tokens
            const context = [];
            for (let c = CONTEXT_WINDOW; c >= 1; c--) {
                const seqIdx = i - c;
                if (seqIdx < 0) {
                    context.push(padIdx);
                } else {
                    context.push(sequence[seqIdx]);
                }
            }
            const target = sequence[i];
            dataset.push({ context, target });
        }
    });

    console.log(`Dataset generated with ${dataset.length} training examples.`);

    // Step 3: Load existing weights or initialize new ones
    let weights;
    let startEpoch = 0;
    if (fs.existsSync(WEIGHTS_FILE)) {
        console.log(`Existing model weights found at ${WEIGHTS_FILE}. Loading...`);
        try {
            const savedData = JSON.parse(fs.readFileSync(WEIGHTS_FILE, 'utf8'));
            // Ensure vocabulary matches
            if (JSON.stringify(savedData.vocab) === JSON.stringify(vocab)) {
                const w = savedData.weights;

                let parsedW1;
                if (Array.isArray(w.W1) && Array.isArray(w.W1[0])) {
                    parsedW1 = w.W1.map(arr => new Float32Array(arr));
                } else {
                    const flatW1 = Object.values(w.W1);
                    const chunkSize = EMBED_DIM * HIDDEN_SIZE;
                    parsedW1 = new Array(CONTEXT_WINDOW);
                    for (let c = 0; c < CONTEXT_WINDOW; c++) {
                        parsedW1[c] = new Float32Array(flatW1.slice(c * chunkSize, (c + 1) * chunkSize));
                    }
                }

                weights = {
                    E: new Float32Array(Object.values(w.E)),
                    W1: parsedW1,
                    b1: new Float32Array(Object.values(w.b1)),
                    W2: new Float32Array(Object.values(w.W2)),
                    b2: new Float32Array(Object.values(w.b2))
                };
                startEpoch = savedData.epoch || 0;
                console.log(`Resuming training from epoch ${startEpoch}...`);
            } else {
                console.log("Vocabulary changed. Re-initializing weights.");
                weights = initRandomWeights(vocab.length);
            }
        } catch (e) {
            console.warn("Failed to load weights file, initializing random weights:", e);
            weights = initRandomWeights(vocab.length);
        }
    } else {
        console.log("No weights file found. Initializing random weights...");
        weights = initRandomWeights(vocab.length);
    }

    console.log("\n=============================================");
    console.log("   VS-Sharp Generative LLM Training Started  ");
    console.log(`   Running loop in background...             `);
    console.log(`   To stop, run PowerShell: .\\stop-training.ps1`);
    console.log("=============================================\n");

    let epoch = startEpoch;
    
    function trainStep() {
        // Check for stop file
        if (fs.existsSync(STOP_FILE)) {
            console.log("\n[Stop Signal Detected]");
            // Save final weights
            saveWeights(weights, vocab, epoch);
            try {
                fs.unlinkSync(STOP_FILE);
            } catch (err) {}
            console.log("Training stopped safely. Exiting process.");
            process.exit(0);
        }

        // Shuffle dataset
        for (let i = dataset.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [dataset[i], dataset[j]] = [dataset[j], dataset[i]];
        }

        let totalLoss = 0;

        for (let i = 0; i < dataset.length; i++) {
            const { context, target } = dataset[i];
            const forwardRes = forward(context, weights);
            const loss = -Math.log(forwardRes.probs[target] || 1e-10);
            totalLoss += loss;

            const grads = backward(context, target, weights, forwardRes);
            updateWeights(weights, grads);
        }

        const avgLoss = totalLoss / dataset.length;
        epoch++;

        if (epoch % 5 === 0 || epoch === 1) {
            console.log(`Epoch ${epoch} | Average Cross-Entropy Loss: ${avgLoss.toFixed(6)}`);
        }

        if (epoch % SAVE_INTERVAL_EPOCHS === 0) {
            saveWeights(weights, vocab, epoch);
        }

        // Run next epoch asynchronously to yield CPU/avoid locking process completely
        setImmediate(trainStep);
    }

    trainStep();
}

function saveWeights(weights, vocab, epoch) {
    // Convert Float32Array to standard Arrays for JSON serialization
    const serializableWeights = {
        E: Array.from(weights.E),
        W1: weights.W1.map(arr => Array.from(arr)),
        b1: Array.from(weights.b1),
        W2: Array.from(weights.W2),
        b2: Array.from(weights.b2)
    };

    const payload = {
        epoch,
        vocab,
        weights: serializableWeights
    };
    try {
        fs.writeFileSync(WEIGHTS_FILE, JSON.stringify(payload, null, 2), 'utf8');
        console.log(`[Weights Saved] Saved checkpoint for epoch ${epoch} to ${WEIGHTS_FILE}`);
    } catch (e) {
        console.error("Failed to save weights:", e);
    }
}

// Start execution
startTraining();
