const fs = require('fs');
const path = require('path');

// --- LLM HYPERPARAMETERS ---
const EMBED_DIM = 16;
const CONTEXT_WINDOW = 3;
const HIDDEN_SIZE = 32;
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
            tokens.push(t.toLowerCase());
        }
    }
    return tokens;
}

// --- INITIALIZE WEIGHTS ---
function initRandomWeights(vocabSize) {
    const scale = 0.1;
    const E = Array.from({ length: vocabSize }, () => 
        Array.from({ length: EMBED_DIM }, () => (Math.random() - 0.5) * scale)
    );
    const W1 = Array.from({ length: CONTEXT_WINDOW * EMBED_DIM }, () => 
        Array.from({ length: HIDDEN_SIZE }, () => (Math.random() - 0.5) * scale)
    );
    const b1 = new Array(HIDDEN_SIZE).fill(0);
    const W2 = Array.from({ length: HIDDEN_SIZE }, () => 
        Array.from({ length: vocabSize }, () => (Math.random() - 0.5) * scale)
    );
    const b2 = new Array(vocabSize).fill(0);

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
    const x = new Array(C * D);
    for (let c = 0; c < C; c++) {
        const idx = contextIdxs[c];
        const emb = E[idx];
        for (let d = 0; d < D; d++) {
            x[c * D + d] = emb[d];
        }
    }

    // 2. Hidden Layer: h = tanh(x * W1 + b1)
    const h = new Array(H);
    for (let j = 0; j < H; j++) {
        let sum = b1[j];
        for (let i = 0; i < C * D; i++) {
            sum += x[i] * W1[i][j];
        }
        h[j] = Math.tanh(sum);
    }

    // 3. Output Logits: logits = h * W2 + b2
    const logits = new Array(V);
    for (let k = 0; k < V; k++) {
        let sum = b2[k];
        for (let j = 0; j < H; j++) {
            sum += h[j] * W2[j][k];
        }
        logits[k] = sum;
    }

    // 4. Softmax
    const max = Math.max(...logits);
    const exps = logits.map(v => Math.exp(v - max));
    const sumExps = exps.reduce((a, b) => a + b, 0);
    const probs = exps.map(v => v / (sumExps || 1e-10));

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
    const dLogits = probs.slice();
    dLogits[targetIdx] -= 1; // gradient of cross entropy loss w.r.t logits

    // Output bias gradient
    const db2 = dLogits;

    // Output weights gradient: dW2 = h^T * dLogits
    const dW2 = Array.from({ length: H }, () => new Array(V));
    for (let j = 0; j < H; j++) {
        for (let k = 0; k < V; k++) {
            dW2[j][k] = h[j] * dLogits[k];
        }
    }

    // dh = dLogits * W2^T
    const dh = new Array(H).fill(0);
    for (let j = 0; j < H; j++) {
        let sum = 0;
        for (let k = 0; k < V; k++) {
            sum += dLogits[k] * W2[j][k];
        }
        dh[j] = sum;
    }

    // d_hidden_raw = dh * (1 - h^2)
    const dHiddenRaw = new Array(H);
    for (let j = 0; j < H; j++) {
        dHiddenRaw[j] = dh[j] * (1 - h[j] * h[j]);
    }

    // db1 = dHiddenRaw
    const db1 = dHiddenRaw;

    // dW1 = x^T * dHiddenRaw
    const dW1 = Array.from({ length: C * D }, () => new Array(H));
    for (let i = 0; i < C * D; i++) {
        for (let j = 0; j < H; j++) {
            dW1[i][j] = x[i] * dHiddenRaw[j];
        }
    }

    // dx = dHiddenRaw * W1^T
    const dx = new Array(C * D).fill(0);
    for (let i = 0; i < C * D; i++) {
        let sum = 0;
        for (let j = 0; j < H; j++) {
            sum += dHiddenRaw[j] * W1[i][j];
        }
        dx[i] = sum;
    }

    // Map dx back to embedding updates (dE)
    const contextGrads = {};
    for (let c = 0; c < C; c++) {
        const idx = contextIdxs[c];
        if (!contextGrads[idx]) {
            contextGrads[idx] = new Array(D).fill(0);
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

    // W2 update
    for (let j = 0; j < W2.length; j++) {
        for (let k = 0; k < W2[j].length; k++) {
            W2[j][k] -= LEARNING_RATE * dW2[j][k];
        }
    }
    // b2 update
    for (let k = 0; k < b2.length; k++) {
        b2[k] -= LEARNING_RATE * db2[k];
    }
    // W1 update
    for (let i = 0; i < W1.length; i++) {
        for (let j = 0; j < W1[i].length; j++) {
            W1[i][j] -= LEARNING_RATE * dW1[i][j];
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
        for (let d = 0; d < EMBED_DIM; d++) {
            E[idx][d] -= LEARNING_RATE * grad[d];
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
                weights = savedData.weights;
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
    const payload = {
        epoch,
        vocab,
        weights
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
