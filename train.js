// train.js — VS#-15M Deep Neural Language Model Trainer
// Trains a 15,000,000 parameter neural network on VerScript & Polyglot Injection Languages (Python, JS, C, etc.)

const fs = require('fs');
const path = require('path');

let tf;
try {
    tf = require('@tensorflow/tfjs-node');
} catch (_) {
    tf = require('@tensorflow/tfjs');
}

// --- ARCHITECTURE CONSTANTS (TARGET: 15 MILLION PARAMETERS) ---
const TARGET_PARAMS = 15000000;
const CONTEXT_WINDOW = 48;
const EMBED_DIM = 256;
const HIDDEN_SIZE_2 = 512;
const LEARNING_RATE = 0.002;
const BATCH_SIZE = 64;
const DEFAULT_EPOCHS = 100;

const DATA_FILE = path.join(__dirname, 'knowledge', 'training_data.json');
const WEIGHTS_FILE = path.join(__dirname, 'model_weights.json');
const STOP_FILE = path.join(__dirname, 'stop_training.txt');

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

// --- DYNAMIC PARAMETER TARGETING ---
// Calculates H1 such that total model parameters == TARGET_PARAMS (15M)
function calculateHiddenSize(vocabSize) {
    const numerator = TARGET_PARAMS - vocabSize * (EMBED_DIM + HIDDEN_SIZE_2 + 1) - HIDDEN_SIZE_2;
    const denominator = CONTEXT_WINDOW * EMBED_DIM + 1 + HIDDEN_SIZE_2;
    return Math.max(64, Math.round(numerator / denominator));
}

// --- BUILD 15M PARAMETER NEURAL MODEL ---
function buildModel(vocabSize, hiddenSize1) {
    const model = tf.sequential();

    // 1. Embedding Layer: [batch, CONTEXT_WINDOW] -> [batch, CONTEXT_WINDOW, EMBED_DIM]
    model.add(tf.layers.embedding({
        inputDim: vocabSize,
        outputDim: EMBED_DIM,
        inputLength: CONTEXT_WINDOW
    }));

    // 2. Flatten: [batch, CONTEXT_WINDOW * EMBED_DIM]
    model.add(tf.layers.flatten());

    // 3. Dense Hidden Layer 1: [batch, H1]
    model.add(tf.layers.dense({
        units: hiddenSize1,
        activation: 'tanh',
        useBias: true
    }));

    // 4. Dense Hidden Layer 2: [batch, H2]
    model.add(tf.layers.dense({
        units: HIDDEN_SIZE_2,
        activation: 'tanh',
        useBias: true
    }));

    // 5. Output Logits: [batch, vocabSize]
    model.add(tf.layers.dense({
        units: vocabSize,
        useBias: true
    }));

    model.compile({
        optimizer: tf.train.adam(LEARNING_RATE),
        loss: 'sparseCategoricalCrossentropy',
        metrics: ['accuracy']
    });

    return model;
}

// --- GENERATE SAMPLE TEST OUTPUT ---
function testGeneration(model, vocab, vocabMap, promptText) {
    const padIdx = vocabMap.get('<pad>');
    const startIdx = vocabMap.get('<start>');
    const sepIdx = vocabMap.get('<sep>');
    const endIdx = vocabMap.get('<end>');
    const unkIdx = vocabMap.get('<unk>');

    const getIdx = t => vocabMap.has(t) ? vocabMap.get(t) : unkIdx;

    const promptTokens = tokenize(promptText);
    const seq = [startIdx];
    for (let t of promptTokens) seq.push(getIdx(t));
    seq.push(sepIdx);

    const generated = [];
    const maxLen = 40;

    for (let step = 0; step < maxLen; step++) {
        const ctx = [];
        for (let c = CONTEXT_WINDOW; c >= 1; c--) {
            const idx = seq.length - c;
            ctx.push(idx < 0 ? padIdx : seq[idx]);
        }

        const inputTensor = tf.tensor2d([ctx], [1, CONTEXT_WINDOW], 'int32');
        const logits = model.predict(inputTensor);
        const probs = tf.softmax(logits).squeeze().arraySync();
        inputTensor.dispose();
        logits.dispose();

        // Sample with temperature
        const temp = 0.8;
        const logProbs = probs.map(p => Math.log(p + 1e-10) / temp);
        const maxLog = Math.max(...logProbs);
        const exps = logProbs.map(lp => Math.exp(lp - maxLog));
        const sumExp = exps.reduce((a, b) => a + b, 0);
        const softProbs = exps.map(e => e / (sumExp || 1e-10));

        let nextIdx = endIdx;
        let cum = 0;
        const r = Math.random();
        for (let i = 0; i < softProbs.length; i++) {
            cum += softProbs[i];
            if (r <= cum) {
                nextIdx = i;
                break;
            }
        }

        if (nextIdx === endIdx) break;
        seq.push(nextIdx);
        generated.push(vocab[nextIdx]);
    }

    return generated.join(' ').replace(/ \n /g, '\n').replace(/ \n/g, '\n').replace(/\n /g, '\n');
}

// --- SAVE MODEL WEIGHTS ---
function saveWeightsCheckpoint(model, vocab, config, epoch, loss, accuracy) {
    const layers = model.layers;
    const E = layers[0].getWeights()[0].arraySync();
    const W1 = layers[2].getWeights()[0].arraySync();
    const b1 = layers[2].getWeights()[1].arraySync();
    const W2 = layers[3].getWeights()[0].arraySync();
    const b2 = layers[3].getWeights()[1].arraySync();
    const W3 = layers[4].getWeights()[0].arraySync();
    const b3 = layers[4].getWeights()[1].arraySync();

    const data = {
        vocab,
        config,
        weights: { E, W1, b1, W2, b2, W3, b3 },
        epoch,
        loss,
        accuracy,
        updatedAt: new Date().toISOString()
    };

    fs.writeFileSync(WEIGHTS_FILE, JSON.stringify(data), 'utf8');
    console.log(`[Checkpoint] Saved epoch ${epoch} weights (${config.parameterDisplay}) to model_weights.json`);
}

// --- MAIN TRAINING LOOP ---
async function main() {
    console.log("==================================================================");
    console.log("🚀 VS#-15M Neural Language Model & Polyglot Trainer (15M Params)");
    console.log("==================================================================");

    // 1. Clear model epochs / stale state
    if (fs.existsSync(WEIGHTS_FILE)) {
        console.log("[Setup] Clearing previous model weights and resetting epochs to 0...");
        fs.unlinkSync(WEIGHTS_FILE);
    }
    if (fs.existsSync(STOP_FILE)) {
        fs.unlinkSync(STOP_FILE);
    }

    // 2. Load Training Dataset
    if (!fs.existsSync(DATA_FILE)) {
        console.log("[Setup] Training dataset not found. Running build_dataset.js first...");
        require('./build_dataset');
    }

    const rawData = JSON.parse(fs.readFileSync(DATA_FILE, 'utf8'));
    console.log(`[Dataset] Loaded ${rawData.length} prompt-response pairs from ${DATA_FILE}`);

    // 3. Build Vocabulary
    const specialTokens = ['<pad>', '<start>', '<sep>', '<end>', '<unk>'];
    const tokenCounts = new Map();
    for (let pair of rawData) {
        const pToks = tokenize(pair.prompt);
        const rToks = tokenize(pair.response);
        for (let t of pToks) tokenCounts.set(t, (tokenCounts.get(t) || 0) + 1);
        for (let t of rToks) tokenCounts.set(t, (tokenCounts.get(t) || 0) + 1);
    }

    // Sort tokens by frequency
    const sortedTokens = Array.from(tokenCounts.entries())
        .sort((a, b) => b[1] - a[1])
        .map(e => e[0]);

    const vocab = [...specialTokens, ...sortedTokens];
    const vocabMap = new Map(vocab.map((t, idx) => [t, idx]));
    const vocabSize = vocab.length;

    console.log(`[Vocab] Built vocabulary of ${vocabSize} unique tokens.`);

    // 4. Compute 15M Model Dimensions
    const hiddenSize1 = calculateHiddenSize(vocabSize);
    const model = buildModel(vocabSize, hiddenSize1);
    const actualParams = model.countParams();

    const config = {
        modelName: "VS#-15M",
        parameterCount: actualParams,
        parameterDisplay: (actualParams / 1000000).toFixed(2) + " Million (" + Math.round(actualParams / 1000000) + "M)",
        architecture: "VS#-15M Deep Feed-Forward Polyglot Syntax Engine",
        vocabSize: vocabSize,
        contextWindow: CONTEXT_WINDOW,
        embedDim: EMBED_DIM,
        hiddenSize: hiddenSize1,
        hiddenSize2: HIDDEN_SIZE_2
    };

    console.log(`[Model] Target Parameters : 15,000,000`);
    console.log(`[Model] Actual Parameters : ${actualParams.toLocaleString()} (${config.parameterDisplay})`);
    console.log(`[Model] Context Window    : ${CONTEXT_WINDOW} tokens`);
    console.log(`[Model] Embedding Dim     : ${EMBED_DIM}`);
    console.log(`[Model] Hidden Layer 1    : ${hiddenSize1}`);
    console.log(`[Model] Hidden Layer 2    : ${HIDDEN_SIZE_2}`);

    // 5. Build Training Windows
    const padIdx = vocabMap.get('<pad>');
    const startIdx = vocabMap.get('<start>');
    const sepIdx = vocabMap.get('<sep>');
    const endIdx = vocabMap.get('<end>');
    const unkIdx = vocabMap.get('<unk>');

    const getIdx = t => vocabMap.has(t) ? vocabMap.get(t) : unkIdx;

    const allInputs = [];
    const allTargets = [];

    for (let pair of rawData) {
        const pToks = tokenize(pair.prompt).map(getIdx);
        const rToks = tokenize(pair.response).map(getIdx);

        const seq = [startIdx, ...pToks, sepIdx, ...rToks, endIdx];

        // Generate sliding windows
        const promptLen = pToks.length + 2; // start + prompt + sep
        for (let i = promptLen; i < seq.length; i++) {
            const ctx = [];
            for (let c = CONTEXT_WINDOW; c >= 1; c--) {
                const pos = i - c;
                ctx.push(pos < 0 ? padIdx : seq[pos]);
            }
            allInputs.push(ctx);
            allTargets.push(seq[i]);
        }
    }

    console.log(`[Dataset] Generated ${allInputs.length.toLocaleString()} training window samples.`);

    // 6. Training Loop
    const totalSamples = allInputs.length;
    const epochs = parseInt(process.env.EPOCHS || process.argv[2] || DEFAULT_EPOCHS, 10);
    console.log(`[Training] Commencing training for ${epochs} epochs (Batch Size: ${BATCH_SIZE})...`);

    // Initial checkpoint save (epoch 0, randomized initialized weights)
    saveWeightsCheckpoint(model, vocab, config, 0, 0, 0);

    for (let epoch = 1; epoch <= epochs; epoch++) {
        if (fs.existsSync(STOP_FILE)) {
            console.log(`[Stop] Detected ${STOP_FILE}. Halting training cleanly.`);
            fs.unlinkSync(STOP_FILE);
            break;
        }

        const t0 = Date.now();

        // Sample a balanced batch or iterate through windows in minibatches
        // We shuffle indices for stochastic mini-batch training
        const indices = Array.from({ length: totalSamples }, (_, i) => i);
        for (let i = indices.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [indices[i], indices[j]] = [indices[j], indices[i]];
        }

        // Train on subset per epoch for fast iterative feedback or full epoch
        const maxSamplesPerEpoch = Math.min(totalSamples, 2048);
        const batchInputs = [];
        const batchTargets = [];

        for (let k = 0; k < maxSamplesPerEpoch; k++) {
            const idx = indices[k];
            batchInputs.push(allInputs[idx]);
            batchTargets.push(allTargets[idx]);
        }

        const xTensor = tf.tensor2d(batchInputs, [maxSamplesPerEpoch, CONTEXT_WINDOW], 'int32');
        const yTensor = tf.tensor1d(batchTargets, 'float32');

        const history = await model.fit(xTensor, yTensor, {
            batchSize: BATCH_SIZE,
            epochs: 1,
            shuffle: true,
            verbose: 0
        });

        const loss = history.history.loss[0];
        const acc = history.history.acc ? history.history.acc[0] : 0;
        const elapsed = ((Date.now() - t0) / 1000).toFixed(2);

        xTensor.dispose();
        yTensor.dispose();

        console.log(`Epoch ${epoch}/${epochs} - Loss: ${loss.toFixed(4)} | Acc: ${(acc * 100).toFixed(1)}% | Time: ${elapsed}s`);

        // Save weights checkpoint every 2 epochs or last
        if (epoch % 2 === 0 || epoch === epochs) {
            saveWeightsCheckpoint(model, vocab, config, epoch, loss, acc);
        }

        // Test generation every 5 epochs
        if (epoch % 5 === 0) {
            const samplePrompt = "hello";
            const sampleGen = testGeneration(model, vocab, vocabMap, samplePrompt);
            console.log(`  Sample Gen ["${samplePrompt}"]: ${sampleGen.slice(0, 80)}...`);
        }
    }

    console.log("==================================================================");
    console.log("✅ Training Complete! Model weights saved to model_weights.json");
    console.log("==================================================================");
}

if (require.main === module) {
    main().catch(err => {
        console.error("Training Error:", err);
        process.exit(1);
    });
}

module.exports = { main, buildModel, calculateHiddenSize, tokenize, TARGET_PARAMS };
