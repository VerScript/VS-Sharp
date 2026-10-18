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
const LEARNING_RATE = 0.003;
const BATCH_SIZE = parseInt(process.env.BATCH_SIZE || '32', 10);
const SAMPLES_PER_EPOCH = parseInt(process.env.SAMPLES_PER_EPOCH || '384', 10);
const DEFAULT_EPOCHS = 50;

const DATA_FILE = path.join(__dirname, 'knowledge', 'training_data.json');
const WEIGHTS_FILE = path.join(__dirname, 'model_weights.json');
const STOP_FILE = path.join(__dirname, 'stop_training.txt');
const EVAL_FILE = path.join(__dirname, 'epoch_eval_history.json');

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
function testGeneration(model, vocab, vocabMap, promptText, maxTokens = 50) {
    const padIdx = vocabMap.get('<pad>') ?? 0;
    const startIdx = vocabMap.get('<start>') ?? 1;
    const sepIdx = vocabMap.get('<sep>') ?? 2;
    const endIdx = vocabMap.get('<end>') ?? 3;
    const unkIdx = vocabMap.get('<unk>') ?? 4;

    const getIdx = t => vocabMap.has(t) ? vocabMap.get(t) : unkIdx;

    const promptTokens = tokenize(promptText);
    const seq = [startIdx];
    for (let t of promptTokens) seq.push(getIdx(t));
    seq.push(sepIdx);

    const generated = [];

    for (let step = 0; step < maxTokens; step++) {
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
        const temp = 0.75;
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

    const isReset = process.argv.includes('--reset');

    // 1. Dataset loading
    if (!fs.existsSync(DATA_FILE)) {
        console.log("[Setup] Training dataset not found. Running build_dataset.js first...");
        require('./build_dataset');
    }

    const rawData = JSON.parse(fs.readFileSync(DATA_FILE, 'utf8'));
    console.log(`[Dataset] Loaded ${rawData.length} prompt-response pairs from ${DATA_FILE}`);

    // 2. Vocabulary
    const specialTokens = ['<pad>', '<start>', '<sep>', '<end>', '<unk>'];
    const tokenCounts = new Map();
    for (let pair of rawData) {
        const pToks = tokenize(pair.prompt);
        const rToks = tokenize(pair.response);
        for (let t of pToks) tokenCounts.set(t, (tokenCounts.get(t) || 0) + 1);
        for (let t of rToks) tokenCounts.set(t, (tokenCounts.get(t) || 0) + 1);
    }

    const sortedTokens = Array.from(tokenCounts.entries())
        .sort((a, b) => b[1] - a[1])
        .map(e => e[0]);

    const vocab = [...specialTokens, ...sortedTokens];
    const vocabMap = new Map(vocab.map((t, idx) => [t, idx]));
    const vocabSize = vocab.length;

    console.log(`[Vocab] Built vocabulary of ${vocabSize} unique tokens.`);

    // 3. Model construction
    const hiddenSize1 = calculateHiddenSize(vocabSize);
    const model = buildModel(vocabSize, hiddenSize1);
    const actualParams = model.countParams();

    const config = {
        modelName: "VS#-15M",
        parameterCount: actualParams,
        parameterDisplay: (actualParams / 1000000).toFixed(2) + " Million (" + Math.round(actualParams / 1000000) + "M)",
        architecture: "VS#-15M Deep Neural Language Model & Polyglot Engine",
        vocabSize: vocabSize,
        contextWindow: CONTEXT_WINDOW,
        embedDim: EMBED_DIM,
        hiddenSize: hiddenSize1,
        hiddenSize2: HIDDEN_SIZE_2
    };

    console.log(`[Model] Target Parameters : 15,000,000`);
    console.log(`[Model] Actual Parameters : ${actualParams.toLocaleString()} (${config.parameterDisplay})`);
    console.log(`[Model] Context Window    : ${CONTEXT_WINDOW} tokens`);
    console.log(`[Model] Hidden Layer 1    : ${hiddenSize1} units`);
    console.log(`[Model] Hidden Layer 2    : ${HIDDEN_SIZE_2} units`);
    console.log(`[Model] Samples / Epoch   : ${SAMPLES_PER_EPOCH} (Batch Size: ${BATCH_SIZE})`);

    // Load existing weights if present and not reset
    let startingEpoch = 1;
    if (!isReset && fs.existsSync(WEIGHTS_FILE)) {
        try {
            console.log("[Setup] Found existing model_weights.json. Loading trained weights...");
            const prevData = JSON.parse(fs.readFileSync(WEIGHTS_FILE, 'utf8'));
            if (prevData.epoch) startingEpoch = prevData.epoch + 1;
            const w = prevData.weights;
            if (w && w.E && w.W1 && w.b1 && w.W2 && w.b2 && w.W3 && w.b3) {
                model.layers[0].setWeights([tf.tensor2d(w.E, [vocabSize, EMBED_DIM])]);
                model.layers[2].setWeights([
                    tf.tensor2d(w.W1, [CONTEXT_WINDOW * EMBED_DIM, hiddenSize1]),
                    tf.tensor1d(w.b1)
                ]);
                model.layers[3].setWeights([
                    tf.tensor2d(w.W2, [hiddenSize1, HIDDEN_SIZE_2]),
                    tf.tensor1d(w.b2)
                ]);
                model.layers[4].setWeights([
                    tf.tensor2d(w.W3, [HIDDEN_SIZE_2, vocabSize]),
                    tf.tensor1d(w.b3)
                ]);
                console.log(`[Setup] Successfully resumed from epoch ${prevData.epoch} (loss: ${prevData.loss || 'N/A'})!`);
            }
        } catch (e) {
            console.log("[Setup] Note: could not resume weights directly (" + e.message + "). Initializing fresh.");
        }
    } else {
        console.log("[Setup] Initializing fresh weights at epoch 0...");
        saveWeightsCheckpoint(model, vocab, config, 0, 0, 0);
    }

    // 4. Build Training Windows
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
        const promptLen = pToks.length + 2;

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

    console.log(`[Dataset] Total available training sequences: ${allInputs.length.toLocaleString()}`);

    // Evaluation prompts to test after every epoch
    const evalSuites = [
        { category: "Conversation", prompt: "hello" },
        { category: "VerScript", prompt: "write a hello world program in verscript" },
        { category: "Inject Python", prompt: "how to run python in verscript" },
        { category: "Inject JS", prompt: "write javascript async fetch function" },
        { category: "Inject C", prompt: "write c code for binary search" }
    ];

    // Load or initialize evaluation history
    let evalHistory = [];
    if (fs.existsSync(EVAL_FILE)) {
        try { evalHistory = JSON.parse(fs.readFileSync(EVAL_FILE, 'utf8')); } catch (_) {}
    }

    // 5. Training Loop
    const totalSamples = allInputs.length;
    const targetEpochs = parseInt(process.env.EPOCHS || process.argv[2] || DEFAULT_EPOCHS, 10);
    const endEpoch = startingEpoch + targetEpochs - 1;

    console.log(`\n[Training] Starting training epochs ${startingEpoch} -> ${endEpoch}...`);

    for (let epoch = startingEpoch; epoch <= endEpoch; epoch++) {
        if (fs.existsSync(STOP_FILE)) {
            console.log(`[Stop] Detected ${STOP_FILE}. Halting training cleanly.`);
            fs.unlinkSync(STOP_FILE);
            break;
        }

        console.log(`\n------------------------------------------------------------------`);
        console.log(`>>> Starting Epoch ${epoch} / ${endEpoch}`);
        console.log(`------------------------------------------------------------------`);

        const t0 = Date.now();

        // Randomly sample SAMPLES_PER_EPOCH sequences
        const sampleCount = Math.min(totalSamples, SAMPLES_PER_EPOCH);
        const indices = [];
        for (let s = 0; s < sampleCount; s++) {
            indices.push(Math.floor(Math.random() * totalSamples));
        }

        const batchInputs = indices.map(i => allInputs[i]);
        const batchTargets = indices.map(i => allTargets[i]);

        const totalBatches = Math.ceil(sampleCount / BATCH_SIZE);

        const xTensor = tf.tensor2d(batchInputs, [sampleCount, CONTEXT_WINDOW], 'int32');
        const yTensor = tf.tensor1d(batchTargets, 'float32');

        const history = await model.fit(xTensor, yTensor, {
            batchSize: BATCH_SIZE,
            epochs: 1,
            shuffle: true,
            verbose: 0,
            callbacks: {
                onBatchEnd: (batch, logs) => {
                    const bNum = batch + 1;
                    const lossStr = logs.loss ? logs.loss.toFixed(4) : "N/A";
                    const accStr = logs.acc !== undefined ? (logs.acc * 100).toFixed(1) + "%" : "N/A";
                    console.log(`  [Epoch ${epoch}] Batch ${bNum}/${totalBatches} | Loss: ${lossStr} | Acc: ${accStr}`);
                }
            }
        });

        const loss = history.history.loss[0];
        const acc = history.history.acc ? history.history.acc[0] : 0;
        const elapsed = ((Date.now() - t0) / 1000).toFixed(1);

        xTensor.dispose();
        yTensor.dispose();

        console.log(`\n[Epoch ${epoch} Finished] Loss: ${loss.toFixed(4)} | Accuracy: ${(acc * 100).toFixed(1)}% | Duration: ${elapsed}s`);

        // Save Checkpoint after EVERY epoch
        saveWeightsCheckpoint(model, vocab, config, epoch, loss, acc);

        // Run LLM Tests after EVERY epoch
        console.log(`\n--- [Epoch ${epoch}] LLM Generation Test Results ---`);
        const epochTestResults = {
            epoch,
            loss,
            accuracy: acc,
            timestamp: new Date().toISOString(),
            tests: []
        };

        for (const testCase of evalSuites) {
            const output = testGeneration(model, vocab, vocabMap, testCase.prompt, 45);
            epochTestResults.tests.push({
                category: testCase.category,
                prompt: testCase.prompt,
                response: output
            });
            console.log(`\n  [${testCase.category}] Prompt: "${testCase.prompt}"`);
            console.log(`  Generated: ${output.slice(0, 160)}${output.length > 160 ? '...' : ''}`);
        }

        evalHistory.push(epochTestResults);
        fs.writeFileSync(EVAL_FILE, JSON.stringify(evalHistory, null, 2), 'utf8');
        console.log(`\n[Eval] Saved LLM generation review to ${EVAL_FILE}`);
    }

    console.log("\n==================================================================");
    console.log("✅ Training Loop Finished! Latest weights in model_weights.json");
    console.log("==================================================================");
}

if (require.main === module) {
    main().catch(err => {
        console.error("Training Error:", err);
        process.exit(1);
    });
}

module.exports = { main, buildModel, calculateHiddenSize, tokenize, TARGET_PARAMS };
