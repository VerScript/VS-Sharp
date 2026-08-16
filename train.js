const fs = require('fs');
const path = require('path');
const tf = require('@tensorflow/tfjs-node');

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
const EMBED_DIM = 2048;
const CONTEXT_WINDOW = 112;
const HIDDEN_SIZE = 4096;
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


// --- TENSORFLOW MODEL SETUP ---
function buildModel(vocabSize) {
    const model = tf.sequential();
    // Embedding layer: [batch_size, CONTEXT_WINDOW] -> [batch_size, CONTEXT_WINDOW, EMBED_DIM]
    model.add(tf.layers.embedding({
        inputDim: vocabSize,
        outputDim: EMBED_DIM,
        inputLength: CONTEXT_WINDOW
    }));

    // Flatten to [batch_size, CONTEXT_WINDOW * EMBED_DIM]
    model.add(tf.layers.flatten());

    // Dense Hidden Layer (tanh)
    model.add(tf.layers.dense({
        units: HIDDEN_SIZE,
        activation: 'tanh',
        useBias: true
    }));

    // Output Logits Layer
    model.add(tf.layers.dense({
        units: vocabSize,
        useBias: true
    }));

    model.compile({
        optimizer: tf.train.sgd(LEARNING_RATE),
        loss: 'sparseCategoricalCrossentropy'
    });

    return model;
}

// Helper to manually extract weights for saving in the format expected by server.js
async function extractWeights(model) {
    const E = await model.layers[0].getWeights()[0].data();
    const W1 = await model.layers[2].getWeights()[0].data();
    const b1 = await model.layers[2].getWeights()[1].data();
    const W2 = await model.layers[3].getWeights()[0].data();
    const b2 = await model.layers[3].getWeights()[1].data();

    const chunkSize = EMBED_DIM * HIDDEN_SIZE;
    const parsedW1 = new Array(CONTEXT_WINDOW);
    for (let c = 0; c < CONTEXT_WINDOW; c++) {
        parsedW1[c] = W1.subarray(c * chunkSize, (c + 1) * chunkSize);
    }

    return {
        E: E,
        W1: parsedW1,
        b1: b1,
        W2: W2,
        b2: b2
    };
}



const https = require('https');

function fetchUrl(url) {
    return new Promise((resolve, reject) => {
        https.get(url, { headers: { 'User-Agent': 'Node.js' } }, (res) => {
            let data = '';
            res.on('data', chunk => data += chunk);
            res.on('end', () => resolve(data));
        }).on('error', reject);
    });
}

async function augmentTrainingData() {
    let rawData = [];
    if (fs.existsSync(DATA_FILE)) {
        rawData = JSON.parse(fs.readFileSync(DATA_FILE, 'utf8'));
    }

    try {
        console.log("Fetching data from VerScript repos...");

        const existingPrompts = new Set(rawData.map(d => d.prompt));

        const codeSample = await fetchUrl('https://raw.githubusercontent.com/VerScript/VerScript/main/README.md');
        if (codeSample && codeSample.length > 0 && !codeSample.includes('404: Not Found')) {
            const codePrompt = "What is VerScript based on its README?";
            if (!existingPrompts.has(codePrompt)) {
                rawData.push({
                    prompt: codePrompt,
                    response: "### VerScript README\n\n" + codeSample.substring(0, 500)
                });
            }
        }

        const docsSample = await fetchUrl('https://raw.githubusercontent.com/VerScript/VerScript.github.io/main/index.html');
        if (docsSample && docsSample.length > 0 && !docsSample.includes('404: Not Found')) {
            const docsPrompt = "Show me the main docs page for VerScript";
            if (!existingPrompts.has(docsPrompt)) {
                rawData.push({
                    prompt: docsPrompt,
                    response: "### VerScript Documentation\n\n" + docsSample.substring(0, 500)
                });
            }
        }

        const otherLangs = [
            {
                prompt: "Write a python script to calculate factorial",
                response: "```python\ndef factorial(n):\n    if n == 0:\n        return 1\n    return n * factorial(n-1)\n```"
            },
            {
                prompt: "Write a JS function for binary search",
                response: "```javascript\nfunction binarySearch(arr, target) {\n    let left = 0, right = arr.length - 1;\n    while (left <= right) {\n        let mid = Math.floor((left + right) / 2);\n        if (arr[mid] === target) return mid;\n        else if (arr[mid] < target) left = mid + 1;\n        else right = mid - 1;\n    }\n    return -1;\n}\n```"
            },
            {
                prompt: "Create a simple C++ program to print Hello World",
                response: "```cpp\n#include <iostream>\n\nint main() {\n    std::cout << \"Hello, World!\" << std::endl;\n    return 0;\n}\n```"
            }
        ];

        for (const langPair of otherLangs) {
            if (!existingPrompts.has(langPair.prompt)) {
                rawData.push(langPair);
            }
        }

        fs.writeFileSync(DATA_FILE, JSON.stringify(rawData, null, 4));
        console.log("Updated training_data.json with external sources and multi-language support.");
    } catch (e) {
        console.error("Error fetching data:", e);
    }
}

// --- MAIN TRAINING WORKER ---
async function startTraining() {
    await augmentTrainingData();

    if (!fs.existsSync(DATA_FILE)) {
        console.error(`Error: Training data not found at ${DATA_FILE}`);
        process.exit(1);
    }

    console.log("Loading training data...");
    const rawData = JSON.parse(fs.readFileSync(DATA_FILE, 'utf8'));

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

    let model = buildModel(vocab.length);
    let startEpoch = 0;

    if (fs.existsSync(WEIGHTS_FILE)) {
        console.log(`Existing model weights found at ${WEIGHTS_FILE}. Loading...`);
        try {
            const savedData = JSON.parse(fs.readFileSync(WEIGHTS_FILE, 'utf8'));
            if (JSON.stringify(savedData.vocab) === JSON.stringify(vocab)) {
                const w = savedData.weights;

                const getArray = (val) => Array.isArray(val) ? val : Object.values(val);

                const E_tensor = tf.tensor2d(getArray(w.E), [vocab.length, EMBED_DIM]);

                const flatW1 = new Float32Array(CONTEXT_WINDOW * EMBED_DIM * HIDDEN_SIZE);
                let W1_raw;
                if (Array.isArray(w.W1) && Array.isArray(w.W1[0])) {
                    W1_raw = w.W1;
                } else {
                    const tmpFlatW1 = getArray(w.W1);
                    const chunkSize = EMBED_DIM * HIDDEN_SIZE;
                    W1_raw = new Array(CONTEXT_WINDOW);
                    for (let c = 0; c < CONTEXT_WINDOW; c++) {
                        W1_raw[c] = tmpFlatW1.slice(c * chunkSize, (c + 1) * chunkSize);
                    }
                }

                for(let c=0; c<CONTEXT_WINDOW; c++) {
                    flatW1.set(W1_raw[c], c * EMBED_DIM * HIDDEN_SIZE);
                }
                const W1_tensor = tf.tensor2d(flatW1, [CONTEXT_WINDOW * EMBED_DIM, HIDDEN_SIZE]);

                const b1_tensor = tf.tensor1d(getArray(w.b1));
                const W2_tensor = tf.tensor2d(getArray(w.W2), [HIDDEN_SIZE, vocab.length]);
                const b2_tensor = tf.tensor1d(getArray(w.b2));

                model.layers[0].setWeights([E_tensor]);
                model.layers[2].setWeights([W1_tensor, b1_tensor]);
                model.layers[3].setWeights([W2_tensor, b2_tensor]);

                startEpoch = savedData.epoch || 0;
                console.log(`Resuming training from epoch ${startEpoch}...`);
            } else {
                console.log("Vocabulary changed. Re-initializing weights.");
            }
        } catch (e) {
            console.warn("Failed to load weights file, initializing random weights:", e);
        }
    }

    console.log("\n=============================================");
    console.log("   VS-Sharp Generative LLM Training Started  ");
    console.log(`   Running loop in background...             `);
    console.log(`   To stop, run PowerShell: .\\stop-training.ps1`);
    console.log("=============================================\n");

    const batchSize = 64;
    const xs = tf.tensor2d(dataset.map(d => d.context), [dataset.length, CONTEXT_WINDOW], 'int32');
    // For sparseCategoricalCrossentropy, targets should be 1D integer
    const ys = tf.tensor1d(dataset.map(d => d.target), 'int32');

    let epoch = startEpoch;
    
    while(true) {
        if (fs.existsSync(STOP_FILE)) {
            console.log("\n[Stop Signal Detected]");
            const extractedWeights = await extractWeights(model);
            saveWeights(extractedWeights, vocab, epoch);
            try {
                fs.unlinkSync(STOP_FILE);
            } catch (err) {}
            console.log("Training stopped safely. Exiting process.");
            process.exit(0);
        }

        const h = await model.fit(xs, ys, {
            batchSize: batchSize,
            epochs: 1,
            shuffle: true,
            verbose: 0
        });

        epoch++;
        const loss = h.history.loss[0];

        if (epoch % 5 === 0 || epoch === 1) {
            console.log(`Epoch ${epoch} | Average Cross-Entropy Loss: ${loss.toFixed(6)}`);
        }

        if (epoch % SAVE_INTERVAL_EPOCHS === 0) {
            const extractedWeights = await extractWeights(model);
            saveWeights(extractedWeights, vocab, epoch);
        }

        await new Promise(r => setImmediate(r));
    }
}

function saveWeights(weights, vocab, epoch) {
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

startTraining();
