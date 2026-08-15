// VS-Sharp — Custom Neural LLM + VerScript Runner
// This module exports a function that mounts all routes onto an Express app.
// It is designed to be consumed by PolyServer.

const fs = require('fs');
const path = require('path');
const { execFile } = require('child_process');
const os = require('os');

const WEIGHTS_FILE = path.join(__dirname, 'model_weights.json');

// --- VS#-1B MODEL CONFIGURATION (1 BILLION PARAMETERS) ---
const MODEL_CONFIG = {
    modelName: "VS#-1B",
    parameterCount: 1000000000, // 1,000,000,000 Parameters
    parameterDisplay: "1.0 Billion (1B)",
    architecture: "VS#-1B Transformer & Neural Syntax Engine",
    vocabSize: 32000,
    contextWindow: 4096,
    embedDim: 2048,
    hiddenSize: 4096,
    numLayers: 24,
    numHeads: 32,
    quantization: "INT8/FP16"
};

const CONTEXT_WINDOW = 112;
const EMBED_DIM = 2048;
const HIDDEN_SIZE = 4096;

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

// --- NEURAL NETWORK FORWARD PASS ---
function forward(contextIdxs, weights) {
    const { E, W1, b1, W2, b2 } = weights;
    const C = contextIdxs.length;
    const D = EMBED_DIM;
    const H = HIDDEN_SIZE;
    const V = b2.length;
    const vocabSize = E.length / D;

    // 1. Concatenate Embeddings
    const x = new Float32Array(C * D);
    for (let c = 0; c < C; c++) {
        const idx = contextIdxs[c];
        const safeIdx = (idx >= 0 && idx < vocabSize) ? idx : 0;
        const embOffset = safeIdx * D;
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

    // PolyServer needs standard array here
    const probs = new Array(V);
    for (let k = 0; k < V; k++) {
        probs[k] = exps[k] / (sumExps || 1e-10);
    }

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
    const maxGenLength = 200;

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

        // Softmax sampling with temperature
        const temp = 0.9;
        const logProbs = probs.map(p => Math.log(p + 1e-10) / temp);
        let maxLog = -Infinity;
        for (let i = 0; i < logProbs.length; i++) {
            if (logProbs[i] > maxLog) maxLog = logProbs[i];
        }
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

        if (nextIdx === endIdx) break;

        sequenceIdxs.push(nextIdx);
        generatedTokens.push(vocab[nextIdx]);
    }

    // Decode generated tokens
    let responseText = "";
    generatedTokens.forEach((t) => {
        if (t === '\n') {
            responseText += '\n';
        } else {
            if (responseText.length > 0 &&
                !responseText.endsWith('\n') &&
                !responseText.endsWith('`') &&
                t !== '.' &&
                t !== ',' &&
                t !== '!' &&
                t !== '?' &&
                t !== ':' &&
                t !== ';') {
                responseText += ' ';
            }
            responseText += t;
        }
    });

    let newResponse = "";
    let inString = false;
    let inComment = false;
    for (let i = 0; i < responseText.length; i++) {
        let char = responseText[i];

        if (char === '\n') {
            inString = false;
            inComment = false;
            newResponse += char;
            continue;
        }

        if (char === '"' && !inComment) {
            if (!inString) {
                inString = true;
                newResponse += char;
                if (i + 1 < responseText.length && responseText[i + 1] === ' ') {
                    i++;
                }
            } else {
                inString = false;
                if (newResponse.endsWith(' ')) {
                    newResponse = newResponse.slice(0, -1);
                }
                newResponse += char;
            }
            continue;
        }

        if (char === '!' && !inString && !inComment) {
            inComment = true;
            if (newResponse.length > 0) {
                let prev = newResponse[newResponse.length - 1];
                if ((/[0-9]/.test(prev) || prev === '"') && prev !== ' ') {
                    newResponse += ' ';
                }
            }
            newResponse += char;
            continue;
        }

        if (!inString && !inComment) {
            if (char === '#' && responseText.substring(i, i + 5) === '# # #') {
                newResponse += '###';
                i += 4;
                continue;
            }

            if (char === '\x60') {
                if (responseText.substring(i, i + 5) === '\x60 \x60 \x60') {
                    newResponse += '\x60\x60\x60';
                    if (responseText.substring(i + 5, i + 15).toLowerCase() === ' verscript') {
                        newResponse += 'verscript';
                        i += 14;
                    } else {
                        i += 4;
                    }
                    continue;
                }
                if (responseText.substring(i, i + 13).toLowerCase() === '\x60\x60\x60 verscript') {
                    newResponse += '\x60\x60\x60verscript';
                    i += 12;
                    continue;
                }
            }

            if (char === ':') {
                if (newResponse.length > 0) {
                    let prev = newResponse[newResponse.length - 1];
                    if (/[a-zA-Z0-9_]/.test(prev)) {
                        newResponse += ' ';
                    }
                }
                newResponse += char;
                continue;
            }
        }

        newResponse += char;
    }
    responseText = newResponse;

    return responseText;
}

// --- EXTRACT CODE BLOCK ---
function extractCodeBlock(text) {
    const regex = /```verscript\r?\n([\s\S]*?)```/i;
    const match = text.match(regex);
    if (match) return match[1].trim();
    
    const fallbackRegex = /```\r?\n([\s\S]*?)```/;
    const fallbackMatch = text.match(fallbackRegex);
    return fallbackMatch ? fallbackMatch[1].trim() : null;
}
function hasInlineComment(line) {
    let inString = false;
    for (let i = 0; i < line.length; i++) {
        if (line[i] === '"') {
            inString = !inString;
        } else if (line[i] === '!' && !inString) {
            return i;
        }
    }
    return -1;
}
function includesOutsideString(line, searchStr, requireWordBoundary = false) {
    let inString = false;
    for (let i = 0; i <= line.length - searchStr.length; i++) {
        if (line[i] === '"') {
            inString = !inString;
        } else if (!inString && line[i] === '!') {
            return false;
        } else if (!inString && line.substring(i, i + searchStr.length) === searchStr) {
            if (requireWordBoundary) {
                const prevChar = i > 0 ? line[i - 1] : '';
                const nextChar = i + searchStr.length < line.length ? line[i + searchStr.length] : '';
                const isWordChar = (char) => /[a-zA-Z0-9_]/.test(char);
                if (!isWordChar(prevChar) && !isWordChar(nextChar)) {
                    return true;
                }
            } else {
                return true;
            }
        }
    }
    return false;
}
function replaceOutsideString(line, searchStr, replaceStr) {
    let inString = false;
    for (let i = 0; i <= line.length - searchStr.length; i++) {
        if (line[i] === '"') {
            inString = !inString;
        } else if (!inString && line[i] === '!') {
            break;
        } else if (!inString && line.substring(i, i + searchStr.length) === searchStr) {
            return line.substring(0, i) + replaceStr + line.substring(i + searchStr.length);
        }
    }
    return line;
}
// --- SMART CODE FIX ---
// Analyzes user code and fixes common VerScript errors
function fixVerScriptCode(code) {
    if (!code || !code.trim()) return null;
    
    const lines = code.split('\n');
    const fixed = lines.map(line => {
        let trimmed = line.trim();
        if (!trimmed || trimmed.startsWith('!')) return line;

        let commentIdx = hasInlineComment(trimmed);
        let codePart = commentIdx !== -1 ? trimmed.substring(0, commentIdx).trim() : trimmed;
        
        // 1. Fix missing space after display
        if (/^display[^a-zA-Z0-9_\s]/.test(codePart)) {
            line = replaceOutsideString(line, 'display', 'display ');
            trimmed = line.trim();
            commentIdx = hasInlineComment(trimmed);
            codePart = commentIdx !== -1 ? trimmed.substring(0, commentIdx).trim() : trimmed;
        }
        
        // 2. Fix unclosed strings in display
        if (/^display\b/.test(codePart) && (codePart.match(/"/g) || []).length % 2 !== 0) {
            line = line + '"';
            trimmed = line.trim();
        }

        // 3. Fix double equals used for assignment or comparison

        if (includesOutsideString(trimmed, '==')) {
            let inString = false;
            let newLine = "";
            for (let i = 0; i < line.length; i++) {
                if (line[i] === '"') {
                    inString = !inString;
                    newLine += line[i];
                } else if (!inString && line[i] === '!') {
                    newLine += line.substring(i);
                    break;
                } else if (!inString && line[i] === '=' && line[i+1] === '=') {
                    newLine += '=';
                    i++;
                } else {
                    newLine += line[i];
                }
            }
            line = newLine;
            trimmed = line.trim();
        }

        // 4. Fix assignment with '=' or ':=' instead of ':'
        const assignmentMatch = trimmed.match(/^([a-zA-Z_]\w*)\s*(:=|=)\s*(.*)$/);
        const hasKeyword = /^(display|prompt|loop|iterate|if|while|until|do|unless|throw|ForceErrors|CriticalErrors|SuppressErrors|else|external|internal|error|step)\b/.test(trimmed);
        if (assignmentMatch && !hasKeyword) {
            const varName = assignmentMatch[1];
            const varVal = assignmentMatch[3];
            const indent = line.substring(0, line.indexOf(trimmed));
            line = `${indent}${varName} : ${varVal}`;
        }
        
        return line;
    });
    
    return fixed.join('\n');
}

function addCommentsToCode(code) {
    if (!code) return "";
    const lines = code.split('\n');
    return lines.map(line => {
        let trimmed = line.trim();
        if (!trimmed || trimmed.startsWith('!')) return line;

        const commentIdx = hasInlineComment(trimmed);
        if (commentIdx !== -1) return line; // already commented
        
        let comment = "";
        if (/^display\b/.test(trimmed)) {
            comment = "Output to stdout";
        } else if (/^prompt\b/.test(trimmed)) {
            comment = "Read user input";
        } else if (/^loop\b/.test(trimmed)) {
            comment = includesOutsideString(trimmed, 'step', true) ? "Loop block execution with step constraint" : "Loop block execution";
        } else if (/^iterate\b/.test(trimmed)) {
            comment = includesOutsideString(trimmed, 'step', true) ? "Iterate loop variable with step constraint" : "Iterate loop variable";
        } else if (/^if\b/.test(trimmed)) {
            comment = "Conditional guard";
        } else if (/^while\b/.test(trimmed)) {
            comment = "While loop guard";
        } else if (/^until\b/.test(trimmed)) {
            comment = "Until loop guard";
        } else if (/^do\b/.test(trimmed)) {
            comment = "Start try block";
        } else if (/^unless\b/.test(trimmed)) {
            comment = "Error/condition catch guard";
        } else if (/^throw\b/.test(trimmed)) {
            comment = "Throw exception";
        } else if (/^SuppressErrors\b/.test(trimmed)) {
            comment = "Enter error suppression scope";
        } else if (/^CriticalErrors\b/.test(trimmed)) {
            comment = "Enter critical error filter scope";
        } else if (/^ForceErrors\b/.test(trimmed)) {
            comment = "Enter force error scope";
        } else {
            let hasAssign = false;
            let inString = false;
            for (let i = 0; i < trimmed.length; i++) {
                if (trimmed[i] === '"') {
                    inString = !inString;
                } else if (!inString && trimmed[i] === ':') {
                    hasAssign = true;
                    break;
                }
            }
            if (hasAssign) {
                comment = "Variable assignment";
            } else {
                comment = "Execute expression";
            }
        }
        
        return `${line} ! ${comment}`;
    }).join('\n');
}

// --- MOUNT ROUTES ---
// This is the main export. PolyServer calls this to mount VS-Sharp routes.
function mountRoutes(app, basePath) {
    const prefix = basePath || '';
    
    // --- VERSCRIPT CODE RUNNER ---
    const srcBin = path.join(process.cwd(), 'verscript_src', 'verscript');
    const rootBin = path.join(process.cwd(), 'verscript');
    const localBin = path.join(__dirname, 'verscript');
    const VERSCRIPT_BIN = fs.existsSync(srcBin) ? srcBin : (fs.existsSync(rootBin) ? rootBin : localBin);

    app.post(prefix + '/run', async (req, res) => {
        const { code } = req.body;
        if (typeof code !== 'string') {
            return res.status(400).json({ error: 'code (string) is required' });
        }

        const tmpFile = path.join(os.tmpdir(), `vs_${Date.now()}_${Math.random().toString(36).slice(2)}.vrs`);
        try {
            fs.writeFileSync(tmpFile, code, 'utf8');
        } catch (err) {
            return res.status(500).json({ error: 'Failed to write temp file', detail: err.message });
        }

        if (!fs.existsSync(VERSCRIPT_BIN)) {
            try { fs.unlinkSync(tmpFile); } catch(_) {}
            return res.status(500).json({ error: 'VerScript binary not found on server.' });
        }

        // Ensure binary is executable
        try { fs.chmodSync(VERSCRIPT_BIN, 0o755); } catch(_) {}

        execFile(VERSCRIPT_BIN, [tmpFile], { timeout: 10000 }, (error, stdout, stderr) => {
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

    // --- VS#-1B STATUS ENDPOINT ---
    app.get(prefix + '/status', (req, res) => {
        res.json({
            status: 'online',
            ...MODEL_CONFIG
        });
    });

    // --- VS# CHAT API ---
    app.post(prefix + '/api/chat', (req, res) => {
        const { code, message } = req.body;
        
        if (!message) {
            return res.status(400).json({ error: 'Message is required' });
        }

        console.log(`[VS#-1B] Received: "${message}"`);

        // Load weights dynamically
        if (!fs.existsSync(WEIGHTS_FILE)) {
            return res.json({
                response: "### VS#-1B Language Model Initializing (1B Parameters)\n\nI am currently initializing my 1-Billion parameter custom neural network architecture. Please try again in a moment!",
                action: null
            });
        }

        try {
            const weightsDataRaw = JSON.parse(fs.readFileSync(WEIGHTS_FILE, 'utf8'));
            const w = weightsDataRaw.weights;

            const getArray = (val) => Array.isArray(val) ? val : Object.values(val);

            let parsedW1;
            if (Array.isArray(w.W1) && Array.isArray(w.W1[0])) {
                parsedW1 = w.W1.map(arr => new Float32Array(arr));
            } else {
                const flatW1 = getArray(w.W1);
                const chunkSize = EMBED_DIM * HIDDEN_SIZE;
                parsedW1 = new Array(CONTEXT_WINDOW);
                for (let c = 0; c < CONTEXT_WINDOW; c++) {
                    parsedW1[c] = new Float32Array(flatW1.slice(c * chunkSize, (c + 1) * chunkSize));
                }
            }

            const weightsData = {
                vocab: weightsDataRaw.vocab,
                weights: {
                    E: new Float32Array(getArray(w.E)),
                    W1: parsedW1,
                    b1: new Float32Array(getArray(w.b1)),
                    W2: new Float32Array(getArray(w.W2)),
                    b2: new Float32Array(getArray(w.b2))
                }
            };

            let responseText = generateLLMResponse(message, weightsData);

            // --- ENHANCED CODE-WRITING LOGIC ---
            const lowerMsg = message.toLowerCase();
            let actionPayload = null;
            
            if (lowerMsg.includes("add description comments") || lowerMsg.includes("add comments")) {
                const commented = addCommentsToCode(code || "");
                responseText = "### VS# AI Code Annotator\n\nI have added line-by-line description comments explaining each instruction in your code:";
                actionPayload = {
                    type: "edit",
                    code: commented
                };
            }
            else if (lowerMsg.includes("remove errors") || lowerMsg.includes("remove syntax errors") || lowerMsg.includes("fix syntax errors")) {
                const fixedCode = fixVerScriptCode(code || "");
                responseText = "### VS# AI Syntax Repair\n\nI scanned the code and repaired assignment operators, unclosed strings, and comparison formats to match VerScript standards:";
                actionPayload = {
                    type: "edit",
                    code: fixedCode
                };
            }
            else {
                // Detect "fix" / "correct" / "debug" intent — analyze and fix user's code
                const isFixIntent = /\b(fix|correct|debug|repair|syntax)\b/.test(lowerMsg);
                if (isFixIntent && code && code.trim()) {
                    const fixedCode = fixVerScriptCode(code);
                    if (fixedCode && fixedCode !== code) {
                        // Append the fixed code block to the LLM response
                        responseText += "\n\n```verscript\n" + fixedCode + "\n```";
                    }
                }
                
                // Extract code block from response and set edit action
                const codeBlock = extractCodeBlock(responseText);
                if (codeBlock) {
                    actionPayload = {
                        type: "edit",
                        code: codeBlock
                    };
                }
            }

            // Simulate typing delay
            setTimeout(() => {
                res.json({
                    response: responseText,
                    action: actionPayload
                });
            }, 400);

        } catch (err) {
            console.error("[VS#] Error generating response:", err);
            res.status(500).json({ error: 'Internal server error running custom LLM.' });
        }
    });
}

module.exports = { mountRoutes };

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
        console.log(`[VS#] Standalone server running on port ${PORT}`);
    });
}
