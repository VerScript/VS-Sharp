
const fs = require('fs');
const path = require('path');
const { exec } = require('child_process');
const os = require('os');
const { GoogleGenAI } = require('@google/genai');

require('dotenv').config();

// --- EXTRACT CODE BLOCK ---
function extractCodeBlock(text) {
    const regex = /\`\`\`verscript\r?\n([\s\S]*?)\`\`\`/i;
    const match = text.match(regex);
    if (match) return match[1].trim();
    
    const fallbackRegex = /\`\`\`\r?\n([\s\S]*?)\`\`\`/i;
    const fallbackMatch = text.match(fallbackRegex);
    return fallbackMatch ? fallbackMatch[1].trim() : null;
}

// --- SMART CODE FIX ---
function fixVerScriptCode(code) {
    if (!code || !code.trim()) return null;
    
    const lines = code.split('\n');
    const fixed = lines.map(line => {
        let trimmed = line.trim();
        if (!trimmed || trimmed.startsWith('!')) return line;
        
        if (trimmed.startsWith('display "') && !trimmed.endsWith('"')) {
            return line + '"';
        }
        
        if (trimmed.startsWith('display') && trimmed.length > 7 && trimmed[7] !== ' ') {
            return line.replace('display', 'display ');
        }
        
        return line;
    });
    
    return fixed.join('\n');
}

// --- MOUNT ROUTES ---
function mountRoutes(app, basePath) {
    const prefix = basePath || '';
    
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

        try { fs.chmodSync(VERSCRIPT_BIN, 0o755); } catch(_) {}

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

    app.post(prefix + '/api/chat', async (req, res) => {
        const { code, message } = req.body;
        
        if (!message) {
            return res.status(400).json({ error: 'Message is required' });
        }

        console.log(`[VS#] Received: "${message}"`);

        try {
            const ai = new GoogleGenAI({});
            let prompt = `You are VS-Sharp (VS#), an AI assistant for the VerScript programming language. ${message}`;
            if (code) {
                prompt += `\n\nHere is the user's current code:\n\`\`\`verscript\n${code}\n\`\`\`\n`;
            }

            const response = await ai.models.generateContent({
                model: 'gemini-2.0-flash',
                contents: prompt,
            });

            let responseText = response.text;
            let actionPayload = null;
            
            const isFixIntent = /\b(fix|correct|debug|repair|syntax)\b/.test(message.toLowerCase());
            if (isFixIntent && code && code.trim()) {
                const fixedCode = fixVerScriptCode(code);
                if (fixedCode && fixedCode !== code) {
                    responseText += "\n\n\`\`\`verscript\n" + fixedCode + "\n\`\`\`";
                }
            }
            
            const codeBlock = extractCodeBlock(responseText);
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
            console.error("[VS#] Error generating response:", err);
            res.status(500).json({ error: 'Internal server error running LLM.' });
        }
    });
}

module.exports = { mountRoutes };
