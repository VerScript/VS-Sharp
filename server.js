require('dotenv').config();
const express = require('express');
const cors = require('cors');
const { GoogleGenAI } = require('@google/genai');

const app = express();
app.use(cors());
app.use(express.json());

const PORT = 3001;

// Real Gemini API client initialization
let ai = null;
if (process.env.GEMINI_API_KEY) {
    try {
        ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });
        console.log("VS# Helper: Gemini API client successfully initialized.");
    } catch (e) {
        console.error("VS# Helper: Failed to initialize Gemini client:", e);
    }
} else {
    console.warn("VS# Helper: GEMINI_API_KEY environment variable not set. Running in simulated smart fallback mode.");
}

// Simulated Smart Fallback for VerScript
function simulateLlm(code, userMessage) {
    const lines = code.split('\n');
    const msg = userMessage.toLowerCase();
    
    // Heuristic: Check for missing quotes in display
    let missingQuotes = false;
    for (let i = 0; i < lines.length; i++) {
        let line = lines[i].trim();
        if (line.includes('!')) {
            line = line.substring(0, line.indexOf('!')).trim();
        }
        if (line.startsWith('display ')) {
            const arg = line.substring(8).trim();
            if (arg.startsWith('"') && !arg.endsWith('"')) {
                missingQuotes = true;
            }
        }
    }

    if (msg.includes('fix') || msg.includes('correct') || msg.includes('error') || msg.includes('wrong')) {
        if (missingQuotes) {
            let fixedCode = lines.map(line => {
                let trimmed = line.trim();
                if (trimmed.startsWith('display "') && !trimmed.endsWith('"') && !trimmed.includes('!')) {
                    return line + '"';
                }
                return line;
            }).join('\n');
            
            return {
                response: "I found a missing closing quote on your display statement. I've automatically added the missing quote to fix the syntax error!",
                newCode: fixedCode
            };
        }
    }

    if (msg.includes('add') || msg.includes('write') || msg.includes('create')) {
        if (msg.includes('loop') || msg.includes('math') || msg.includes('variable')) {
            let extraCode = "\n! Added by VS#\nx : 5\ny : 10\nresult : x + y\ndisplay \"Result is:\"\ndisplay result";
            return {
                response: "I've added a math evaluation example with variables `x` and `y` and displayed their sum.",
                newCode: code + extraCode
            };
        }
    }

    // Default conversational response
    return {
        response: "Hello! I am VS#, your AI assistant. (Running in Smart Simulator Mode. Set GEMINI_API_KEY in your .env to enable the full live Gemini LLM). How can I help you write or fix your VerScript code today?",
        newCode: code
    };
}

app.post('/api/chat', async (req, res) => {
    const { code, message } = req.body;
    
    if (!message) {
        return res.status(400).json({ error: 'Message is required' });
    }

    // If Gemini client is active, use the real LLM
    if (ai) {
        try {
            const prompt = `The user is writing VerScript code in their editor.
Current Editor Code:
"""
${code || ''}
"""

User Message: "${message}"

Respond to the user's message, explain any changes, and write or modify code if they ask you to.`;

            const response = await ai.models.generateContent({
                model: 'gemini-2.5-flash',
                contents: prompt,
                config: {
                    systemInstruction: `You are VS#, an AI assistant for VerScript (a beginner programming language).
VerScript syntax rules:
1. Output statements: 'display <expression>'. The expression can be a string (wrapped in double quotes like "hello"), a number, or a variable name. Math operations (+, -, *, /) are evaluated left-to-right.
2. Variable declaration & assignment: '<variable> : <expression>'. Variables are dynamically typed.
3. User Input: 'prompt <variable>'. This prompts the user for input.
4. Comments start with '!' and go until the end of the line.

You must respond in JSON format matching this JSON schema:
{
  "response": "Your textual explanation or response to the user's query.",
  "newCode": "The full code after any edits. If you did not make any edits or write any code, this must be identical to the input code."
}`,
                    responseMimeType: 'application/json'
                }
            });

            const result = JSON.parse(response.text.trim());
            return res.json({
                response: result.response || "Here is the response.",
                newCode: result.newCode || code
            });
        } catch (err) {
            console.error("Gemini API Error, falling back to simulator:", err);
            // Fall back to simulator if API call fails
            const fallback = simulateLlm(code || '', message);
            return res.json(fallback);
        }
    }

    // Fallback if no API client is initialized
    const fallback = simulateLlm(code || '', message);
    // Simulate short network latency
    setTimeout(() => {
        res.json(fallback);
    }, 600);
});

app.listen(PORT, () => {
    console.log(`VS# Helper AI running on http://localhost:${PORT}`);
});
