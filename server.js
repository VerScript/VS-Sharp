const express = require('express');
const cors = require('cors');

const app = express();
app.use(cors());
app.use(express.json());

const PORT = 3001;

// Simulated AI Logic for VerScript
function analyzeCode(code, userMessage) {
    const lines = code.split('\n');
    let response = "I'm VS#, your VerScript assistant! ";
    
    // Simple heuristics
    let missingQuotes = false;
    let missingVariables = false;
    
    for (let i = 0; i < lines.length; i++) {
        let line = lines[i].trim();
        // Ignore comments
        if (line.includes('!')) {
            line = line.substring(0, line.indexOf('!')).trim();
        }
        if (!line) continue;
        
        // Check display syntax
        if (line.startsWith('display ')) {
            const arg = line.substring(8).trim();
            if (arg.startsWith('"') && !arg.endsWith('"')) {
                missingQuotes = true;
            } else if (!arg.startsWith('"') && arg.match(/^[a-zA-Z_]+$/)) {
                // It's a variable display, we can't easily check if it's declared here without a full parser,
                // but we can pretend we are analyzing it.
            }
        }
    }
    
    const msg = userMessage.toLowerCase();
    
    if (msg.includes('error') || msg.includes('wrong') || msg.includes('fix')) {
        if (missingQuotes) {
            return "It looks like you have a missing closing quote `\"` on one of your `display` statements. Make sure all strings are properly wrapped!";
        } else {
            return "Your syntax looks mostly correct to me. Are you getting an error in the terminal output?";
        }
    }
    
    if (msg.includes('hello') || msg.includes('hi')) {
        return "Hello! I am VS#, your AI assistant. I can analyze your VerScript code in the editor. Ask me to find errors or explain syntax!";
    }
    
    return "I'm currently running in simulated heuristic mode! I see your code is " + lines.length + " lines long. " + (missingQuotes ? "Watch out for those missing quotes!" : "Looks good syntactically!");
}

app.post('/api/chat', (req, res) => {
    const { code, message } = req.body;
    
    if (!message) {
        return res.status(400).json({ error: 'Message is required' });
    }
    
    const aiResponse = analyzeCode(code || '', message);
    
    // Simulate AI typing delay
    setTimeout(() => {
        res.json({ response: aiResponse });
    }, 800);
});

app.listen(PORT, () => {
    console.log(`VS# Helper AI running on http://localhost:${PORT}`);
});
