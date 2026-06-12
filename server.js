const express = require('express');
const cors = require('cors');

const app = express();
app.use(cors());
app.use(express.json());

const PORT = process.env.PORT || 3001;

// --- CUSTOM NEURAL NETWORK INTENT CLASSIFIER (BUILT FROM SCRATCH) ---

const trainingData = [
  { text: "hello hi hey there greetings salut yo hello!", intent: "greeting" },
  { text: "how are you who are you what is your name", intent: "greeting" },
  { text: "fix my code correct error wrong syntax issue repair", intent: "fix" },
  { text: "solve this error find missing quotes debug resolve", intent: "fix" },
  { text: "something is wrong with my script can you fix it", intent: "fix" },
  { text: "write a program create code make display script generate", intent: "write" },
  { text: "give me a script that prompts user write code", intent: "write" },
  { text: "create an addition program for adding numbers", intent: "write" },
  { text: "explain display prompt variables syntax how to use", intent: "explain" },
  { text: "what does prompt do how do I assign variable explain", intent: "explain" },
  { text: "help command usage keywords guidelines info helper", intent: "help" },
  { text: "what commands can I run what keywords are there", intent: "help" }
];

// Tokenize and clean text
function tokenize(text) {
  return text.toLowerCase()
             .replace(/[.,\/#!$%\^&\*;:{}=\-_`~()?]/g, "")
             .split(/\s+/)
             .filter(w => w.length > 0);
}

// Build vocabulary
const vocab = [];
trainingData.forEach(d => {
  tokenize(d.text).forEach(w => {
    if (!vocab.includes(w)) vocab.push(w);
  });
});

function getBagOfWords(text) {
  const vector = new Array(vocab.length).fill(0);
  tokenize(text).forEach(w => {
    const idx = vocab.indexOf(w);
    if (idx !== -1) vector[idx] = 1;
  });
  return vector;
}

// Neural Network Structure
const INPUT_SIZE = vocab.length;
const HIDDEN_SIZE = 8;
const OUTPUT_SIZE = 5; // greeting, fix, write, explain, help
const INTENTS = ["greeting", "fix", "write", "explain", "help"];

let W1 = Array.from({ length: INPUT_SIZE }, () => Array.from({ length: HIDDEN_SIZE }, () => Math.random() * 2 - 1));
let b1 = Array.from({ length: HIDDEN_SIZE }, () => 0);
let W2 = Array.from({ length: HIDDEN_SIZE }, () => Array.from({ length: OUTPUT_SIZE }, () => Math.random() * 2 - 1));
let b2 = Array.from({ length: OUTPUT_SIZE }, () => 0);

function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}

function sigmoidDerivative(x) {
  return x * (1 - x);
}

function softmax(arr) {
  const max = Math.max(...arr);
  const exps = arr.map(x => Math.exp(x - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map(x => x / sum);
}

function forward(input) {
  const hidden = new Array(HIDDEN_SIZE).fill(0);
  for (let j = 0; j < HIDDEN_SIZE; j++) {
    let sum = b1[j];
    for (let i = 0; i < INPUT_SIZE; i++) {
      sum += input[i] * W1[i][j];
    }
    hidden[j] = sigmoid(sum);
  }

  const output = new Array(OUTPUT_SIZE).fill(0);
  for (let k = 0; k < OUTPUT_SIZE; k++) {
    let sum = b2[k];
    for (let j = 0; j < HIDDEN_SIZE; j++) {
      sum += hidden[j] * W2[j][k];
    }
    output[k] = sum;
  }
  const probs = softmax(output);
  return { hidden, probs };
}

// Backpropagation Training
function train() {
  const learningRate = 0.15;
  const epochs = 800;

  for (let epoch = 0; epoch < epochs; epoch++) {
    trainingData.forEach(d => {
      const input = getBagOfWords(d.text);
      const target = new Array(OUTPUT_SIZE).fill(0);
      target[INTENTS.indexOf(d.intent)] = 1;

      const { hidden, probs } = forward(input);

      // Backpropagation
      const outputErrors = probs.map((p, i) => p - target[i]);

      const hiddenErrors = new Array(HIDDEN_SIZE).fill(0);
      for (let j = 0; j < HIDDEN_SIZE; j++) {
        let sum = 0;
        for (let k = 0; k < OUTPUT_SIZE; k++) {
          sum += outputErrors[k] * W2[j][k];
        }
        hiddenErrors[j] = sum * sigmoidDerivative(hidden[j]);
      }

      // Update Weights and Biases
      for (let j = 0; j < HIDDEN_SIZE; j++) {
        for (let k = 0; k < OUTPUT_SIZE; k++) {
          W2[j][k] -= learningRate * outputErrors[k] * hidden[j];
        }
      }
      for (let k = 0; k < OUTPUT_SIZE; k++) {
        b2[k] -= learningRate * outputErrors[k];
      }

      for (let i = 0; i < INPUT_SIZE; i++) {
        for (let j = 0; j < HIDDEN_SIZE; j++) {
          W1[i][j] -= learningRate * hiddenErrors[j] * input[i];
        }
      }
      for (let j = 0; j < HIDDEN_SIZE; j++) {
        b1[j] -= learningRate * hiddenErrors[j];
      }
    });
  }
}

// Train classifier on startup
console.log("Training Neural Network from scratch...");
train();
console.log("Neural Network trained successfully!");

function classifyIntent(message) {
  const input = getBagOfWords(message);
  const { probs } = forward(input);
  const maxIdx = probs.indexOf(Math.max(...probs));
  return INTENTS[maxIdx];
}

// --- VERSCRIPT CODE ANALYZER AND PARSER ---

function parseAndFixCode(code) {
  const lines = code.split('\n');
  const fixedLines = [];
  let isModified = false;
  const declaredVars = new Set();
  const referencedVars = new Set();
  
  // Phase 1: Track declarations and correct simple missing quotes
  for (let i = 0; i < lines.length; i++) {
    let line = lines[i];
    let trimmed = line.trim();
    
    // Skip comments
    const commentIdx = trimmed.indexOf('!');
    if (commentIdx !== -1) {
      trimmed = trimmed.substring(0, commentIdx).trim();
    }
    if (!trimmed) {
      fixedLines.push(line);
      continue;
    }
    
    // Check variable declarations (var: expr)
    if (trimmed.includes(':')) {
      const parts = trimmed.split(':');
      const varName = parts[0].trim();
      declaredVars.add(varName);
      
      // Track variables in the assignment expression
      const expr = parts.slice(1).join(':').trim();
      expr.split(/[\+\-\*\/]/).forEach(token => {
        const cleanToken = token.trim();
        if (cleanToken && !cleanToken.startsWith('"') && isNaN(Number(cleanToken))) {
          referencedVars.add(cleanToken);
        }
      });
    }
    
    // Check missing display quotes
    if (trimmed.startsWith('display "')) {
      const content = trimmed.substring(8).trim();
      if (content.startsWith('"') && !content.endsWith('"')) {
        // Fix missing closing quote
        line = line + '"';
        isModified = true;
      }
    } else if (trimmed.startsWith('display ')) {
      // Track referenced variables in display
      const expr = trimmed.substring(8).trim();
      expr.split(/[\+\-\*\/]/).forEach(token => {
        const cleanToken = token.trim();
        if (cleanToken && !cleanToken.startsWith('"') && isNaN(Number(cleanToken))) {
          referencedVars.add(cleanToken);
        }
      });
    }
    
    // Track variables in prompt
    if (trimmed.startsWith('prompt ')) {
      const varName = trimmed.substring(7).trim();
      declaredVars.add(varName);
    }
    
    fixedLines.push(line);
  }
  
  // Phase 2: Detect undefined variables
  const undefinedVars = [...referencedVars].filter(v => !declaredVars.has(v));
  if (undefinedVars.length > 0) {
    // Insert default declarations at the top of the script
    const insertions = undefinedVars.map(v => `${v} : 0  ! VS# defined this variable to prevent syntax errors`);
    fixedLines.unshift(...insertions);
    isModified = true;
  }
  
  return {
    isModified: isModified,
    code: fixedLines.join('\n'),
    diagnostics: {
      undefinedVars,
      hasModifiedQuotes: isModified && undefinedVars.length === 0
    }
  };
}

// --- MAIN AI CHAT CONTROLLER ---

function processRequest(code, message) {
  const intent = classifyIntent(message);
  const cleanMsg = message.toLowerCase();
  
  let responseText = "";
  let actionPayload = null;
  
  switch (intent) {
    case "greeting":
      responseText = "Hello! I am VS#, your fully custom neural network assistant for VerScript.\n\nI can analyze your editor code, explain language keywords, and even **write or fix code directly in your editor**. Try asking me to write a calculator or find syntax errors!";
      break;
      
    case "explain":
      responseText = "### VerScript Syntax Reference:\n\n1. **Console Output**:\n   `display \"Hello World\"` - Outputs text or numbers.\n\n2. **Variables**:\n   `x : 10` - Assigns a value to a variable.\n\n3. **Arithmetic**:\n   `sum : x + 5` - Supports `+`, `-`, `*`, and `/` operators.\n\n4. **User Input**:\n   `prompt age` - Asks the user for input and saves it to a variable.";
      break;
      
    case "help":
      responseText = "I can help you build programs in VerScript! Available intents I understand:\n- **Greeting**: Say hello to interact.\n- **Explain**: Learn about syntax.\n- **Write Code**: Ask me to generate scripts (e.g., 'write a prompt script').\n- **Fix Errors**: Ask me to fix your code directly in the editor.";
      break;
      
    case "write":
      let generatedCode = "";
      if (cleanMsg.includes("add") || cleanMsg.includes("calc") || cleanMsg.includes("math") || cleanMsg.includes("sum")) {
        generatedCode = `num1 : 15\nnum2 : 25\nsum : num1 + num2\ndisplay "The sum is:"\ndisplay sum`;
        responseText = "I've written an addition program that defines two numbers, adds them, and displays the sum in the console.";
      } else if (cleanMsg.includes("prompt") || cleanMsg.includes("name") || cleanMsg.includes("welcome") || cleanMsg.includes("ask")) {
        generatedCode = `display "Enter your name:"\nprompt username\ndisplay "Welcome back, "\ndisplay username`;
        responseText = "I've generated an interactive prompt script that asks for a username and displays a welcome message.";
      } else {
        generatedCode = `display "Hello World from VerScript!"\nx : 100\ndisplay x`;
        responseText = "I've written a basic hello world script with a variable declaration.";
      }
      
      actionPayload = {
        type: "edit",
        code: generatedCode
      };
      break;
      
    case "fix":
      const fixResult = parseAndFixCode(code);
      if (fixResult.isModified) {
        responseText = "I analyzed your code and found issues! I've corrected them directly in the editor:\n";
        if (fixResult.diagnostics.undefinedVars.length > 0) {
          responseText += `- **Undefined variables**: Declared \`${fixResult.diagnostics.undefinedVars.join(", ")}\` at the top of the file.\n`;
        }
        if (fixResult.diagnostics.hasModifiedQuotes) {
          responseText += `- **Missing quotes**: Added missing closing double quotes to your display statement(s).\n`;
        }
        actionPayload = {
          type: "edit",
          code: fixResult.code
        };
      } else {
        responseText = "I analyzed your code and couldn't find any common syntax errors! Your syntax (missing quotes or undefined variables) looks correct.";
      }
      break;
      
    default:
      responseText = "I'm VS#, running on a local neural network classifier. I see your message, but I'm not entirely sure how to respond. Ask me to 'help', 'explain syntax', 'write code', or 'fix errors'!";
  }
  
  return {
    response: responseText,
    action: actionPayload
  };
}

app.post('/api/chat', (req, res) => {
    const { code, message } = req.body;
    
    if (!message) {
        return res.status(400).json({ error: 'Message is required' });
    }
    
    const result = processRequest(code || '', message);
    
    // Simulate AI typing delay
    setTimeout(() => {
        res.json(result);
    }, 600);
});

app.listen(PORT, () => {
    console.log(`VS# Helper AI running on port ${PORT}`);
});
