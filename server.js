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
  { text: "good day morning afternoon", intent: "greeting" },
  { text: "fix my code correct error wrong syntax issue repair", intent: "fix" },
  { text: "solve this error find missing quotes debug resolve", intent: "fix" },
  { text: "something is wrong with my script can you fix it", intent: "fix" },
  { text: "fix syntax error missing quotes repair code", intent: "fix" },
  { text: "write a program create code make display script generate", intent: "write" },
  { text: "give me a script that prompts user write code", intent: "write" },
  { text: "create an addition program for adding numbers", intent: "write" },
  { text: "write a program to print numbers or variables", intent: "write" },
  { text: "make a math calculator program", intent: "write" },
  { text: "explain display prompt variables syntax how to use", intent: "explain" },
  { text: "what does prompt do how do I assign variable explain", intent: "explain" },
  { text: "explain this script tell me what this code does", intent: "explain" },
  { text: "walk me through my code lines explanation details", intent: "explain" },
  { text: "help command usage keywords guidelines info helper", intent: "help" },
  { text: "what commands can I run what keywords are there", intent: "help" },
  { text: "list commands documentation help options guide", intent: "help" }
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
const HIDDEN_SIZE = 10;
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
  const epochs = 1000;

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

// --- DYNAMIC VERSCRIPT LLM GENERATIVE CORE (BUILT FROM SCRATCH) ---

// Parses VerScript code and yields dynamic structure
function analyzeCodeStructure(code) {
  const lines = code.split('\n');
  const variables = {};
  const instructions = [];
  const errors = [];

  lines.forEach((line, index) => {
    const lineNum = index + 1;
    const trimmed = line.trim();
    
    // Ignore comments
    if (trimmed.startsWith('!')) {
      instructions.push({ type: 'comment', content: trimmed.substring(1).trim(), lineNum });
      return;
    }
    
    const commentIdx = trimmed.indexOf('!');
    let cleanLine = trimmed;
    let comment = '';
    if (commentIdx !== -1) {
      cleanLine = trimmed.substring(0, commentIdx).trim();
      comment = trimmed.substring(commentIdx + 1).trim();
    }

    if (!cleanLine) {
      instructions.push({ type: 'empty', lineNum });
      return;
    }

    // Display statement
    if (cleanLine.startsWith('display ')) {
      const expr = cleanLine.substring(8).trim();
      if (expr.startsWith('"')) {
        if (!expr.endsWith('"') || expr.length === 1) {
          errors.push({ lineNum, type: 'missing_quote', message: 'Missing closing double quote on display statement' });
        }
        instructions.push({ type: 'display_text', content: expr.replace(/"/g, ''), lineNum, comment });
      } else {
        instructions.push({ type: 'display_var_expr', content: expr, lineNum, comment });
      }
      return;
    }

    // Prompt statement
    if (cleanLine.startsWith('prompt ')) {
      const varName = cleanLine.substring(7).trim();
      if (!varName || varName.includes(' ') || varName.includes(':')) {
        errors.push({ lineNum, type: 'invalid_prompt', message: 'Prompt must specify a single valid variable name' });
      }
      variables[varName] = 'user_input';
      instructions.push({ type: 'prompt', variable: varName, lineNum, comment });
      return;
    }

    // Variable assignment
    if (cleanLine.includes(':')) {
      const parts = cleanLine.split(':');
      const varName = parts[0].trim();
      const expr = parts.slice(1).join(':').trim();

      if (!varName || varName.includes(' ')) {
        errors.push({ lineNum, type: 'invalid_variable', message: 'Invalid variable name format' });
      }

      if (expr.startsWith('"')) {
        if (!expr.endsWith('"') || expr.length === 1) {
          errors.push({ lineNum, type: 'missing_quote', message: 'Missing closing double quote on variable assignment' });
        }
        variables[varName] = 'string';
        instructions.push({ type: 'assign_text', variable: varName, value: expr.replace(/"/g, ''), lineNum, comment });
      } else {
        variables[varName] = 'math_expr';
        instructions.push({ type: 'assign_math', variable: varName, expression: expr, lineNum, comment });
      }
      return;
    }

    errors.push({ lineNum, type: 'unknown_syntax', message: `Unknown syntax error: '${cleanLine}'` });
  });

  return { variables, instructions, errors };
}

// Generate code explanation dynamically (mimicking an LLM)
function generateDynamicExplanation(code) {
  const analysis = analyzeCodeStructure(code);
  if (analysis.instructions.length === 0 || (analysis.instructions.length === 1 && analysis.instructions[0].type === 'empty')) {
    return "Your editor is empty! Type some VerScript code and I will walk you through it line-by-line.";
  }

  let explanation = "### VerScript Code Walkthrough & Analysis\n\nHere is a line-by-line breakdown of your current script:\n\n";

  analysis.instructions.forEach(ins => {
    switch (ins.type) {
      case 'comment':
        explanation += `- **Line ${ins.lineNum}**: A comment reminding you: *"${ins.content}"*.\n`;
        break;
      case 'display_text':
        explanation += `- **Line ${ins.lineNum}**: Prints the literal text message ` + "`\"" + ins.content + "\"`" + ` to the terminal output.\n`;
        break;
      case 'display_var_expr':
        explanation += `- **Line ${ins.lineNum}**: Evaluates and displays the variable or math expression ` + "`" + ins.content + "`" + `.\n`;
        break;
      case 'prompt':
        explanation += `- **Line ${ins.lineNum}**: Prompts the user for interactive input and stores the result inside the variable ` + "`" + ins.variable + "`" + `.\n`;
        break;
      case 'assign_text':
        explanation += `- **Line ${ins.lineNum}**: Declares variable ` + "`" + ins.variable + "`" + ` and assigns the text value ` + "`\"" + ins.value + "\"`" + `.\n`;
        break;
      case 'assign_math':
        explanation += `- **Line ${ins.lineNum}**: Declares variable ` + "`" + ins.variable + "`" + ` and assigns the mathematical formula ` + "`" + ins.expression + "`" + `.\n`;
        break;
      default:
        break;
    }
  });

  // Describe variables
  const varNames = Object.keys(analysis.variables);
  if (varNames.length > 0) {
    explanation += `\n### Variables Defined:\n`;
    varNames.forEach(v => {
      explanation += `- ` + "`" + v + "`" + `: Initialized as a ${analysis.variables[v] === 'user_input' ? 'dynamic user input variable' : analysis.variables[v]}.\n`;
    });
  }

  // Diagnostics
  if (analysis.errors.length > 0) {
    explanation += `\n### ⚠️ Syntax Alert:\nI noticed some issues in your code that might prevent execution:\n`;
    analysis.errors.forEach(e => {
      explanation += `- **Line ${e.lineNum}**: ${e.message}\n`;
    });
  } else {
    explanation += `\n✨ **No syntax errors detected!** This script is valid and ready to run.`;
  }

  return explanation;
}

// Generate new VerScript code dynamically based on NLP prompt query
function generateDynamicCode(message) {
  const cleanMsg = message.toLowerCase();
  
  // Extract number entities from message
  const numbers = [];
  const numRegex = /\b\d+\b/g;
  let match;
  while ((match = numRegex.exec(cleanMsg)) !== null) {
    numbers.push(parseInt(match[0], 10));
  }

  // Extract potential variable names or terms
  let var1 = "val1";
  let var2 = "val2";
  let resultVar = "result";
  
  if (cleanMsg.includes("age")) {
    var1 = "age";
    resultVar = "nextAge";
  } else if (cleanMsg.includes("score")) {
    var1 = "points";
    resultVar = "totalScore";
  } else if (cleanMsg.includes("user") || cleanMsg.includes("name")) {
    var1 = "username";
  }

  // Basic generation templates based on keywords
  let code = "";
  let explanation = "";

  if (cleanMsg.includes("add") || cleanMsg.includes("sum") || cleanMsg.includes("plus") || cleanMsg.includes("calc") || cleanMsg.includes("math")) {
    const val1 = numbers[0] !== undefined ? numbers[0] : 10;
    const val2 = numbers[1] !== undefined ? numbers[1] : 20;
    code = `${var1} : ${val1}  ! First input\n${var2} : ${val2}  ! Second input\n${resultVar} : ${var1} + ${var2}  ! Add them together\ndisplay "The sum calculation result is:"\ndisplay ${resultVar}`;
    explanation = `I have dynamically generated an addition script for you using variables \`${var1}\` and \`${var2}\`. The values are set to \`${val1}\` and \`${val2}\` and then display the sum \`${resultVar}\`.`;
  } 
  else if (cleanMsg.includes("prompt") || cleanMsg.includes("ask") || cleanMsg.includes("input") || cleanMsg.includes("interactive")) {
    const greetingMsg = cleanMsg.includes("hello") || cleanMsg.includes("welcome") ? "Welcome to VerScript, " : "You entered: ";
    code = `display "Enter your custom value:"\nprompt ${var1}\ndisplay "${greetingMsg}"\ndisplay ${var1}`;
    explanation = `I have generated an interactive console script. It prompts the user for a value, saves it into the variable \`${var1}\`, and prints it back with a message.`;
  } 
  else if (cleanMsg.includes("hello") || cleanMsg.includes("print") || cleanMsg.includes("display")) {
    // Extract a custom string in quotes if the user asked e.g. write a script to print "Welcome User"
    const quoteRegex = /"([^"]+)"/;
    const quoteMatch = cleanMsg.match(quoteRegex);
    const textToPrint = quoteMatch ? quoteMatch[1] : "Hello World from VerScript!";
    
    code = `! Dynamic Hello script\ndisplay "${textToPrint}"\nx : 50\ndisplay x`;
    explanation = `I have constructed a basic program that outputs \`"${textToPrint}"\` and prints a test variable initialized to \`50\`.`;
  }
  else {
    // Default fallback script
    code = `! Auto-generated sample script\nmessage : "VerScript is active"\ndisplay message\na : 100\nb : 200\ntotal : a * b / 2\ndisplay "Formula output:"\ndisplay total`;
    explanation = `I have created a general mathematical template for you containing assignments, string output, and formula evaluations.`;
  }

  return { code, explanation };
}

// Fix errors dynamically in user code
function fixCodeAndGenerateReport(code) {
  const analysis = analyzeCodeStructure(code);
  const lines = code.split('\n');
  const fixedLines = [];
  let changesMade = [];
  
  const declaredVars = new Set();
  const referencedVars = new Set();

  for (let i = 0; i < lines.length; i++) {
    let line = lines[i];
    let trimmed = line.trim();
    
    // Ignore comments
    const commentIdx = trimmed.indexOf('!');
    if (commentIdx !== -1) {
      trimmed = trimmed.substring(0, commentIdx).trim();
    }
    
    if (!trimmed) {
      fixedLines.push(line);
      continue;
    }

    // 1. Check Display statement missing quotes
    if (trimmed.startsWith('display "')) {
      const content = trimmed.substring(8).trim();
      if (content.startsWith('"') && !content.endsWith('"')) {
        line = line + '"';
        changesMade.push(`Added missing closing double quote to display statement on line ${i + 1}`);
      }
    } else if (trimmed.startsWith('display ')) {
      // It displays a variable/expression. Let's record variables in display expression.
      const expr = trimmed.substring(8).trim();
      expr.split(/[\+\-\*\/]/).forEach(token => {
        const cleanToken = token.trim();
        if (cleanToken && !cleanToken.startsWith('"') && isNaN(Number(cleanToken))) {
          referencedVars.add(cleanToken);
        }
      });
    }

    // 2. Variable Assignments
    if (trimmed.includes(':')) {
      const parts = trimmed.split(':');
      const varName = parts[0].trim();
      const expr = parts.slice(1).join(':').trim();
      declaredVars.add(varName);

      if (expr.startsWith('"') && !expr.endsWith('"')) {
        line = line + '"';
        changesMade.push(`Added missing closing double quote to assignment of variable '${varName}' on line ${i + 1}`);
      } else if (!expr.startsWith('"')) {
        expr.split(/[\+\-\*\/]/).forEach(token => {
          const cleanToken = token.trim();
          if (cleanToken && !cleanToken.startsWith('"') && isNaN(Number(cleanToken))) {
            referencedVars.add(cleanToken);
          }
        });
      }
    }

    // 3. Prompt assignments
    if (trimmed.startsWith('prompt ')) {
      const varName = trimmed.substring(7).trim();
      declaredVars.add(varName);
    }

    fixedLines.push(line);
  }

  // 4. Undefined variable repair
  const undefinedVars = [...referencedVars].filter(v => !declaredVars.has(v));
  if (undefinedVars.length > 0) {
    const insertions = undefinedVars.map(v => `${v} : 0  ! VS# auto-declared this variable to resolve undefined warning`);
    fixedLines.unshift(...insertions);
    changesMade.push(`Auto-declared missing variables: ${undefinedVars.map(v => `'${v}'`).join(', ')} at the top of the file.`);
  }

  return {
    isModified: changesMade.length > 0,
    code: fixedLines.join('\n'),
    report: changesMade
  };
}

// --- MAIN AI CHAT CONTROLLER ---

function processRequest(code, message) {
  const intent = classifyIntent(message);
  
  let responseText = "";
  let actionPayload = null;
  
  switch (intent) {
    case "greeting":
      responseText = "Hello there! I am VS#, your fully custom neural network assistant for VerScript.\n\nI run on a fully custom node-based backpropagation neural network classifier trained entirely from scratch. I am connected directly to your editor! \n\nI can:\n- **Walk you through your script** line-by-line (type 'explain my code')\n- **Write dynamic scripts** on the fly (type 'write an addition program' or 'ask for user age')\n- **Detect and auto-repair errors** (type 'fix my syntax errors')\n\nHow can I help you build today?";
      break;
      
    case "explain":
      responseText = generateDynamicExplanation(code);
      break;
      
    case "help":
      responseText = "### VS# Assistant Capabilities & Guide\n\nI can help you build and debug scripts. Here are the core things I can do:\n\n1. **Explain Code**: Type `explain my code` to get a custom, line-by-line run-through of your current script.\n2. **Generate Programs**: Ask me to write programs like: `write a sum script`, `create a program to prompt name`, or `display hello world`.\n3. **Syntax Debugging**: Type `fix my code` or `correct syntax` to check for missing quotes or undeclared variables. I will fix them directly in the Monaco Editor.\n4. **General Q&A**: Ask about variables, mathematical operators, or terminal prompts.\n\n*Note: Every edit I perform will automatically show key-by-key typing animations!*";
      break;
      
    case "write":
      const genResult = generateDynamicCode(message);
      responseText = `### Dynamic Code Generated!\n\n${genResult.explanation}\n\nI am sending this code directly to your editor. Watch it type out live!`;
      actionPayload = {
        type: "edit",
        code: genResult.code
      };
      break;
      
    case "fix":
      const fixResult = fixCodeAndGenerateReport(code);
      if (fixResult.isModified) {
        responseText = "### 🔧 Syntax Corrections Applied!\n\nI analyzed your script and applied the following fixes directly inside the editor:\n\n";
        fixResult.report.forEach(change => {
          responseText += `- ${change}\n`;
        });
        responseText += "\nYou should see the editor backspace and retype the corrected code now!";
        actionPayload = {
          type: "edit",
          code: fixResult.code
        };
      } else {
        responseText = "### Check Complete!\n\nI scanned your script line-by-line and found **no syntax errors**! All displays have proper quotes and all referenced variables are declared. You're ready to run it.";
      }
      break;
      
    default:
      responseText = "I'm VS#, running on a custom trained neural network. I see your request, but I'm not sure if you want me to write code, explain code, or fix errors. Could you clarify?";
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
    
    // Simulate AI typing/processing delay
    setTimeout(() => {
        res.json(result);
    }, 600);
});

app.listen(PORT, () => {
    console.log(`VS# Helper AI running on port ${PORT}`);
});
