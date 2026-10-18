// build_dataset.js — Polyglot & Multi-Repository Dataset Compiler for VS#
// Compiles training data across all VerScript org repos and polyglot languages accessible via "inject"

const fs = require('fs');
const path = require('path');

const ORG_ROOT = path.resolve(__dirname, '..');
const OUTPUT_FILE = path.join(__dirname, 'knowledge', 'training_data.json');

const trainingData = [];

// Helper to add prompt-response pair
function addPair(prompt, response) {
    if (!prompt || !response) return;
    const cleanPrompt = prompt.trim();
    const cleanResponse = response.trim();
    if (cleanPrompt.length < 2 || cleanResponse.length < 5) return;
    trainingData.push({ prompt: cleanPrompt, response: cleanResponse });
}

// 1. CONVERSATIONAL & IDENTITY TRAINING
const conversationalPairs = [
    {
        p: "hello",
        r: "Hello! I am VS#, your neural assistant for VerScript and polyglot programming. I can generate, explain, and debug code in VerScript, Python, JavaScript, and C. How can I help you today?"
    },
    {
        p: "hi",
        r: "Hi there! I am VS#, the neural language model for VerScript. Whether you need pure VerScript code or polyglot code injected with Python, JavaScript, or C, I am ready to assist."
    },
    {
        p: "hey",
        r: "Hey! I am VS#, powered by a custom neural network. I can write VerScript scripts, debug syntax, or write multi-language code using the inject keyword."
    },
    {
        p: "who are you",
        r: "I am VS# (VS-Sharp), an autonomous AI coding companion and neural language engine designed specifically for VerScript and polyglot development."
    },
    {
        p: "what is vs#",
        r: "VS# is the official neural network language model and code intelligence engine for VerScript. It generates idiomatic VerScript, supports polyglot code injection (Python, JavaScript, C), and integrates directly with the IDE."
    },
    {
        p: "what languages can you code in",
        r: "I specialize in VerScript, as well as all languages supported through VerScript's 'inject' directive: Python, JavaScript, C/C++, HTML/CSS, and shell scripting."
    },
    {
        p: "how does the inject keyword work",
        r: "In VerScript, the 'inject' keyword allows embedding foreign programming languages directly into your script. For example:\n\n```verscript\ndisplay \"Running injected Python:\"\ninject python\n  for i in range(5):\n      print(f\"Python step: {i}\")\n```\n\nYou can inject python, javascript, c, or dynamically eval nested verscript with 'inject verscript'."
    },
    {
        p: "how to run python in verscript",
        r: "Use the 'inject python' block in VerScript:\n\n```verscript\ndisplay \"Executing Python snippet:\"\ninject python\n  import math\n  val = math.sqrt(144)\n  print(\"Square root is:\", val)\n```"
    },
    {
        p: "how to run javascript in verscript",
        r: "Use the 'inject javascript' block in VerScript:\n\n```verscript\ndisplay \"Executing JavaScript snippet:\"\ninject javascript\n  const items = [10, 20, 30];\n  const sum = items.reduce((a, b) => a + b, 0);\n  console.log(\"Sum:\", sum);\n```"
    },
    {
        p: "how to run c in verscript",
        r: "Use the 'inject c' block in VerScript:\n\n```verscript\ndisplay \"Executing native C code:\"\ninject c\n  #include <stdio.h>\n  int square(int n) { return n * n; }\n  printf(\"Square of 8 is: %d\\n\", square(8));\n```"
    },
    {
        p: "how to dynamically evaluate verscript",
        r: "Use 'inject verscript' to execute dynamic code blocks inside your script:\n\n```verscript\ndisplay \"Dynamic VerScript evaluation:\"\ninject verscript\n  iterate k from 1 to 3\n    display \"Nested iteration: \" + k\n```"
    }
];

conversationalPairs.forEach(({ p, r }) => addPair(p, r));

// 2. POLYGLOT "INJECT" CODE GENERATION EXAMPLES
const polyglotExamples = [
    // Python
    {
        p: "write python code to calculate factorial",
        r: "Here is a Python function to compute the factorial:\n\n```python\ndef factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)\n\nprint(\"Factorial of 5:\", factorial(5))\n```"
    },
    {
        p: "write verscript with injected python for math",
        r: "Here is a VerScript program with injected Python:\n\n```verscript\n! Polyglot Math Program\ndisplay \"Starting Python Math Execution...\" ?color=\"cyan\"\ninject python\n  import math\n  radius = 5.0\n  area = math.pi * (radius ** 2)\n  print(f\"Circle area with radius {radius}: {area:.2f}\")\ndisplay \"Python Math Finished.\" ?color=\"green\"\n```"
    },
    {
        p: "write python fibonacci generator",
        r: "Here is a Python script that computes Fibonacci numbers:\n\n```python\ndef fibonacci(count):\n    seq = [0, 1]\n    while len(seq) < count:\n        seq.append(seq[-1] + seq[-2])\n    return seq[:count]\n\nprint(\"First 10 Fibonacci numbers:\", fibonacci(10))\n```"
    },
    {
        p: "write python code for list comprehension",
        r: "Here is a Python example using list comprehensions and filters:\n\n```python\nnumbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]\nevens = [n for n in numbers if n % 2 == 0]\nsquares = [n ** 2 for n in evens]\nprint(\"Even squares:\", squares)\n```"
    },
    {
        p: "write verscript injecting python to read a file",
        r: "Here is a VerScript script utilizing injected Python to read files:\n\n```verscript\ndisplay \"Reading file with Python injector:\"\ninject python\n  with open('data.txt', 'w') as f:\n      f.write('Hello from injected Python file writer!')\n  with open('data.txt', 'r') as f:\n      print(\"File content:\", f.read())\n```"
    },

    // JavaScript
    {
        p: "write javascript async fetch function",
        r: "Here is a modern JavaScript async/await fetch utility:\n\n```javascript\nasync function fetchUserData(userId) {\n  try {\n    const response = await fetch(`https://api.example.com/users/${userId}`);\n    const data = await response.json();\n    console.log(\"User data:\", data);\n    return data;\n  } catch (err) {\n    console.error(\"Failed to fetch user:\", err.message);\n  }\n}\n```"
    },
    {
        p: "write verscript with injected javascript array processing",
        r: "Here is a VerScript script with injected JavaScript:\n\n```verscript\n! Polyglot JS Array Processing\ndisplay \"Processing arrays in JS...\" ?color=\"purple\"\ninject javascript\n  const scores = [85, 92, 78, 99, 88];\n  const topScores = scores.filter(s => s >= 90);\n  const average = scores.reduce((acc, val) => acc + val, 0) / scores.length;\n  console.log(\"Top scores (>=90):\", topScores);\n  console.log(\"Average score:\", average.toFixed(2));\ndisplay \"JS Processing Complete.\" ?color=\"green\"\n```"
    },
    {
        p: "write javascript express server route",
        r: "Here is an Express route handler in JavaScript:\n\n```javascript\nconst express = require('express');\nconst app = express();\napp.use(express.json());\n\napp.get('/api/status', (req, res) => {\n  res.json({ status: 'online', timestamp: Date.now() });\n});\n\napp.post('/api/compute', (req, res) => {\n  const { x, y } = req.body;\n  res.json({ result: Number(x) + Number(y) });\n});\n```"
    },
    {
        p: "write verscript with injected javascript dom manipulation",
        r: "Here is a VerScript code block injecting JavaScript for frontend interaction:\n\n```verscript\ndisplay \"Injecting Web JavaScript:\"\ninject javascript\n  const btn = document.getElementById('run-btn');\n  if (btn) {\n    btn.addEventListener('click', () => {\n      console.log('Button clicked via injected script');\n    });\n  }\n```"
    },

    // C / C++
    {
        p: "write c code for binary search",
        r: "Here is an implementation of binary search in C:\n\n```c\n#include <stdio.h>\n\nint binarySearch(int arr[], int size, int target) {\n    int low = 0, high = size - 1;\n    while (low <= high) {\n        int mid = low + (high - low) / 2;\n        if (arr[mid] == target) return mid;\n        if (arr[mid] < target) low = mid + 1;\n        else high = mid - 1;\n    }\n    return -1;\n}\n\nint main() {\n    int data[] = { 2, 5, 8, 12, 16, 23, 38, 56 };\n    int idx = binarySearch(data, 8, 16);\n    printf(\"Found target 16 at index: %d\\n\", idx);\n    return 0;\n}\n```"
    },
    {
        p: "write verscript with injected c for high speed calculation",
        r: "Here is a VerScript program injecting native C:\n\n```verscript\n! Native C Compute in VerScript\ndisplay \"Running low-level C algorithm...\" ?color=\"cyan\"\ninject c\n  #include <stdio.h>\n  long long sumOfSquares(int n) {\n      long long sum = 0;\n      for (int i = 1; i <= n; i++) sum += (long long)i * i;\n      return sum;\n  }\n  printf(\"Sum of squares up to 1000: %lld\\n\", sumOfSquares(1000));\ndisplay \"Native execution complete.\" ?color=\"green\"\n```"
    },
    {
        p: "write c struct and linked list",
        r: "Here is a basic singly-linked list structure in C:\n\n```c\n#include <stdio.h>\n#include <stdlib.h>\n\ntypedef struct Node {\n    int val;\n    struct Node *next;\n} Node;\n\nNode* createNode(int v) {\n    Node *n = (Node*)malloc(sizeof(Node));\n    n->val = v;\n    n->next = NULL;\n    return n;\n}\n```"
    },

    // HTML / CSS
    {
        p: "write html and css card component",
        r: "Here is a responsive modern card component in HTML and CSS:\n\n```html\n<div class=\"card\">\n  <h3 class=\"card-title\">VerScript Polyglot</h3>\n  <p class=\"card-desc\">Seamlessly run Python, JavaScript, and C inside VerScript scripts.</p>\n  <button class=\"card-btn\">Explore Docs</button>\n</div>\n\n<style>\n.card {\n  background: #1e1e2e;\n  color: #cdd6f4;\n  padding: 1.5rem;\n  border-radius: 12px;\n  box-shadow: 0 4px 12px rgba(0,0,0,0.3);\n  border: 1px solid #313244;\n}\n.card-title { color: #89b4fa; margin-top: 0; }\n.card-btn {\n  background: #a6e3a1;\n  color: #11111b;\n  border: none;\n  padding: 0.5rem 1rem;\n  border-radius: 6px;\n  font-weight: bold;\n  cursor: pointer;\n}\n</style>\n```"
    }
];

polyglotExamples.forEach(({ p, r }) => addPair(p, r));

// 3. CORE VERSCRIPT SYNTAX & PATTERNS
const verscriptSyntaxPairs = [
    {
        p: "write a hello world program in verscript",
        r: "Here is the standard Hello World script in VerScript:\n\n```verscript\n! Hello World in VerScript\ndisplay \"Hello World from VerScript!\" ?color=\"cyan\"\n```"
    },
    {
        p: "how to declare variables in verscript",
        r: "In VerScript, variables are declared and assigned using the colon ':' operator:\n\n```verscript\nname : \"VerScript Developer\"\nage : 25\nis_active : true\nscores : [95, 88, 100]\ndisplay name\ndisplay age\n```"
    },
    {
        p: "how to write loops in verscript",
        r: "VerScript provides both 'iterate' and 'loop' constructs:\n\n```verscript\n! Iterate from range\niterate i from 1 to 5\n  display \"Step: \" + i\n\n! Loop fixed times\nloop 3 times\n  display \"Repeating action!\"\n```"
    },
    {
        p: "how to write conditional if else in verscript",
        r: "Here is how to write conditional statements in VerScript:\n\n```verscript\nscore : 85\nif score >= 90\n  display \"Grade: A\"\nelse\n  display \"Grade: B or lower\"\n```"
    },
    {
        p: "how to define functions in verscript",
        r: "Use 'def' to define functions and 'reply' to return values in VerScript:\n\n```verscript\ndef add(a, b)\n  result : a + b\n  reply result\n\nsum : add(15, 27)\ndisplay \"The sum is: \" + sum\n```"
    },
    {
        p: "how to prompt user for input in verscript",
        r: "Use the 'prompt' statement to take console input from the user:\n\n```verscript\ndisplay \"Please enter your name:\"\nprompt username\ndisplay \"Welcome, \" + username + \"!\"\n```"
    },
    {
        p: "how to delete variables with del in verscript",
        r: "Use the 'del' statement to delete variables or clear memory:\n\n```verscript\ntemp_val : 999\ndisplay \"Temp value: \" + temp_val\ndel temp_val\n! temp_val is now removed from the scope\n```"
    },
    {
        p: "how to use classes and entities in verscript",
        r: "In VerScript, objects and data models can be defined with 'entity' or 'class':\n\n```verscript\nclass Person\n  def init(name, role)\n    this.name : name\n    this.role : role\n  def greet()\n    display \"Hello, I am \" + this.name + \" (\" + this.role + \")\"\n\nuser : inst Person(\"Alex\", \"Engineer\")\nuser.greet()\n```"
    },
    {
        p: "how to import standard libraries in verscript",
        r: "Use 'import' to include standard library modules:\n\n```verscript\nimport Math\nimport Strings\n\nangle : 45\nsin_val : Math.sin(angle)\ndisplay \"Sine of 45: \" + sin_val\n\nmsg : \"verscript\"\nupper_msg : Strings.upper(msg)\ndisplay upper_msg\n```"
    }
];

verscriptSyntaxPairs.forEach(({ p, r }) => addPair(p, r));

// 4. INGEST FILES FROM ALL 5 VERSCRIPT REPOSITORIES
console.log("[Dataset] Ingesting files across VerScript organization repositories...");

const reposToScan = [
    { name: 'VerScript', path: path.join(ORG_ROOT, 'VerScript') },
    { name: 'VerScript.github.io', path: path.join(ORG_ROOT, 'VerScript.github.io') },
    { name: 'IDE', path: path.join(ORG_ROOT, 'IDE') },
    { name: 'polyserver', path: path.join(ORG_ROOT, 'polyserver') },
    { name: 'VS-Sharp', path: path.join(ORG_ROOT, 'VS-Sharp') }
];

function scanDirectory(dir, repoName) {
    if (!fs.existsSync(dir)) return;
    const entries = fs.readdirSync(dir, { withFileTypes: true });

    for (const entry of entries) {
        const fullPath = path.join(dir, entry.name);
        if (entry.name === '.git' || entry.name === 'node_modules' || entry.name === 'dist' || entry.name === 'build') continue;

        if (entry.isDirectory()) {
            scanDirectory(fullPath, repoName);
        } else if (entry.isFile()) {
            const ext = path.extname(entry.name).toLowerCase();
            const relPath = path.relative(ORG_ROOT, fullPath);

            // VRS Standard Library & Test files
            if (ext === '.vrs') {
                try {
                    const content = fs.readFileSync(fullPath, 'utf8').trim();
                    if (content.length > 20 && content.length < 5000) {
                        const baseName = path.basename(entry.name, '.vrs');
                        addPair(`write ${baseName} code in verscript`, `### VerScript ${baseName} Implementation\n\n\`\`\`verscript\n${content}\n\`\`\``);
                        addPair(`show code for ${entry.name}`, `### Script: ${entry.name}\n\n\`\`\`verscript\n${content}\n\`\`\``);
                        addPair(`explain ${entry.name}`, `### Walkthrough of ${entry.name}\n\nThis script demonstrates VerScript functionality:\n\n\`\`\`verscript\n${content}\n\`\`\``);
                    }
                } catch (_) {}
            }

            // JavaScript Source Files
            else if (ext === '.js' && !entry.name.includes('test') && !entry.name.includes('config')) {
                try {
                    const content = fs.readFileSync(fullPath, 'utf8').trim();
                    if (content.length > 50 && content.length < 3500) {
                        const baseName = path.basename(entry.name, '.js');
                        addPair(`write javascript code for ${baseName}`, `### JavaScript Module: ${baseName}\n\n\`\`\`javascript\n${content}\n\`\`\``);
                        addPair(`how does ${entry.name} work in ${repoName}`, `### ${entry.name} in ${repoName}\n\nHere is the implementation:\n\n\`\`\`javascript\n${content}\n\`\`\``);
                    }
                } catch (_) {}
            }

            // C Source & Header Files
            else if ((ext === '.c' || ext === '.h') && entry.name !== 'sqlite3.c') {
                try {
                    const content = fs.readFileSync(fullPath, 'utf8').trim();
                    if (content.length > 50 && content.length < 3500) {
                        const baseName = path.basename(entry.name, ext);
                        addPair(`show c code for ${baseName}`, `### C Implementation: ${entry.name}\n\n\`\`\`c\n${content}\n\`\`\``);
                    }
                } catch (_) {}
            }

            // Markdown documentation files
            else if (ext === '.md' && entry.name !== 'package-lock.json') {
                try {
                    const content = fs.readFileSync(fullPath, 'utf8').trim();
                    if (content.length > 50 && content.length < 2500) {
                        const baseName = path.basename(entry.name, '.md');
                        addPair(`explain ${baseName} in ${repoName}`, `### Documentation for ${baseName}\n\n${content}`);
                    }
                } catch (_) {}
            }
        }
    }
}

reposToScan.forEach(repo => scanDirectory(repo.path, repo.name));

// 5. INGEST ARTICLES FROM docs/articles_data.js IF AVAILABLE
const articlesFile = path.join(ORG_ROOT, 'VerScript.github.io', 'docs', 'articles_data.js');
if (fs.existsSync(articlesFile)) {
    try {
        const fileContent = fs.readFileSync(articlesFile, 'utf8');
        // Extract articles objects roughly
        const regex = /title:\s*["'`](.*?)["'`][\s\S]*?content:\s*["'`]([\s\S]*?)["'`]\s*[,}]/g;
        let match;
        while ((match = regex.exec(fileContent)) !== null) {
            const title = match[1].trim();
            const rawBody = match[2].trim().replace(/\\n/g, '\n').replace(/\\"/g, '"');
            if (title && rawBody.length > 40) {
                addPair(`explain ${title} in verscript`, `### ${title}\n\n${rawBody.substring(0, 1000)}`);
                addPair(`what is ${title}`, `### ${title}\n\n${rawBody.substring(0, 1000)}`);
            }
        }
    } catch (_) {}
}

// 6. SAVE DATASET
fs.mkdirSync(path.dirname(OUTPUT_FILE), { recursive: true });
fs.writeFileSync(OUTPUT_FILE, JSON.stringify(trainingData, null, 2), 'utf8');

console.log(`[Dataset] Successfully compiled ${trainingData.length} training examples across all repos into:`);
console.log(`          ${OUTPUT_FILE}`);
