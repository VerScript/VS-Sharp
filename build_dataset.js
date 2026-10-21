// build_dataset.js — Polyglot & Multi-Repository Dataset Compiler for VS#
// Compiles dense, multi-thousand high quality training pairs across all VerScript org repos and polyglot languages

const fs = require('fs');
const path = require('path');
const vm = require('vm');

const ORG_ROOT = path.resolve(__dirname, '..');
const OUTPUT_FILE = path.join(__dirname, 'knowledge', 'training_data.json');

const trainingData = [];
const seenPrompts = new Set();

function addPair(prompt, response, variations = []) {
    if (!prompt || !response) return;
    const cleanPrompt = prompt.trim();
    const cleanResponse = response.trim();
    const key = cleanPrompt.toLowerCase();
    if (cleanPrompt.length < 2 || cleanResponse.length < 5) return;

    if (!seenPrompts.has(key)) {
        seenPrompts.add(key);
        trainingData.push({ prompt: cleanPrompt, response: cleanResponse });
    }

    for (const v of variations) {
        const vClean = v.trim();
        const vKey = vClean.toLowerCase();
        if (vClean.length >= 2 && !seenPrompts.has(vKey)) {
            seenPrompts.add(vKey);
            trainingData.push({ prompt: vClean, response: cleanResponse });
        }
    }
}

// ============================================================================
// 1. CONVERSATIONAL & REASONING DIALOGUES (FLUENT ENGLISH, NO GARBLE)
// ============================================================================
const conversationalDialogues = [
    {
        p: "hello",
        vars: ["hi", "hey", "hello there", "greetings", "good morning", "good evening", "hi vs#"],
        r: "Hello! I am VS#, your AI coding companion and neural engine for VerScript and polyglot development. I can generate, explain, and debug code in VerScript, Python, JavaScript, and C. How can I help you today?"
    },
    {
        p: "who are you",
        vars: ["what is vs#", "introduce yourself", "tell me about yourself", "what model are you", "what is your name"],
        r: "I am VS# (VS-Sharp), an autonomous AI coding assistant powered by a custom 15-million parameter neural language model. I am designed specifically for the VerScript programming ecosystem and foreign code injection via the 'inject' keyword."
    },
    {
        p: "how are you",
        vars: ["how are you doing", "are you ready", "are you operational", "system check"],
        r: "I am running at optimal performance! I am ready to generate VerScript programs, embed Python, JavaScript, or C snippets, analyze syntax, or explain code line-by-line."
    },
    {
        p: "what can you do",
        vars: ["help", "what are your capabilities", "what features do you support", "how can you assist me"],
        r: "I can assist you with:\n1. Generating complete VerScript scripts (loops, functions, classes, stdlib).\n2. Embedding foreign code using 'inject python', 'inject javascript', and 'inject c'.\n3. Diagnosing syntax errors and repairing assignment operators (changing '=' to ':').\n4. Providing step-by-step code walkthroughs and explanations."
    },
    {
        p: "what is verscript",
        vars: ["explain verscript", "tell me about verscript", "what kind of language is verscript"],
        r: "VerScript is a high-performance, expressive programming language combining the speed of native C with the clean, readable ergonomics of Python. It features clean indentation, no semicolon requirements, first-class polyglot injection, and a robust standard library."
    },
    {
        p: "how does the inject keyword work",
        vars: ["explain inject keyword", "what is inject in verscript", "how to inject code in verscript", "polyglot injection in verscript"],
        r: "The 'inject' directive allows embedding foreign programming languages directly into your VerScript file:\n\n```verscript\ndisplay \"Running injected Python:\"\ninject python\n  for i in range(5):\n      print(f\"Step {i}\")\n```\n\nYou can inject 'python', 'javascript', 'c', or dynamically evaluate nested VerScript with 'inject verscript'."
    },
    {
        p: "thank you",
        vars: ["thanks", "thanks a lot", "thank you so much", "awesome thank you", "appreciate it"],
        r: "You're very welcome! Let me know if you need any more code, syntax help, or polyglot scripts. Happy coding!"
    }
];

conversationalDialogues.forEach(item => addPair(item.p, item.r, item.vars));

// ============================================================================
// 2. CORE VERSCRIPT SYNTAX & LANGUAGE SPECIFICATIONS
// ============================================================================
const syntaxItems = [
    {
        p: "write a hello world program in verscript",
        vars: ["hello world in verscript", "how to print hello world in verscript", "verscript hello world example"],
        r: "Here is the standard Hello World script in VerScript:\n\n```verscript\n! Hello World in VerScript\ndisplay \"Hello World from VerScript!\" ?color=\"cyan\"\n```\n\n- `display` prints text to standard output.\n- `?color=\"cyan\"` adds ANSI color formatting."
    },
    {
        p: "how to declare variables in verscript",
        vars: ["variable declaration in verscript", "assigning variables in verscript", "how to assign variables in verscript"],
        r: "In VerScript, variables are assigned using the colon ':' operator:\n\n```verscript\nusername : \"VerScript Developer\"\nage : 24\nis_online : true\nscores : [92, 88, 95]\ndisplay \"User: \" + username\ndisplay \"Age: \" + age\n```"
    },
    {
        p: "how to write loops in verscript",
        vars: ["loops in verscript", "iterate in verscript", "for loop in verscript", "while loop in verscript"],
        r: "VerScript offers several loop constructs:\n\n```verscript\n! 1. Iterate over a numeric range\niterate i from 1 to 5\n  display \"Iterate step: \" + i\n\n! 2. Fixed count loop\nloop 3 times\n  display \"Looping repeating action!\"\n\n! 3. While conditional loop\ncounter : 0\nwhile counter < 3\n  display \"Counter: \" + counter\n  counter : counter + 1\n```"
    },
    {
        p: "how to write conditional if else in verscript",
        vars: ["if else in verscript", "conditionals in verscript", "branching in verscript"],
        r: "Here is how to write conditional statements in VerScript:\n\n```verscript\nscore : 88\nif score >= 90\n  display \"Grade: A (Distinction)\" ?color=\"green\"\nelse if score >= 75\n  display \"Grade: B (Proficient)\" ?color=\"yellow\"\nelse\n  display \"Grade: C (Needs Review)\" ?color=\"red\"\n```"
    },
    {
        p: "how to define functions in verscript",
        vars: ["functions in verscript", "def statement in verscript", "return from function in verscript"],
        r: "Functions are defined with 'def' and return values with 'reply':\n\n```verscript\ndef calculate_area(width, height)\n  area : width * height\n  reply area\n\nresult : calculate_area(12, 5)\ndisplay \"Area calculation: \" + result\n```"
    },
    {
        p: "how to take user input in verscript",
        vars: ["prompt in verscript", "input in verscript", "how to ask user for input in verscript"],
        r: "Use the 'prompt' statement to take console input from the user:\n\n```verscript\ndisplay \"Please enter your name:\"\nprompt user_name\ndisplay \"Welcome to VerScript, \" + user_name + \"!\"\n```"
    },
    {
        p: "how to create a class and instance in verscript",
        vars: ["classes in verscript", "oop in verscript", "inst keyword in verscript", "objects in verscript"],
        r: "Define classes using 'class' and instantiate them with 'inst':\n\n```verscript\nclass Account\n  def init(owner, balance)\n    this.owner : owner\n    this.balance : balance\n  def deposit(amount)\n    this.balance : this.balance + amount\n    display \"Deposited: \" + amount + \" | Balance: \" + this.balance\n\nacc : inst Account(\"Alex\", 1000)\nacc.deposit(250)\n```"
    },
    {
        p: "how to delete variables with del in verscript",
        vars: ["del in verscript", "delete variable in verscript", "free variable in verscript"],
        r: "Use 'del' to remove variables from memory scope:\n\n```verscript\ntemp_val : 9999\ndisplay \"Temp value: \" + temp_val\ndel temp_val\n! temp_val is now removed from memory\n```"
    },
    {
        p: "how to use pattern matching with match and with in verscript",
        vars: ["match with in verscript", "switch case in verscript", "pattern matching in verscript"],
        r: "Pattern matching in VerScript uses 'match' and 'with':\n\n```verscript\ncode : 200\nmatch code\n  with 200\n    display \"Status: OK (200)\"\n  with 404\n    display \"Status: Not Found (404)\"\n  with 500\n    display \"Status: Server Error (500)\"\n  with _\n    display \"Unknown Status\"\n```"
    }
];

syntaxItems.forEach(item => addPair(item.p, item.r, item.vars));

// ============================================================================
// 3. POLYGLOT "INJECT" COMPREHENSIVE ALGORITHMIC & DEV SUITES
// ============================================================================

// --- PYTHON INJECTION SUITE ---
const pythonExamples = [
    {
        p: "how to run python in verscript",
        vars: ["inject python in verscript", "run python script in verscript"],
        r: "Use 'inject python' to execute native Python blocks inside VerScript:\n\n```verscript\ndisplay \"Starting Python compute:\" ?color=\"cyan\"\ninject python\n  import math\n  val = 144\n  root = math.sqrt(val)\n  print(f\"Square root of {val}: {root}\")\ndisplay \"Python block completed.\" ?color=\"green\"\n```"
    },
    {
        p: "write python code for binary search",
        vars: ["binary search in python", "how to write binary search in python"],
        r: "Here is an implementation of binary search in Python:\n\n```python\ndef binary_search(arr, target):\n    low, high = 0, len(arr) - 1\n    while low <= high:\n        mid = (low + high) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            low = mid + 1\n        else:\n            high = mid - 1\n    return -1\n\nitems = [2, 5, 8, 12, 16, 23, 38, 56]\nidx = binary_search(items, 23)\nprint(\"Target found at index:\", idx)\n```"
    },
    {
        p: "write python code for factorial",
        vars: ["factorial in python", "how to calculate factorial in python"],
        r: "Here is a Python factorial function:\n\n```python\ndef factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)\n\nprint(\"Factorial of 5:\", factorial(5))\n```"
    },
    {
        p: "write python fibonacci generator",
        vars: ["fibonacci in python", "how to generate fibonacci numbers in python"],
        r: "Here is a Python function generating Fibonacci sequences:\n\n```python\ndef fibonacci(n):\n    seq = [0, 1]\n    while len(seq) < n:\n        seq.append(seq[-1] + seq[-2])\n    return seq[:n]\n\nprint(\"First 8 Fibonacci numbers:\", fibonacci(8))\n```"
    },
    {
        p: "write python quicksort algorithm",
        vars: ["quicksort in python", "how to write quicksort in python"],
        r: "Here is quicksort implemented in Python:\n\n```python\ndef quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]\n    left = [x for x in arr if x < pivot]\n    mid = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return quicksort(left) + mid + quicksort(right)\n\nprint(\"Sorted:\", quicksort([38, 27, 43, 3, 9, 82, 10]))\n```"
    },
    {
        p: "write python prime checker",
        vars: ["is prime in python", "how to check if number is prime in python"],
        r: "Here is a Python function to check for prime numbers:\n\n```python\ndef is_prime(n):\n    if n < 2:\n        return False\n    for i in range(2, int(n**0.5) + 1):\n        if n % i == 0:\n            return False\n    return True\n\nprimes = [x for x in range(2, 30) if is_prime(x)]\nprint(\"Primes under 30:\", primes)\n```"
    }
];

pythonExamples.forEach(item => addPair(item.p, item.r, item.vars));

// --- JAVASCRIPT INJECTION SUITE ---
const jsExamples = [
    {
        p: "how to run javascript in verscript",
        vars: ["inject javascript in verscript", "run js in verscript"],
        r: "Use 'inject javascript' to execute JavaScript code inside VerScript:\n\n```verscript\ndisplay \"Running injected JavaScript:\" ?color=\"purple\"\ninject javascript\n  const items = [10, 20, 30, 40];\n  const sum = items.reduce((a, b) => a + b, 0);\n  console.log(\"Total sum from JS:\", sum);\ndisplay \"JS finished.\" ?color=\"green\"\n```"
    },
    {
        p: "write javascript async fetch function",
        vars: ["async fetch in javascript", "fetch api in javascript", "how to fetch data in javascript"],
        r: "Here is an async/await fetch helper in JavaScript:\n\n```javascript\nasync function fetchUserData(userId) {\n  try {\n    const response = await fetch(`https://api.example.com/users/${userId}`);\n    if (!response.ok) throw new Error(`HTTP error ${response.status}`);\n    const data = await response.json();\n    console.log(\"User data:\", data);\n    return data;\n  } catch (err) {\n    console.error(\"Fetch failed:\", err.message);\n  }\n}\n```"
    },
    {
        p: "write javascript express server route",
        vars: ["express server in javascript", "express route in js"],
        r: "Here is an Express server route in JavaScript:\n\n```javascript\nconst express = require('express');\nconst app = express();\napp.use(express.json());\n\napp.get('/api/status', (req, res) => {\n  res.json({ status: 'online', timestamp: Date.now() });\n});\n\napp.post('/api/compute', (req, res) => {\n  const { val1, val2 } = req.body;\n  res.json({ result: Number(val1) + Number(val2) });\n});\n```"
    },
    {
        p: "write javascript array filter and map",
        vars: ["filter and map in javascript", "array methods in javascript"],
        r: "Here is an example using JavaScript `filter` and `map`:\n\n```javascript\nconst products = [\n  { name: \"Notebook\", price: 15, inStock: true },\n  { name: \"Keyboard\", price: 65, inStock: false },\n  { name: \"Monitor\", price: 200, inStock: true }\n];\n\nconst available = products\n  .filter(p => p.inStock)\n  .map(p => `${p.name} ($${p.price})`);\n\nconsole.log(\"Available items:\", available);\n```"
    }
];

jsExamples.forEach(item => addPair(item.p, item.r, item.vars));

// --- C INJECTION SUITE ---
const cExamples = [
    {
        p: "how to run c in verscript",
        vars: ["inject c in verscript", "run c code in verscript"],
        r: "Use 'inject c' to execute high-speed native C code in VerScript:\n\n```verscript\ndisplay \"Running low-level C algorithm:\" ?color=\"cyan\"\ninject c\n  #include <stdio.h>\n  int square(int n) { return n * n; }\n  printf(\"Square of 9 is: %d\\n\", square(9));\ndisplay \"Native C execution finished.\" ?color=\"green\"\n```"
    },
    {
        p: "write c code for binary search",
        vars: ["binary search in c", "c binary search implementation"],
        r: "Here is binary search implemented in C:\n\n```c\n#include <stdio.h>\n\nint binary_search(int arr[], int size, int target) {\n    int low = 0, high = size - 1;\n    while (low <= high) {\n        int mid = low + (high - low) / 2;\n        if (arr[mid] == target) return mid;\n        if (arr[mid] < target) low = mid + 1;\n        else high = mid - 1;\n    }\n    return -1;\n}\n\nint main() {\n    int data[] = { 3, 7, 12, 19, 25, 33, 42 };\n    int idx = binary_search(data, 7, 25);\n    printf(\"Found target 25 at index: %d\\n\", idx);\n    return 0;\n}\n```"
    },
    {
        p: "write c code for linked list",
        vars: ["linked list in c", "c struct linked list"],
        r: "Here is a singly-linked list in C:\n\n```c\n#include <stdio.h>\n#include <stdlib.h>\n\ntypedef struct Node {\n    int val;\n    struct Node* next;\n} Node;\n\nNode* create_node(int v) {\n    Node* n = (Node*)malloc(sizeof(Node));\n    n->val = v;\n    n->next = NULL;\n    return n;\n}\n```"
    }
];

cExamples.forEach(item => addPair(item.p, item.r, item.vars));

// ============================================================================
// 4. INGEST OFFICIAL ACADEMY CURRICULUM (69 CHAPTERS + CODE BLOCKS + EXERCISES)
// ============================================================================
console.log("[Dataset] Ingesting VerScript Academy official curriculum...");
const articlesPath = path.join(ORG_ROOT, 'VerScript.github.io', 'docs', 'articles_data.js');
if (fs.existsSync(articlesPath)) {
    try {
        const fileContent = fs.readFileSync(articlesPath, 'utf8');
        const sandbox = {};
        vm.createContext(sandbox);
        vm.runInContext(fileContent, sandbox);

        if (Array.isArray(sandbox.ARTICLES)) {
            for (const article of sandbox.ARTICLES) {
                // Chapter summary
                addPair(
                    `explain ${article.title} in verscript`,
                    `### ${article.title}\n\n${article.summary}`,
                    [`what is ${article.title}`, `learn ${article.title}`]
                );

                // Code Blocks
                if (Array.isArray(article.codeBlocks)) {
                    for (const cb of article.codeBlocks) {
                        addPair(
                            `show example of ${cb.title}`,
                            `### ${cb.title}\n\n\`\`\`verscript\n${cb.code}\n\`\`\``,
                            [`code example for ${article.title}`]
                        );
                    }
                }

                // Exercises & Solutions
                if (Array.isArray(article.exercises)) {
                    for (const ex of article.exercises) {
                        addPair(
                            ex.prompt.replace(/<[^>]+>/g, ''),
                            `### Solution: ${ex.title}\n\n\`\`\`verscript\n${ex.solution}\n\`\`\``,
                            [`how to solve ${ex.title}`, `solution for ${ex.title}`]
                        );
                    }
                }
            }
        }
    } catch (e) {
        console.warn("[Dataset] Warning loading articles_data.js:", e.message);
    }
}

// ============================================================================
// 5. INGEST ALL STDLIB FUNCTIONS (MATH, STRINGS, ARRAYS, MATRIX, VECTOR, ETC.)
// ============================================================================
console.log("[Dataset] Ingesting VerScript Standard Libraries & Functions...");
const stdlibDir = path.join(ORG_ROOT, 'VerScript', 'stdlib');
if (fs.existsSync(stdlibDir)) {
    for (const f of fs.readdirSync(stdlibDir)) {
        if (!f.endsWith('.vrs')) continue;
        const libName = path.basename(f, '.lib.vrs');
        const content = fs.readFileSync(path.join(stdlibDir, f), 'utf8');
        const lines = content.split('\n');

        let currentFunc = null;
        let currentBody = [];

        for (const line of lines) {
            const m = line.match(/^\s*def\s+(?:func\s+)?([a-zA-Z0-9_]+)(.*)/);
            if (m) {
                if (currentFunc && currentBody.length > 0) {
                    const funcCode = currentBody.join('\n').trim();
                    addPair(
                        `how to use ${libName}.${currentFunc} in verscript`,
                        `Here is the \`${libName}.${currentFunc}\` implementation:\n\n\`\`\`verscript\n${funcCode}\n\`\`\``,
                        [`show ${libName}.${currentFunc} function`, `write code for ${libName}.${currentFunc}`]
                    );
                }
                currentFunc = m[1];
                currentBody = [line];
            } else if (currentFunc) {
                if (line.match(/^\s*(?:lib|dynamic:|const:)/)) {
                    const funcCode = currentBody.join('\n').trim();
                    addPair(
                        `how to use ${libName}.${currentFunc} in verscript`,
                        `Here is the \`${libName}.${currentFunc}\` implementation:\n\n\`\`\`verscript\n${funcCode}\n\`\`\``
                    );
                    currentFunc = null;
                    currentBody = [];
                } else {
                    currentBody.push(line);
                }
            }
        }
        if (currentFunc && currentBody.length > 0) {
            const funcCode = currentBody.join('\n').trim();
            addPair(
                `how to use ${libName}.${currentFunc} in verscript`,
                `Here is the \`${libName}.${currentFunc}\` implementation:\n\n\`\`\`verscript\n${funcCode}\n\`\`\``
            );
        }

        // Full library
        if (content.length > 50 && content.length < 3000) {
            addPair(
                `show full ${libName} standard library code in verscript`,
                `### ${libName} Standard Library\n\n\`\`\`verscript\n${content}\n\`\`\``,
                [`how to import ${libName} in verscript`]
            );
        }
    }
}

// ============================================================================
// 6. INGEST ALL TESTS & SOURCE CODE ACROSS ALL REPOS
// ============================================================================
console.log("[Dataset] Ingesting test suites and repository source files...");
const repos = [
    { name: 'VerScript', path: path.join(ORG_ROOT, 'VerScript') },
    { name: 'VerScript.github.io', path: path.join(ORG_ROOT, 'VerScript.github.io') },
    { name: 'IDE', path: path.join(ORG_ROOT, 'IDE') },
    { name: 'polyserver', path: path.join(ORG_ROOT, 'polyserver') },
    { name: 'VS-Sharp', path: path.join(ORG_ROOT, 'VS-Sharp') },
    { name: 'verscript-studio-android', path: path.join(ORG_ROOT, 'verscript-studio-android') }
];

function scanFiles(dir, repoName) {
    if (!fs.existsSync(dir)) return;
    const entries = fs.readdirSync(dir, { withFileTypes: true });
    for (const e of entries) {
        const full = path.join(dir, e.name);
        if (e.name === '.git' || e.name === 'node_modules' || e.name === 'dist' || e.name === 'build' || e.name === 'android') continue;

        if (e.isDirectory()) {
            scanFiles(full, repoName);
        } else if (e.isFile()) {
            const ext = path.extname(e.name).toLowerCase();
            const base = path.basename(e.name, ext);

            if (ext === '.vrs' && !dir.includes('stdlib')) {
                try {
                    const content = fs.readFileSync(full, 'utf8').trim();
                    if (content.length > 20 && content.length < 2500) {
                        addPair(
                            `show ${base} test script in verscript`,
                            `### Test Script: ${e.name}\n\n\`\`\`verscript\n${content}\n\`\`\``,
                            [`how to test ${base} in verscript`]
                        );
                    }
                } catch (_) {}
            } else if (ext === '.js' && !e.name.includes('config') && !e.name.includes('test')) {
                try {
                    const content = fs.readFileSync(full, 'utf8').trim();
                    if (content.length > 50 && content.length < 2000) {
                        addPair(
                            `show javascript code for ${base} in ${repoName}`,
                            `### ${e.name} (${repoName})\n\n\`\`\`javascript\n${content}\n\`\`\``
                        );
                    }
                } catch (_) {}
            } else if ((ext === '.c' || ext === '.h') && e.name !== 'sqlite3.c') {
                try {
                    const content = fs.readFileSync(full, 'utf8').trim();
                    if (content.length > 50 && content.length < 2000) {
                        addPair(
                            `show c source code for ${e.name}`,
                            `### ${e.name}\n\n\`\`\`c\n${content}\n\`\`\``
                        );
                    }
                } catch (_) {}
            }
        }
    }
}

repos.forEach(r => scanFiles(r.path, r.name));

// ============================================================================
// 7. SAVE MASTER DATASET
// ============================================================================
fs.mkdirSync(path.dirname(OUTPUT_FILE), { recursive: true });
fs.writeFileSync(OUTPUT_FILE, JSON.stringify(trainingData, null, 2), 'utf8');

console.log("==================================================================");
console.log(`✅ Successfully compiled ${trainingData.length.toLocaleString()} rich training pairs!`);
console.log(`📁 Saved to: ${OUTPUT_FILE}`);
console.log("==================================================================");
