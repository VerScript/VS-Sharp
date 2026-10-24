// build_dataset.js — Master Polyglot & Multi-Repository Dataset Compiler for VS#
// Compiles dense, multi-thousand high quality training pairs across all VerScript org repos,
// synthetic algorithmic generators, polyglot injections, code repairs, and curriculum.

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
        vars: ["hi", "hey", "hello there", "greetings", "good morning", "good evening", "hi vs#", "hey vs#", "yo"],
        r: "Hello! I am VS#, your autonomous AI coding companion and neural engine for VerScript and polyglot development. I can generate, explain, debug, and translate code in VerScript, Python, JavaScript, and C. How can I help you today?"
    },
    {
        p: "who are you",
        vars: ["what is vs#", "introduce yourself", "tell me about yourself", "what model are you", "what is your name", "who made you"],
        r: "I am VS# (VS-Sharp), an AI programming model powered by a custom 15-million parameter neural architecture. I specialize in the VerScript programming ecosystem, standard libraries, and polyglot execution using the 'inject' directive."
    },
    {
        p: "how are you",
        vars: ["how are you doing", "are you ready", "are you operational", "system check", "status check"],
        r: "I am running at optimal performance! I am ready to generate VerScript programs, embed foreign languages, diagnose syntax pitfalls, or explain code line-by-line."
    },
    {
        p: "what can you do",
        vars: ["help", "what are your capabilities", "what features do you support", "how can you assist me", "show capabilities"],
        r: "I can assist you with:\n1. Generating complete VerScript programs (loops, functions, classes, stdlib).\n2. Embedding foreign code using 'inject python', 'inject javascript', and 'inject c'.\n3. Diagnosing syntax errors and repairing common mistakes (like changing '=' to ':').\n4. Translating code between Python, JavaScript, C, and VerScript.\n5. Step-by-step walkthroughs and code explanations."
    },
    {
        p: "what is verscript",
        vars: ["explain verscript", "tell me about verscript", "what kind of language is verscript", "about verscript"],
        r: "VerScript is a high-performance, expressive programming language combining the speed of native C with clean, readable ergonomics. It features colon assignment (':'), clean indentation, first-class polyglot injection, and a robust standard library for math, strings, matrices, system tasks, and networking."
    },
    {
        p: "how does the inject keyword work",
        vars: ["explain inject keyword", "what is inject in verscript", "how to inject code in verscript", "polyglot injection in verscript", "inject directive"],
        r: "The 'inject' directive allows embedding foreign programming languages directly into your VerScript file:\n\n```verscript\ndisplay \"Running injected Python:\"\ninject python\n  for i in range(5):\n      print(f\"Step {i}\")\n```\n\nYou can inject 'python', 'javascript', 'c', or dynamically evaluate nested VerScript with 'inject verscript'."
    },
    {
        p: "thank you",
        vars: ["thanks", "thanks a lot", "thank you so much", "awesome thank you", "appreciate it", "great job"],
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
        vars: ["variable declaration in verscript", "assigning variables in verscript", "how to assign variables in verscript", "variables in verscript"],
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
        vars: ["functions in verscript", "def statement in verscript", "return from function in verscript", "how to return values in verscript"],
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
// 3. CODE REPAIR & SYNTAX DIAGNOSTICS (TEACHES DEBUGGING & ERROR AVOIDANCE)
// ============================================================================
const codeRepairs = [
    {
        errorSnippet: "count = 10",
        p: "how to fix count = 10 in verscript",
        vars: ["syntax error count = 10", "error invalid assignment = in verscript", "why does count = 10 fail in verscript", "fix assignment syntax in verscript"],
        r: "In VerScript, variable assignment uses the colon `:` operator rather than `=`. The `=` symbol is reserved for equality comparisons or default parameters.\n\n```verscript\n! Incorrect:\n! count = 10\n\n! Corrected:\ncount : 10\ndisplay \"Count is: \" + count\n```"
    },
    {
        errorSnippet: "return result",
        p: "how to fix return in verscript function",
        vars: ["error unexpected token return in verscript", "how to return in verscript", "why does return not work in verscript", "fix return statement in verscript"],
        r: "VerScript uses the keyword `reply` to return values from functions, not `return`.\n\n```verscript\n! Incorrect:\n! def add(a, b)\n!   return a + b\n\n! Corrected:\ndef add(a, b)\n  reply a + b\n```"
    },
    {
        errorSnippet: "new User('Alice')",
        p: "how to fix new User in verscript",
        vars: ["syntax error unexpected keyword new in verscript", "how to instantiate class in verscript", "fix class instantiation in verscript"],
        r: "In VerScript, objects are instantiated using the `inst` keyword, not `new`.\n\n```verscript\n! Incorrect:\n! u : new User(\"Alice\")\n\n! Corrected:\nu : inst User(\"Alice\")\n```"
    },
    {
        errorSnippet: "for (let i = 0; i < 5; i++)",
        p: "how to fix c-style for loop in verscript",
        vars: ["syntax error for let i in verscript", "how to write for loop in verscript", "fix loop syntax in verscript"],
        r: "VerScript uses clean `iterate` or `loop` syntax instead of C-style `for` loops.\n\n```verscript\n! Incorrect:\n! for (let i = 0; i < 5; i++)\n\n! Corrected:\niterate i from 0 to 4\n  display \"Index: \" + i\n```"
    },
    {
        errorSnippet: "display(hello)",
        p: "how to fix display(hello) in verscript",
        vars: ["error printing variable in verscript", "display syntax in verscript", "how to print in verscript"],
        r: "In VerScript, `display` is a statement that accepts space-separated arguments or expressions without required outer parentheses:\n\n```verscript\n! Corrected:\nmessage : \"Hello World\"\ndisplay message\ndisplay \"Message: \" + message ?color=\"cyan\"\n```"
    },
    {
        errorSnippet: "def add(a, b):",
        p: "how to fix colon after def in verscript",
        vars: ["syntax error colon after function declaration in verscript", "def syntax in verscript"],
        r: "In VerScript, function definitions do not take a trailing colon `:` at the end of the `def` line:\n\n```verscript\n! Incorrect:\n! def calculate(x, y):\n\n! Corrected:\ndef calculate(x, y)\n  reply x * y\n```"
    },
    {
        errorSnippet: "import math",
        p: "how to import libraries in verscript",
        vars: ["import statement in verscript", "how to use standard libraries in verscript", "how to load modules in verscript"],
        r: "In VerScript, standard libraries (like `Math`, `Strings`, `Arrays`, `Files`, `System`, `Matrix`) are built-in and globally available in the runtime environment:\n\n```verscript\nval : Math.sqrt(64)\ndisplay \"Square root: \" + val\n```"
    }
];
codeRepairs.forEach(item => addPair(item.p, item.r, item.vars));

// ============================================================================
// 3.5 COMPREHENSIVE STDLIB METHOD WALKTHROUGHS & DEMOS
// ============================================================================
const stdlibApiWalkthroughs = [
    // Math
    { lib: "Math", fn: "abs", sig: "Math.abs(val)", desc: "returns the absolute value of a number", code: "val : -42\nresult : Math.abs(val)\ndisplay \"Absolute value: \" + result" },
    { lib: "Math", fn: "floor", sig: "Math.floor(val)", desc: "rounds down to the nearest integer", code: "val : 7.89\nresult : Math.floor(val)\ndisplay \"Floored value: \" + result" },
    { lib: "Math", fn: "ceil", sig: "Math.ceil(val)", desc: "rounds up to the nearest integer", code: "val : 3.14\nresult : Math.ceil(val)\ndisplay \"Ceiled value: \" + result" },
    { lib: "Math", fn: "round", sig: "Math.round(val)", desc: "rounds to the nearest integer", code: "val : 4.5\nresult : Math.round(val)\ndisplay \"Rounded value: \" + result" },
    { lib: "Math", fn: "sqrt", sig: "Math.sqrt(val)", desc: "calculates the square root of a non-negative number", code: "val : 144\nresult : Math.sqrt(val)\ndisplay \"Square root: \" + result" },
    { lib: "Math", fn: "pow", sig: "Math.pow(base, exp)", desc: "raises base to the power of exp", code: "result : Math.pow(2, 8)\ndisplay \"2^8 = \" + result" },
    { lib: "Math", fn: "min", sig: "Math.min(a, b)", desc: "returns the minimum of two numeric values", code: "result : Math.min(18, 42)\ndisplay \"Minimum: \" + result" },
    { lib: "Math", fn: "max", sig: "Math.max(a, b)", desc: "returns the maximum of two numeric values", code: "result : Math.max(18, 42)\ndisplay \"Maximum: \" + result" },
    { lib: "Math", fn: "hypot", sig: "Math.hypot(a, b)", desc: "computes the hypotenuse sqrt(a^2 + b^2)", code: "h : Math.hypot(3, 4)\ndisplay \"Hypotenuse: \" + h" },

    // Strings
    { lib: "Strings", fn: "upper", sig: "Strings.upper(str)", desc: "converts all characters in string to uppercase", code: "s : \"verscript\"\ndisplay Strings.upper(s)" },
    { lib: "Strings", fn: "lower", sig: "Strings.lower(str)", desc: "converts all characters in string to lowercase", code: "s : \"VERSCRIPT\"\ndisplay Strings.lower(s)" },
    { lib: "Strings", fn: "trim", sig: "Strings.trim(str)", desc: "removes whitespace from beginning and end of string", code: "s : \"  clean text  \"\ndisplay Strings.trim(s)" },
    { lib: "Strings", fn: "contains", sig: "Strings.contains(str, sub)", desc: "checks if substring exists within target string", code: "if Strings.contains(\"hello world\", \"world\")\n  display \"Substring found!\"" },
    { lib: "Strings", fn: "startsWith", sig: "Strings.startsWith(str, prefix)", desc: "returns true if string begins with prefix", code: "if Strings.startsWith(\"https://verscript.org\", \"https\")\n  display \"Secure URL\"" },
    { lib: "Strings", fn: "endsWith", sig: "Strings.endsWith(str, suffix)", desc: "returns true if string ends with suffix", code: "if Strings.endsWith(\"app.vrs\", \".vrs\")\n  display \"VerScript file detected\"" },
    { lib: "Strings", fn: "replace", sig: "Strings.replace(str, old_sub, new_sub)", desc: "replaces substring matches with new text", code: "res : Strings.replace(\"hello world\", \"world\", \"VerScript\")\ndisplay res" },
    { lib: "Strings", fn: "split", sig: "Strings.split(str, delim)", desc: "splits string by delimiter into an array of strings", code: "tokens : Strings.split(\"apple,banana,orange\", \",\")\ndisplay tokens" },
    { lib: "Strings", fn: "join", sig: "Strings.join(arr, delim)", desc: "joins array elements into a single delimited string", code: "words : [\"High\", \"Performance\", \"VerScript\"]\ndisplay Strings.join(words, \" \")" },
    { lib: "Strings", fn: "reverse", sig: "Strings.reverse(str)", desc: "reverses the character sequence of a string", code: "display Strings.reverse(\"verscript\")" },

    // Arrays
    { lib: "Arrays", fn: "push", sig: "Arrays.push(arr, item)", desc: "appends an item to the end of the array", code: "nums : [1, 2, 3]\nArrays.push(nums, 4)\ndisplay nums" },
    { lib: "Arrays", fn: "pop", sig: "Arrays.pop(arr)", desc: "removes and returns the last element of the array", code: "nums : [10, 20, 30]\nval : Arrays.pop(nums)\ndisplay \"Popped: \" + val" },
    { lib: "Arrays", fn: "slice", sig: "Arrays.slice(arr, start, end)", desc: "extracts a sub-slice of elements", code: "items : [10, 20, 30, 40, 50]\nsub : Arrays.slice(items, 1, 4)\ndisplay sub" },
    { lib: "Arrays", fn: "sort", sig: "Arrays.sort(arr)", desc: "sorts array elements in ascending order", code: "items : [8, 3, 1, 9, 2]\nsorted : Arrays.sort(items)\ndisplay sorted" },
    { lib: "Arrays", fn: "reverse", sig: "Arrays.reverse(arr)", desc: "reverses array elements in place", code: "items : [1, 2, 3, 4]\nrev : Arrays.reverse(items)\ndisplay rev" },
    { lib: "Arrays", fn: "unique", sig: "Arrays.unique(arr)", desc: "removes duplicate elements from array", code: "items : [1, 2, 2, 3, 1, 4]\ndisplay Arrays.unique(items)" },

    // Matrix
    { lib: "Matrix", fn: "identity", sig: "Matrix.identity(n)", desc: "creates an n x n identity matrix", code: "I : Matrix.identity(3)\nMatrix.print(I)" },
    { lib: "Matrix", fn: "add", sig: "Matrix.add(m1, m2)", desc: "adds two matrices of identical dimensions", code: "A : [[1, 2], [3, 4]]\nB : [[5, 6], [7, 8]]\nC : Matrix.add(A, B)\nMatrix.print(C)" },
    { lib: "Matrix", fn: "mult", sig: "Matrix.mult(m1, m2)", desc: "multiplies two compatible matrices", code: "A : [[1, 2], [3, 4]]\nB : [[2, 0], [1, 2]]\nC : Matrix.mult(A, B)\nMatrix.print(C)" },
    { lib: "Matrix", fn: "transpose", sig: "Matrix.transpose(m)", desc: "transposes matrix rows and columns", code: "A : [[1, 2, 3], [4, 5, 6]]\nT : Matrix.transpose(A)\nMatrix.print(T)" },
    { lib: "Matrix", fn: "det", sig: "Matrix.det(m)", desc: "calculates the determinant of a square matrix", code: "A : [[4, 7], [2, 6]]\nd : Matrix.det(A)\ndisplay \"Determinant: \" + d" },

    // Files & System
    { lib: "Files", fn: "readText", sig: "Files.readText(path)", desc: "reads text file content synchronously", code: "data : Files.readText(\"config.json\")\ndisplay data" },
    { lib: "Files", fn: "writeText", sig: "Files.writeText(path, content)", desc: "writes text content to a file", code: "Files.writeText(\"output.txt\", \"Hello from VerScript\")\ndisplay \"File saved successfully\"" },
    { lib: "Files", fn: "exists", sig: "Files.exists(path)", desc: "checks if file exists on disk", code: "if Files.exists(\"notes.txt\")\n  display \"File exists!\"" },
    { lib: "System", fn: "time", sig: "System.time()", desc: "returns current epoch timestamp in milliseconds", code: "now : System.time()\ndisplay \"Current timestamp: \" + now" },
    { lib: "System", fn: "platform", sig: "System.platform()", desc: "returns host operating system", code: "os_name : System.platform()\ndisplay \"Operating system: \" + os_name" },
    { lib: "Log", fn: "info", sig: "Log.info(msg)", desc: "logs an informational message", code: "Log.info(\"Application service started\")" },
    { lib: "Log", fn: "error", sig: "Log.error(msg)", desc: "logs an error message", code: "Log.error(\"Failed to connect to host\")" },
    { lib: "Assert", fn: "equal", sig: "Assert.equal(actual, expected)", desc: "asserts two values are equal", code: "Assert.equal(5 * 5, 25, \"Math verification failed\")" }
];

stdlibApiWalkthroughs.forEach(item => {
    addPair(
        `how to use ${item.lib}.${item.fn} in verscript`,
        `### \`${item.sig}\`\n\n${item.desc}.\n\n\`\`\`verscript\n${item.code}\n\`\`\``,
        [
            `how to call ${item.lib}.${item.fn} in verscript`,
            `show example of ${item.lib}.${item.fn} in verscript`,
            `explain ${item.lib}.${item.fn} in verscript`
        ]
    );
});

// ============================================================================
// 4. COMBINATORIAL ALGORITHMIC & MATHEMATICAL PROGRAM GENERATORS
// ============================================================================
console.log("[Dataset] Generating combinatorial algorithms & mathematical programs...");

const algorithms = [
    {
        name: "fibonacci",
        generate: (n) => ({
            p: `write a program to calculate the first ${n} fibonacci numbers in verscript`,
            vars: [`generate ${n} fibonacci numbers in verscript`, `fibonacci sequence of length ${n} in verscript`],
            r: `Here is a VerScript program calculating the first ${n} Fibonacci numbers:\n\n\`\`\`verscript\ndef fibonacci(n)\n  if n <= 0\n    reply []\n  if n == 1\n    reply [0]\n  seq : [0, 1]\n  while length(seq) < n\n    last_idx : length(seq) - 1\n    next_val : seq[last_idx] + seq[last_idx - 1]\n    push(seq, next_val)\n  reply seq\n\nresult : fibonacci(${n})\ndisplay \"First ${n} Fibonacci numbers: \" + result\n\`\`\``
        })
    },
    {
        name: "is_prime",
        generate: (num) => ({
            p: `write a program to check if ${num} is a prime number in verscript`,
            vars: [`is ${num} prime in verscript`, `check prime number ${num} in verscript`],
            r: `Here is a VerScript function to verify if ${num} is prime:\n\n\`\`\`verscript\ndef is_prime(n)\n  if n < 2\n    reply false\n  if n == 2\n    reply true\n  limit : Math.floor(Math.sqrt(n))\n  iterate d from 2 to limit\n    if n % d == 0\n      reply false\n  reply true\n\ncheck_num : ${num}\nif is_prime(check_num)\n  display check_num + \" is PRIME!\" ?color=\"green\"\nelse\n  display check_num + \" is NOT prime.\" ?color=\"red\"\n\`\`\``
        })
    },
    {
        name: "factorial",
        generate: (n) => ({
            p: `write a function to compute the factorial of ${n} in verscript`,
            vars: [`calculate ${n} factorial in verscript`, `factorial of ${n} in verscript`],
            r: `Here is how to calculate ${n}! in VerScript:\n\n\`\`\`verscript\ndef factorial(n)\n  if n <= 1\n    reply 1\n  total : 1\n  iterate i from 2 to n\n    total : total * i\n  reply total\n\nresult : factorial(${n})\ndisplay \"Factorial of ${n} is: \" + result\n\`\`\``
        })
    },
    {
        name: "sum_range",
        generate: (max) => ({
            p: `write a program to calculate the sum of numbers from 1 to ${max} in verscript`,
            vars: [`sum from 1 to ${max} in verscript`, `add numbers up to ${max} in verscript`],
            r: `Here is a VerScript script summing numbers from 1 to ${max}:\n\n\`\`\`verscript\ntotal : 0\niterate i from 1 to ${max}\n  total : total + i\ndisplay \"Sum from 1 to ${max} = \" + total\n\`\`\``
        })
    },
    {
        name: "bubble_sort",
        generate: (items) => ({
            p: `write a program to sort [${items.join(', ')}] using bubble sort in verscript`,
            vars: [`bubble sort [${items.join(', ')}] in verscript`, `how to bubble sort an array in verscript`],
            r: `Here is a bubble sort implementation in VerScript:\n\n\`\`\`verscript\ndef bubble_sort(arr)\n  n : length(arr)\n  iterate i from 0 to n - 1\n    iterate j from 0 to n - i - 2\n      if arr[j] > arr[j + 1]\n        temp : arr[j]\n        arr[j] : arr[j + 1]\n        arr[j + 1] : temp\n  reply arr\n\nnumbers : [${items.join(', ')}]\nsorted : bubble_sort(numbers)\ndisplay \"Sorted array: \" + sorted\n\`\`\``
        })
    },
    {
        name: "binary_search",
        generate: (target) => ({
            p: `write binary search to find ${target} in a sorted array in verscript`,
            vars: [`search for ${target} with binary search in verscript`, `binary search algorithm in verscript`],
            r: `Here is a binary search algorithm in VerScript:\n\n\`\`\`verscript\ndef binary_search(arr, target)\n  low : 0\n  high : length(arr) - 1\n  while low <= high\n    mid : Math.floor((low + high) / 2)\n    if arr[mid] == target\n      reply mid\n    else if arr[mid] < target\n      low : mid + 1\n    else\n      high : mid - 1\n  reply -1\n\ndata : [2, 5, 8, 12, 19, 23, 31, 45, 56]\nidx : binary_search(data, ${target})\nif idx != -1\n  display \"Target ${target} found at index: \" + idx\nelse\n  display \"Target ${target} not in list.\"\n\`\`\``
        })
    },
    {
        name: "temperature_converter",
        generate: (c) => ({
            p: `write a function to convert ${c} celsius to fahrenheit in verscript`,
            vars: [`celsius to fahrenheit converter in verscript`, `convert ${c} C to F in verscript`],
            r: `Here is a temperature conversion function in VerScript:\n\n\`\`\`verscript\ndef c_to_f(celsius)\n  reply (celsius * 9 / 5) + 32\n\nc : ${c}\nf : c_to_f(c)\ndisplay c + \" C = \" + f + \" F\"\n\`\`\``
        })
    },
    {
        name: "palindrome_check",
        generate: (word) => ({
            p: `write a program to check if \"${word}\" is a palindrome in verscript`,
            vars: [`check if ${word} is palindrome in verscript`, `palindrome checker in verscript`],
            r: `Here is a palindrome checker in VerScript:\n\n\`\`\`verscript\ndef is_palindrome(text)\n  clean : Strings.lower(text)\n  rev : Strings.reverse(clean)\n  reply clean == rev\n\nword : \"${word}\"\nif is_palindrome(word)\n  display \"'\" + word + \"' is a palindrome!\" ?color=\"green\"\nelse\n  display \"'\" + word + \"' is NOT a palindrome.\" ?color=\"yellow\"\n\`\`\``
        })
    }
];

// Generate dynamic parameter instances
[5, 8, 10, 12, 15, 20, 25].forEach(n => {
    const item = algorithms.find(a => a.name === "fibonacci").generate(n);
    addPair(item.p, item.r, item.vars);
});

[7, 13, 17, 29, 37, 43, 59, 71, 89, 97, 101, 149].forEach(num => {
    const item = algorithms.find(a => a.name === "is_prime").generate(num);
    addPair(item.p, item.r, item.vars);
});

[3, 4, 5, 6, 7, 8, 9, 10].forEach(n => {
    const item = algorithms.find(a => a.name === "factorial").generate(n);
    addPair(item.p, item.r, item.vars);
});

[10, 25, 50, 100, 200, 500, 1000].forEach(max => {
    const item = algorithms.find(a => a.name === "sum_range").generate(max);
    addPair(item.p, item.r, item.vars);
});

[
    [5, 2, 9, 1, 5, 6],
    [64, 34, 25, 12, 22, 11, 90],
    [3, 0, 2, 5, -1, 4, 1],
    [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
].forEach(arr => {
    const item = algorithms.find(a => a.name === "bubble_sort").generate(arr);
    addPair(item.p, item.r, item.vars);
});

[12, 23, 45, 99, 5].forEach(t => {
    const item = algorithms.find(a => a.name === "binary_search").generate(t);
    addPair(item.p, item.r, item.vars);
});

[0, 20, 25, 37, 100, -10].forEach(c => {
    const item = algorithms.find(a => a.name === "temperature_converter").generate(c);
    addPair(item.p, item.r, item.vars);
});

["racecar", "madam", "verscript", "level", "radar", "developer"].forEach(w => {
    const item = algorithms.find(a => a.name === "palindrome_check").generate(w);
    addPair(item.p, item.r, item.vars);
});

// ============================================================================
// 5. CROSS-LANGUAGE CODE TRANSLATION SUITES
// ============================================================================
const translations = [
    {
        lang: "Python",
        fromCode: "def greet(name):\n    return f'Hello, {name}!'",
        toVrs: "def greet(name)\n  reply \"Hello, \" + name + \"!\"",
        task: "greet function"
    },
    {
        lang: "Python",
        fromCode: "total = 0\nfor x in range(1, 6):\n    total += x",
        toVrs: "total : 0\niterate x from 1 to 5\n  total : total + x",
        task: "sum loop"
    },
    {
        lang: "JavaScript",
        fromCode: "const square = (n) => n * n;",
        toVrs: "def square(n)\n  reply n * n",
        task: "square arrow function"
    },
    {
        lang: "JavaScript",
        fromCode: "const numbers = [1, 2, 3];\nnumbers.push(4);",
        toVrs: "numbers : [1, 2, 3]\npush(numbers, 4)",
        task: "array push"
    },
    {
        lang: "C",
        fromCode: "int max(int a, int b) {\n    if (a > b) return a;\n    return b;\n}",
        toVrs: "def max(a, b)\n  if a > b\n    reply a\n  reply b",
        task: "max comparison function"
    }
];

translations.forEach(t => {
    addPair(
        `convert this ${t.lang} code to verscript: ${t.task}`,
        `Here is the equivalent VerScript code:\n\n**${t.lang} Original:**\n\`\`\`${t.lang.toLowerCase()}\n${t.fromCode}\n\`\`\`\n\n**VerScript Equivalent:**\n\`\`\`verscript\n${t.toVrs}\n\`\`\``,
        [`translate ${t.lang} ${t.task} to verscript`, `how to write ${t.lang} ${t.task} in verscript`]
    );
});

// ============================================================================
// 6. POLYGLOT "INJECT" COMPREHENSIVE SUITES (PYTHON, JS, C, VERSCRIPT)
// ============================================================================
const polyglotSuites = [
    {
        lang: "python",
        title: "HTTP requests in Python",
        p: "how to make an http request using inject python in verscript",
        vars: ["http get in injected python", "fetch api in python inside verscript"],
        r: "Use 'inject python' with urllib or requests:\n\n```verscript\ndisplay \"Fetching data via Injected Python:\" ?color=\"cyan\"\ninject python\n  import urllib.request\n  import json\n  url = \"https://api.github.com\"\n  req = urllib.request.Request(url, headers={'User-Agent': 'VS-Sharp'})\n  with urllib.request.urlopen(req) as resp:\n      data = json.loads(resp.read().decode('utf-8'))\n      print(\"GitHub API Status:\", resp.status)\n      print(\"Current User URL:\", data.get('current_user_url'))\ndisplay \"Python HTTP complete.\" ?color=\"green\"\n```"
    },
    {
        lang: "python",
        title: "Matrix multiplication in Python",
        p: "how to multiply matrices in python inside verscript",
        vars: ["matrix multiplication using inject python in verscript"],
        r: "Here is matrix multiplication executed via injected Python:\n\n```verscript\ndisplay \"Multiplying 2x2 matrices:\"\ninject python\n  A = [[1, 2], [3, 4]]\n  B = [[5, 6], [7, 8]]\n  C = [[sum(a * b for a, b in zip(row_a, col_b)) for col_b in zip(*B)] for row_a in A]\n  print(\"Resulting Matrix:\", C)\n```"
    },
    {
        lang: "javascript",
        title: "Async fetch in JavaScript",
        p: "write javascript async fetch function",
        vars: ["async fetch in javascript", "fetch api in javascript", "how to fetch data in javascript"],
        r: "Here is an async/await fetch helper in JavaScript:\n\n```javascript\nasync function fetchUserData(userId) {\n  try {\n    const response = await fetch(`https://api.example.com/users/${userId}`);\n    if (!response.ok) throw new Error(`HTTP error ${response.status}`);\n    const data = await response.json();\n    console.log(\"User data:\", data);\n    return data;\n  } catch (err) {\n    console.error(\"Fetch failed:\", err.message);\n  }\n}\n```"
    },
    {
        lang: "javascript",
        title: "Express server in JavaScript",
        p: "write javascript express server route",
        vars: ["express server in javascript", "express route in js"],
        r: "Here is an Express server route in JavaScript:\n\n```javascript\nconst express = require('express');\nconst app = express();\napp.use(express.json());\n\napp.get('/api/status', (req, res) => {\n  res.json({ status: 'online', timestamp: Date.now() });\n});\n\napp.post('/api/compute', (req, res) => {\n  const { val1, val2 } = req.body;\n  res.json({ result: Number(val1) + Number(val2) });\n});\n```"
    },
    {
        lang: "c",
        title: "Binary search in C",
        p: "write c code for binary search",
        vars: ["binary search in c", "c binary search implementation"],
        r: "Here is binary search implemented in C:\n\n```c\n#include <stdio.h>\n\nint binary_search(int arr[], int size, int target) {\n    int low = 0, high = size - 1;\n    while (low <= high) {\n        int mid = low + (high - low) / 2;\n        if (arr[mid] == target) return mid;\n        if (arr[mid] < target) low = mid + 1;\n        else high = mid - 1;\n    }\n    return -1;\n}\n\nint main() {\n    int data[] = { 3, 7, 12, 19, 25, 33, 42 };\n    int idx = binary_search(data, 7, 25);\n    printf(\"Found target 25 at index: %d\\n\", idx);\n    return 0;\n}\n```"
    },
    {
        lang: "c",
        title: "Quicksort in C",
        p: "write c code for quicksort",
        vars: ["quicksort algorithm in c", "c quicksort implementation"],
        r: "Here is quicksort implemented in C:\n\n```c\n#include <stdio.h>\n\nvoid swap(int* a, int* b) { int t = *a; *a = *b; *b = t; }\n\nint partition(int arr[], int low, int high) {\n    int pivot = arr[high];\n    int i = (low - 1);\n    for (int j = low; j < high; j++) {\n        if (arr[j] < pivot) {\n            i++;\n            swap(&arr[i], &arr[j]);\n        }\n    }\n    swap(&arr[i + 1], &arr[high]);\n    return (i + 1);\n}\n\nvoid quicksort(int arr[], int low, int high) {\n    if (low < high) {\n        int pi = partition(arr, low, high);\n        quicksort(arr, low, pi - 1);\n        quicksort(arr, pi + 1, high);\n    }\n}\n```"
    }
];
polyglotSuites.forEach(s => addPair(s.p, s.r, s.vars));

// ============================================================================
// 7. INGEST ACADEMY CURRICULUM (69 ARTICLES)
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
                addPair(
                    `explain ${article.title} in verscript`,
                    `### ${article.title}\n\n${article.summary}`,
                    [`what is ${article.title}`, `learn ${article.title}`]
                );

                if (Array.isArray(article.codeBlocks)) {
                    for (const cb of article.codeBlocks) {
                        addPair(
                            `show example of ${cb.title}`,
                            `### ${cb.title}\n\n\`\`\`verscript\n${cb.code}\n\`\`\``,
                            [`code example for ${article.title}`]
                        );
                    }
                }

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
// 8. INGEST STDLIB FUNCTIONS
// ============================================================================
console.log("[Dataset] Ingesting VerScript Standard Libraries & Functions...");
const stdlibDirs = [
    path.join(__dirname, 'stdlib'),
    path.join(ORG_ROOT, 'VerScript', 'stdlib')
];

for (const sDir of stdlibDirs) {
    if (!fs.existsSync(sDir)) continue;
    for (const f of fs.readdirSync(sDir)) {
        if (!f.endsWith('.vrs')) continue;
        const libName = path.basename(f, '.lib.vrs');
        const content = fs.readFileSync(path.join(sDir, f), 'utf8');
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
// 9. SCAN ALL REPOSITORIES AND TEST SUITES
// ============================================================================
console.log("[Dataset] Ingesting test suites and repository source files...");
const repos = [
    { name: 'VerScript', path: path.join(ORG_ROOT, 'VerScript') },
    { name: 'VerScript.github.io', path: path.join(ORG_ROOT, 'VerScript.github.io') },
    { name: 'IDE', path: path.join(ORG_ROOT, 'IDE') },
    { name: 'polyserver', path: path.join(ORG_ROOT, 'polyserver') },
    { name: 'VS-Sharp', path: path.join(ORG_ROOT, 'VS-Sharp') },
    { name: 'tests', path: path.join(ORG_ROOT, 'tests') },
    { name: 'verscript-studio-android', path: path.join(ORG_ROOT, 'verscript-studio-android') }
];

function scanFiles(dir, repoName) {
    if (!fs.existsSync(dir)) return;
    let entries;
    try { entries = fs.readdirSync(dir, { withFileTypes: true }); } catch (_) { return; }

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
            } else if (ext === '.js' && !e.name.includes('config') && !e.name.includes('test') && !e.name.includes('model_weights')) {
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

// Ingest root .vrs test files
if (fs.existsSync(ORG_ROOT)) {
    try {
        const rootEntries = fs.readdirSync(ORG_ROOT, { withFileTypes: true });
        for (const re of rootEntries) {
            if (re.isFile() && re.name.endsWith('.vrs')) {
                const fullPath = path.join(ORG_ROOT, re.name);
                const content = fs.readFileSync(fullPath, 'utf8').trim();
                const base = path.basename(re.name, '.vrs');
                if (content.length > 20 && content.length < 2500) {
                    addPair(
                        `show ${base} test script in verscript`,
                        `### Test Script: ${re.name}\n\n\`\`\`verscript\n${content}\n\`\`\``,
                        [`how to write ${base} in verscript`, `example test for ${base} in verscript`]
                    );
                }
            }
        }
    } catch (_) {}
}

// ============================================================================
// 10. SAVE MASTER DATASET
// ============================================================================
fs.mkdirSync(path.dirname(OUTPUT_FILE), { recursive: true });
fs.writeFileSync(OUTPUT_FILE, JSON.stringify(trainingData, null, 2), 'utf8');

console.log("==================================================================");
console.log(`✅ Successfully compiled ${trainingData.length.toLocaleString()} rich training pairs!`);
console.log(`📁 Saved to: ${OUTPUT_FILE}`);
console.log("==================================================================");
