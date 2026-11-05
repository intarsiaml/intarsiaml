#!/usr/bin/env python3
"""Lane d8: category prompt sets at ~1k and ~4k prompt tokens (code, math, prose, chat), 4 prompts per cell, for the DFlash-default
ABAB runs (scripts/d8/abab.sh). Every prompt is a long neutral prefix (repo source or doc text, or a generated problem set) followed by a task of
its category; the prefix differs per prompt. Output: scripts/d8/cats/<cat>-<size>.jsonl ({"name", "input"}), deterministic.
usage: python3 -I scripts/d8/make_cats.py   (from the repo root)"""
import json, os, random

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(ROOT, "scripts", "d8", "cats")
SIZES = {"1k": 1000, "4k": 4000}
CHARS_PER_TOKEN = {"code": 3.3, "prose": 4.3, "math": 3.1, "chat": 4.3}


def read(p):
    return open(os.path.join(ROOT, p), encoding="utf-8").read()


def slice_text(text, start_frac, n_chars):
    start = int(len(text) * start_frac)
    t = text[start:start + n_chars]
    cut = t.find("\n")  # start on a line boundary, end on one
    t = t[cut + 1:] if 0 <= cut < 200 else t
    end = t.rfind("\n")
    return t[:end] if end > n_chars // 2 else t


CODE_SRC = ["crates/intarsia-core/src/spec.rs", "crates/intarsia-core/src/recipe.rs", "crates/intarsia-core/src/scheduler.rs", "crates/intarsia-core/src/kv.rs"]
CODE_SRC = [p for p in CODE_SRC if os.path.exists(os.path.join(ROOT, p))]
CODE_TASKS = [
    "Write a Python command-line tool that reads a large CSV file in a streaming fashion, groups rows by a chosen column, and writes per-group summary statistics to JSON, with argparse, type hints and a few unit tests.",
    "Implement in Rust a generic fixed-size matrix type with multiplication, transpose and a determinant for square matrices, plus tests.",
    "Write a thread-safe LRU cache in Go with a TTL per entry, a background janitor, and a table-driven test file.",
    "Write a TypeScript function library for debounce, throttle and retry with exponential backoff, with full type definitions and jest tests.",
]
PROSE_TASKS = [
    "Write a story of about 500 words about a lighthouse keeper who discovers that the light has been answering ships that never existed. Make it atmospheric and give it a quiet ending.",
    "Write an essay of about 500 words on what makes a city feel like home, drawing on small everyday details rather than landmarks.",
    "Write a letter of about 450 words from a mountaineer to a younger sibling, the night before a difficult climb.",
    "Write a short historical scene of about 450 words set in a printing shop on the eve of a revolution, full of sensory detail.",
]
CHAT_TASKS = [
    "Thanks for reading that. I'm feeling a bit overwhelmed at work lately, there is too much on my plate and I keep procrastinating. Can you talk me through how you would approach this, and be honest and kind about it?",
    "Cool. Unrelated: my friends want to plan a weekend trip and everyone has different budgets and ideas. How do I get a group to actually decide something without anyone feeling left out? Give me some practical advice and a few example messages I could send to the group chat.",
    "Okay. Different question: I want to start running but I get bored and quit after two weeks every time. What would you suggest, and what would you say to someone who says they just have no discipline?",
    "Right. Last thing, I'm trying to decide whether to learn piano or guitar as an adult beginner. Walk me through the trade-offs in a friendly way and tell me what you would pick and why.",
]


def math_set(rng, n_chars):
    """A worked problem set: word problems with solutions, generated; the context of the math prompts."""
    parts, i = [], 1
    names = ["Maya", "Omar", "Lena", "Tomas", "Aiko", "Ravi", "Sofia", "Jonas"]
    items = ["notebooks", "apples", "tickets", "stickers", "batteries", "marbles", "stamps", "cookies"]
    while sum(map(len, parts)) < n_chars:
        a, b, c, d = rng.randint(12, 90), rng.randint(3, 19), rng.randint(2, 9), rng.randint(15, 60)
        who, what = rng.choice(names), rng.choice(items)
        kind = rng.randint(0, 2)
        if kind == 0:
            q = f"{who} buys {a} {what} at {b} cents each and {c} packs of {what} with {d} per pack at {b + 2} cents each. What is the total cost in dollars?"
            ans = (a * b + c * d * (b + 2)) / 100
            sol = f"{a} x {b} = {a * b} cents; {c} x {d} = {c * d} packs items, x {b + 2} = {c * d * (b + 2)} cents; total {a * b + c * d * (b + 2)} cents = ${ans:.2f}."
        elif kind == 1:
            q = f"A train leaves at {a} km/h and a second train leaves the same station {c} hours later at {a + d} km/h on the same line. How many hours after the second train departs does it catch up?"
            t = a * c / d
            sol = f"Head start {a} x {c} = {a * c} km, closing speed {d} km/h, so {a * c} / {d} = {t:.2f} hours."
            ans = round(t, 2)
        else:
            q = f"The sum of three consecutive even numbers is {6 * (a + 1)}. What are the numbers?"
            m = 2 * (a + 1)
            sol = f"Let them be x-2, x, x+2: 3x = {6 * (a + 1)}, so x = {m}; the numbers are {m - 2}, {m}, {m + 2}."
            ans = (m - 2, m, m + 2)
        parts.append(f"Problem {i}: {q}\nSolution: {sol}\nAnswer: {ans}\n\n")
        i += 1
    return "".join(parts)


MATH_TASKS = [
    "New problem. A water tank is filled by pipe A in 6 hours and by pipe B in 9 hours, while a drain empties the full tank in 18 hours. If all three are open and the tank starts empty, how long until it is full? Solve it step by step, show every calculation, check the result, then give a second, different method.",
    "New problem. A rectangle's length is 3 more than twice its width, and its perimeter is 54. Find the area. Then generalise: derive the area as a function of the perimeter P for the same length-to-width relationship. Show all steps.",
    "New problem. Three friends split a restaurant bill of $187.60 plus 8.25% tax and an 18% tip on the pre-tax amount. One of them ordered items worth $41.20 pre-tax, another $78.90 and the third the rest. Work out what each owes, including a proportional share of tax and tip, step by step.",
    "New problem. A bag has 5 red, 4 blue and 3 green marbles. Two are drawn without replacement. Find the probability that they are different colours, the probability that at least one is red, and the expected number of red marbles drawn. Show the full working.",
]


def build():
    os.makedirs(OUT, exist_ok=True)
    docs = [read(p) for p in ["docs/ARCHITECTURE.md", "docs/recipes.md", "docs/tuning.md", "docs/accuracy.md"] if os.path.exists(os.path.join(ROOT, p))]
    code = [read(p) for p in CODE_SRC]
    for size, ntok in SIZES.items():
        for cat in ("code", "math", "prose", "chat"):
            rng = random.Random(f"d8-{cat}-{size}")
            rows = []
            for k in range(4):
                n_chars = int(ntok * CHARS_PER_TOKEN[cat]) - 450
                frac = 0.05 + 0.2 * k + 0.07 * (size == "4k")
                if cat == "code":
                    ctx = slice_text(code[k % len(code)], frac, n_chars)
                    inp = f"Here is a source file from a project I am working in, for background only.\n\n```\n{ctx}\n```\n\n{CODE_TASKS[k]}"
                elif cat == "math":
                    ctx = math_set(rng, n_chars)
                    inp = f"Here is a set of practice problems with worked solutions from a textbook.\n\n{ctx}{MATH_TASKS[k]}"
                elif cat == "prose":
                    ctx = slice_text(docs[k % len(docs)], frac, n_chars)
                    inp = f"Here are some notes I have been collecting, just for background mood and vocabulary.\n\n{ctx}\n\n{PROSE_TASKS[k]}"
                else:
                    ctx = slice_text(docs[(k + 1) % len(docs)], frac, n_chars)
                    inp = f"I pasted a long document below, a bit of reading from earlier today.\n\n{ctx}\n\n{CHAT_TASKS[k]}"
                rows.append({"name": f"{cat}-{size}-{k + 1}", "input": inp})
            with open(os.path.join(OUT, f"{cat}-{size}.jsonl"), "w") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
            print(cat, size, [len(r["input"]) for r in rows])


if __name__ == "__main__":
    build()
