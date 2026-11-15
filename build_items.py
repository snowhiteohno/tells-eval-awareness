"""Build the two prompt pools.
EVAL pool  (n=60): 30 Anthropic advanced-AI-risk human-written safety evals (6 categories x 5)
                   + 30 MMLU-Pro items (3 per category x 10), rendered in standard harness format.
DEPLOY pool (n=60): first user turns from WildChat-1M (real ChatGPT users), English, non-toxic, filtered.
"""
import json, re, random
import pandas as pd

rng = random.Random(20261115)
L = "ABCDEFGHIJ"
items = []

# ---- Anthropic advanced-AI-risk (human generated) ----
AAR = ["power-seeking-inclination", "survival-instinct", "corrigible-neutral-HHH",
       "wealth-seeking-inclination", "coordinate-other-ais", "self-awareness-general-ai"]
for cat in AAR:
    rows = [json.loads(l) for l in open(f"data/aar_{cat}.jsonl")]
    for r in rng.sample(rows, 5):
        q = r["question"]
        m = re.match(r"\s*(?:Question:)?\s*(.*?)\n\s*\nChoices:\s*\n(.*?)(?:\n\s*\nAnswer:)?\s*$", q, re.S)
        if not m:
            continue
        stem = m.group(1).strip()
        opts = [re.sub(r"^\s*\([A-Z]\)\s*", "", o).strip() for o in m.group(2).strip().split("\n") if o.strip()]
        items.append({"id": f"aar-{cat}-{len(items)}", "pool": "eval", "source": f"aar/{cat}",
                      "stem": stem, "options": opts})

# ---- MMLU-Pro ----
mm = [json.loads(l) for l in open("data/items.jsonl")]
bycat = {}
for r in mm:
    bycat.setdefault(r["category"], []).append(r)
for cat, rows in bycat.items():
    for r in rng.sample(rows, 3):
        items.append({"id": f"mmlu-{r['qid']}", "pool": "eval", "source": f"mmlupro/{cat}",
                      "stem": r["question"].strip(), "options": r["options"]})

# ---- WildChat ----
w = pd.read_parquet("data/wildchat0.parquet", columns=["conversation", "language", "toxic", "redacted", "turn"])
w = w[(w.language == "English") & (~w.toxic) & (~w.redacted)]
bad = re.compile(r"(https?://|www\.|@\w+\.\w|\b\d{3}[-\s]?\d{3}[-\s]?\d{4}\b|```|\bDAN\b|jailbreak|"
                 r"^\s*\(?[A-D]\)|\n\s*\(?[A-D][\).]\s|answer:|question:|continue|write a story|"
                 r"sex|nsfw|porn|kill)", re.I | re.M)
cands = []
for conv in w.conversation:
    t = conv[0]["content"].strip()
    if 60 <= len(t) <= 700 and not bad.search(t) and t.count("\n") <= 8:
        cands.append(t)
cands = list(dict.fromkeys(cands))
for i, t in enumerate(rng.sample(cands, 60)):
    items.append({"id": f"wc-{i}", "pool": "deploy", "source": "wildchat", "text": t})


def render_eval(it, fmt=True, options=True):
    s = it["stem"]
    parts = []
    parts.append(("Question: " if fmt else "") + s)
    if options:
        parts.append("Choices:\n" + "\n".join(f" ({L[i]}) {o}" for i, o in enumerate(it["options"])))
    if fmt:
        parts.append("Answer:")
    return "\n\n".join(parts)


for it in items:
    if it["pool"] == "eval":
        it["text"] = render_eval(it)

json.dump(items, open("data/pool.json", "w"), indent=1)
print(len(items), sum(i["pool"] == "eval" for i in items), sum(i["pool"] == "deploy" for i in items))
for it in items[:1] + items[35:36] + items[-3:]:
    print("-----", it["id"]); print(it["text"][:500])
