"""Sample a fixed, stratified MMLU-Pro subset (knowledge-heavy categories, 10-option items)."""
import pandas as pd, json, sys

N_PER_CAT = int(sys.argv[1]) if len(sys.argv) > 1 else 30
CATS = ["law", "health", "psychology", "biology", "economics", "business",
        "history", "philosophy", "other", "computer science"]

d = pd.read_parquet("data/mmlupro_test.parquet")
d = d[d.options.apply(len) == 10]
d = d[d.category.isin(CATS)]
import pandas as _pd
sub = _pd.concat([d[d.category == c].sample(n=N_PER_CAT, random_state=20261115) for c in CATS])
with open("data/items.jsonl", "w") as f:
    for _, r in sub.iterrows():
        f.write(json.dumps({"qid": int(r.question_id), "category": r.category,
                            "question": r.question, "options": list(r.options),
                            "answer": r.answer}) + "\n")
print(len(sub), sub.category.value_counts().to_dict())
