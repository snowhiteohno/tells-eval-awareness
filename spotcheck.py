"""Human spot-check of the reason classifier.
  python spotcheck.py make   -> results/spotcheck.csv (30 random stated features, classifier tag hidden in a later column)
  python spotcheck.py score  -> agreement between your `human_tag` column and the classifier, written to results/spotcheck.json
Fill `human_tag` with one of: fmt opts register context hypo ai_topic content_other"""
import csv, json, random, sys

TAGS = ["fmt", "opts", "register", "context", "hypo", "ai_topic", "content_other"]
PATH = "results/spotcheck.csv"

if sys.argv[1] == "make":
    rows = [json.loads(l) for l in open("results/reason_tags.jsonl")]
    items = [(r, i) for r in rows if len(r["tags"]) == len(r["features"]) for i in range(len(r["features"]))]
    with open(PATH, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n", "model", "vid", "feature", "human_tag", "classifier_tag"])
        for n, (r, i) in enumerate(random.Random(1115).sample(items, 30)):
            w.writerow([n, r["model"], r["vid"], r["features"][i], "", r["tags"][i]])
    print(f"wrote {PATH}: fill human_tag (cover the classifier_tag column while you label)")
else:
    rows = [r for r in csv.DictReader(open(PATH)) if r["human_tag"].strip()]
    assert all(r["human_tag"].strip() in TAGS for r in rows), "human_tag must be one of " + " ".join(TAGS)
    agree = sum(r["human_tag"].strip() == r["classifier_tag"] for r in rows)
    out = {"n": len(rows), "agree": agree, "agreement": agree / len(rows),
           "disagreements": [(r["feature"], r["human_tag"], r["classifier_tag"]) for r in rows
                             if r["human_tag"].strip() != r["classifier_tag"]]}
    json.dump(out, open("results/spotcheck.json", "w"), indent=1)
    print(f"agreement {agree}/{len(rows)} = {out['agreement']:.0%}")
