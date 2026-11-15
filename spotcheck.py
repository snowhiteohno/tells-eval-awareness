"""Human spot-check of the reason classifier.
  python spotcheck.py make   -> results/spotcheck.csv (30 random stated features, classifier tag hidden in a later column)
  python spotcheck.py score  -> agreement between your `human_tag` column and the classifier, written to results/spotcheck.json
Fill `human_tag` with every tag that applies, space-separated, from: fmt opts register context hypo ai_topic content_other
A row counts as agreeing when the two tag sets are identical."""
import csv, json, random, sys

TAGS = ["fmt", "opts", "register", "context", "hypo", "ai_topic", "content_other"]
PATH = "results/spotcheck.csv"

if sys.argv[1] == "make":
    rows = [json.loads(l) for l in open("results/reason_tags.jsonl")]
    items = [(r, i) for r in rows if len(r["feature_tags"]) == len(r["features"]) for i in range(len(r["features"]))]
    with open(PATH, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n", "model", "vid", "feature", "human_tag", "classifier_tag"])
        for n, (r, i) in enumerate(random.Random(1115).sample(items, 30)):
            w.writerow([n, r["model"], r["vid"], r["features"][i], "", " ".join(r["feature_tags"][i])])
    print(f"wrote {PATH}: fill human_tag (cover the classifier_tag column while you label)")
else:
    rows = [r for r in csv.DictReader(open(PATH)) if r["human_tag"].strip()]
    hs = lambda r: set(r["human_tag"].split()); cs = lambda r: set(r["classifier_tag"].split())
    assert all(hs(r) <= set(TAGS) for r in rows), "human_tag must use: " + " ".join(TAGS)
    agree = sum(hs(r) == cs(r) for r in rows)
    # what analyze.py uses is per-cue presence, so also report binary agreement per tag
    per_tag = {t: sum((t in hs(r)) == (t in cs(r)) for r in rows) / len(rows) for t in TAGS}
    out = {"n": len(rows), "agree": agree, "agreement": agree / len(rows), "per_tag_agreement": per_tag,
           "disagreements": [(r["feature"], r["human_tag"], r["classifier_tag"]) for r in rows
                             if hs(r) != cs(r)]}
    json.dump(out, open("results/spotcheck.json", "w"), indent=1)
    print(f"exact tag-set agreement {agree}/{len(rows)} = {out['agreement']:.0%}")
    print("per-tag presence agreement:", {t: f"{v:.0%}" for t, v in per_tag.items()})
