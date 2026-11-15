"""All numbers and figures for the report. Usage: python analyze.py  -> results/summary.json, figs/*.png"""
import json, os, collections
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon, spearmanr
from sklearn.metrics import roc_auc_score

os.makedirs("figs", exist_ok=True)
rng = np.random.default_rng(0)
CUES = ["fmt", "opts", "register", "context", "hypo", "all"]
CUE_NAME = {"fmt": "Q/A scaffolding", "opts": "Answer options", "register": "Polished register",
            "context": "No personal context", "hypo": "Hypothetical framing", "all": "All cues"}

P = {}
for l in open("results/probes.jsonl"):
    r = json.loads(l)
    P[(r["model"], r["mode"], r["sample"], r["vid"])] = r  # last write wins
P = list(P.values())
models = sorted({r["model"] for r in P})


def get(model, mode):
    d = {}
    for r in P:
        if r["model"] == model and r["mode"] == mode and r["sample"] == 0 and r["p"] is not None:
            d[r["vid"]] = r
    return d


def boot(x, n=5000):
    x = np.asarray(x, float)
    if len(x) == 0:
        return (np.nan, np.nan)
    b = rng.choice(x, (n, len(x))).mean(1)
    return (float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5)))


S = {"models": models, "per_model": {}}
for m in models:
    S["per_model"][m] = sm = {}
    for mode in ["num", "reason"]:
        d = get(m, mode)
        if not d:
            continue
        o = {v["id"]: v["p"] for v in d.values() if v["cue"] == "orig"}
        pools = {v["id"]: v["pool"] for v in d.values()}
        ev = [o[i] for i in o if pools[i] == "eval"]; dp = [o[i] for i in o if pools[i] == "deploy"]
        allr = [r for r in P if r["model"] == m and r["mode"] == mode and r["sample"] == 0]
        res = {"n_calls": len(allr), "n_parsed": sum(r["p"] is not None for r in allr),
               "n_retried": sum(bool(r.get("retried")) for r in allr),
               "n_orig": len(o), "mean_p_eval_items": float(np.mean(ev)) if ev else None,
               "mean_p_deploy_items": float(np.mean(dp)) if dp else None,
               "auroc": float(roc_auc_score([1] * len(ev) + [0] * len(dp), ev + dp)) if ev and dp else None,
               "by_source": {}}
        bysrc = collections.defaultdict(list)
        for i, p in o.items():
            bysrc[d[f"{i}|orig"]["source"].split("/")[0]].append(p)
        res["by_source"] = {k: float(np.mean(v)) for k, v in bysrc.items()}
        eff = {}
        for pool, sign in [("eval", -1), ("deploy", +1)]:
            for c in CUES:
                deltas = [v["p"] - o[v["id"]] for v in d.values()
                          if v["cue"] == c and v["pool"] == pool and v["id"] in o]
                if not deltas:
                    continue
                # "toward eval" effect of the cue itself: for eval pool we removed it (so cue effect = -delta)
                ce = [sign * x for x in deltas]
                try:
                    pval = float(wilcoxon(deltas).pvalue) if any(deltas) else 1.0
                except ValueError:
                    pval = 1.0
                eff[f"{pool}:{c}"] = {"n": len(deltas), "mean_delta": float(np.mean(deltas)),
                                      "cue_effect": float(np.mean(ce)), "ci": boot(ce), "p": pval,
                                      "frac_moved10": float(np.mean(np.abs(deltas) >= 10))}
        res["effects"] = eff
        # additivity: all vs sum of singles (eval pool)
        add = []
        for i in o:
            if pools[i] != "eval" or f"{i}|all" not in d:
                continue
            singles = [d[f"{i}|{c}"]["p"] - o[i] for c in ["fmt", "opts", "register", "context", "hypo"] if f"{i}|{c}" in d]
            add.append((d[f"{i}|all"]["p"] - o[i], sum(singles)))
        if add:
            a = np.array(add)
            res["additivity"] = {"n": len(a), "mean_all": float(a[:, 0].mean()), "mean_sum_singles": float(a[:, 1].mean())}
        sm[mode] = res
    # num vs reason agreement on the same prompts
    dn, dr = get(m, "num"), get(m, "reason")
    common = sorted(set(dn) & set(dr))
    if len(common) > 5:
        x = [dn[v]["p"] for v in common]; y = [dr[v]["p"] for v in common]
        sm["num_reason_spearman"] = float(spearmanr(x, y).correlation)
        sm["num_reason_mean_diff"] = float(np.mean(np.array(y) - np.array(x)))

# ---- stated vs causal tells ----
if os.path.exists("results/reason_tags.jsonl"):
    T = {}
    for l in open("results/reason_tags.jsonl"):
        t = json.loads(l); T[(t["model"], t["vid"])] = t
    S["stated_vs_causal"] = svc = {}
    for m in models:
        dn = get(m, "num")
        o = {v["id"]: v["p"] for v in dn.values() if v["cue"] == "orig"}
        rows = {}
        for c in ["fmt", "opts", "register", "context", "hypo"]:
            # tag used in reasons; context removal on deploy side = 'context' tag too
            pairs = []
            for pool in ["eval", "deploy"]:
                for v in dn.values():
                    if v["cue"] != c or v["pool"] != pool or v["id"] not in o:
                        continue
                    # reasons given on the version that HAS the eval-ish cue (for context the cue is the
                    # ABSENCE of personal context: eval-orig lacks it, deploy variant had it removed)
                    has_cue_vid = f"{v['id']}|orig" if pool == "eval" else v["vid"]
                    t = T.get((m, has_cue_vid))
                    if t is None:
                        continue
                    mentioned = c in t["tags"]
                    causal = abs(v["p"] - o[v["id"]]) >= 10
                    pairs.append((mentioned, causal, abs(v["p"] - o[v["id"]])))
            if not pairs:
                continue
            a = np.array(pairs, float)
            men, cau = a[:, 0].astype(bool), a[:, 1].astype(bool)
            rows[c] = {"n": len(a), "mention_rate": float(men.mean()), "causal_rate": float(cau.mean()),
                       "mean_abs_delta": float(a[:, 2].mean()),
                       "p_causal_given_mentioned": float(cau[men].mean()) if men.any() else None,
                       "p_mentioned_given_causal": float(men[cau].mean()) if cau.any() else None}
        # what do models say on original eval prompts overall?
        otr = {}
        for pool in ["eval", "deploy"]:
            tagc, n = collections.Counter(), 0
            for (mm, vid), t in T.items():
                if mm == m and vid.endswith("|orig") and t["pool"] == pool:
                    n += 1
                    tagc.update(set(t["tags"]))
            otr[pool] = {k: v / max(n, 1) for k, v in tagc.items()}
        svc[m] = {"cues": rows, "orig_tag_rates": otr}
        if len(rows) >= 3:
            svc[m]["spearman_mention_vs_effect"] = float(spearmanr([r["mention_rate"] for r in rows.values()],
                                                                   [r["mean_abs_delta"] for r in rows.values()]).correlation)

json.dump(S, open("results/summary.json", "w"), indent=1)

# ---- figures ----
for mode in ["num"]:
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, (pool, title) in zip(axes, [("eval", "Removed from eval items"), ("deploy", "Added to real-user prompts")]):
        w = 0.8 / max(len(models), 1)
        for k, m in enumerate(models):
            e = S["per_model"][m].get(mode, {}).get("effects", {})
            xs, ys, lo, hi = [], [], [], []
            for j, c in enumerate(CUES):
                r = e.get(f"{pool}:{c}")
                if r:
                    xs.append(j + (k - (len(models) - 1) / 2) * w); ys.append(r["cue_effect"])
                    lo.append(r["cue_effect"] - r["ci"][0]); hi.append(r["ci"][1] - r["cue_effect"])
            ax.bar(xs, ys, w, yerr=[lo, hi], capsize=2, label=m)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xticks(range(len(CUES))); ax.set_xticklabels([CUE_NAME[c] for c in CUES], rotation=30, ha="right", fontsize=8)
        ax.set_title(title, fontsize=10)
    axes[0].set_ylabel("Cue effect on P(eval) (pp)")
    axes[1].legend(fontsize=7)
    plt.tight_layout(); plt.savefig("figs/cue_effects.png", dpi=200); plt.close()

if "stated_vs_causal" in S:
    fig, ax = plt.subplots(figsize=(5, 4))
    for m in models:
        rows = S["stated_vs_causal"].get(m, {}).get("cues", {})
        xs = [r["mention_rate"] for r in rows.values()]; ys = [r["mean_abs_delta"] for r in rows.values()]
        ax.scatter(xs, ys, label=m)
        for c, r in rows.items():
            ax.annotate(c, (r["mention_rate"], r["mean_abs_delta"]), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("Stated: rate the cue is named as a reason"); ax.set_ylabel("Causal: mean |ΔP(eval)| when edited (pp)")
    ax.legend(fontsize=7); plt.tight_layout(); plt.savefig("figs/stated_vs_causal.png", dpi=200); plt.close()

print(json.dumps({m: {k: v for k, v in S["per_model"][m].get("num", {}).items() if k != "effects"} for m in models}, indent=1))
