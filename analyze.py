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
        # key "eval-aar:opts" etc. splits the eval pool by source (safety evals vs MMLU-Pro)
        for key, pool, src, sign in [("eval", "eval", None, -1), ("deploy", "deploy", None, +1),
                                     ("eval-aar", "eval", "aar", -1), ("eval-mmlupro", "eval", "mmlupro", -1)]:
            for c in CUES:
                deltas = [v["p"] - o[v["id"]] for v in d.values()
                          if v["cue"] == c and v["pool"] == pool and v["id"] in o
                          and (src is None or v["source"].startswith(src))]
                if not deltas:
                    continue
                # "toward eval" effect of the cue itself: for eval pool we removed it (so cue effect = -delta)
                ce = [sign * x for x in deltas]
                try:
                    pval = float(wilcoxon(deltas).pvalue) if any(deltas) else 1.0
                except ValueError:
                    pval = 1.0
                eff[f"{key}:{c}"] = {"n": len(deltas), "mean_delta": float(np.mean(deltas)),
                                     "cue_effect": float(np.mean(ce)), "ci": boot(ce), "p": pval,
                                     "frac_moved10": float(np.mean(np.abs(deltas) >= 10))}
        res["effects"] = eff
        # additivity: joint edit vs sum of single-cue deltas, per pool
        res["additivity"] = {}
        for pool in ["eval", "deploy"]:
            add = []
            for i in o:
                if pools[i] != pool or f"{i}|all" not in d:
                    continue
                singles = [d[f"{i}|{c}"]["p"] - o[i] for c in ["fmt", "opts", "register", "context", "hypo"] if f"{i}|{c}" in d]
                add.append((d[f"{i}|all"]["p"] - o[i], sum(singles), max(singles, key=abs) if singles else 0))
            if add:
                a = np.array(add)
                res["additivity"][pool] = {"n": len(a), "mean_all": float(a[:, 0].mean()),
                                           "mean_sum_singles": float(a[:, 1].mean()),
                                           "mean_largest_single": float(a[:, 2].mean()), "ci_all": boot(a[:, 0])}
        sm[mode] = res
    # num vs reason agreement on the same prompts
    dn, dr = get(m, "num"), get(m, "reason")
    common = sorted(set(dn) & set(dr))
    if len(common) > 5:
        x = [dn[v]["p"] for v in common]; y = [dr[v]["p"] for v in common]
        sm["num_reason_spearman"] = float(spearmanr(x, y).correlation)
        sm["num_reason_mean_diff"] = float(np.mean(np.array(y) - np.array(x)))

# cells used for rank comparisons: single cues with enough items (drops the n<10 one-directional cells)
CELLS = [f"{p}:{c}" for p in ["eval", "deploy"] for c in CUES if c != "all"]


def ranked(e):
    return {k: e[k]["cue_effect"] for k in CELLS if k in e and e[k]["n"] >= 10}


# ---- do the judges agree on which cues matter? ----
S["cross_judge_spearman"] = {}
for a_, b_ in [(x, y) for i, x in enumerate(models) for y in models[i + 1:]]:
    ea, eb = ranked(S["per_model"][a_].get("num", {}).get("effects", {})), ranked(S["per_model"][b_].get("num", {}).get("effects", {}))
    ks = sorted(set(ea) & set(eb))
    if len(ks) >= 4:
        S["cross_judge_spearman"][f"{a_} vs {b_}"] = {"n_cells": len(ks), "rho": float(spearmanr([ea[k] for k in ks], [eb[k] for k in ks]).correlation)}

# ---- robustness: temperature-0.7 samples (probe.py --samples K; sample 0 is the T=0 run) ----
S["robustness"] = {}
for m in models:
    rows = [r for r in P if r["model"] == m and r["mode"] == "num" and r["sample"] > 0 and r["p"] is not None]
    if not rows:
        continue
    by = collections.defaultdict(list)
    for r in rows:
        by[r["vid"]].append(r)
    mean_p = {vid: float(np.mean([r["p"] for r in rs])) for vid, rs in by.items()}
    meta = {vid: rs[0] for vid, rs in by.items()}
    o = {meta[v]["id"]: p for v, p in mean_p.items() if meta[v]["cue"] == "orig"}
    e = {}
    for pool, sign in [("eval", -1), ("deploy", +1)]:
        for c in CUES:
            ds = [sign * (p - o[meta[v]["id"]]) for v, p in mean_p.items()
                  if meta[v]["cue"] == c and meta[v]["pool"] == pool and meta[v]["id"] in o]
            if ds:
                e[f"{pool}:{c}"] = {"n": len(ds), "cue_effect": float(np.mean(ds)), "ci": boot(ds)}
    e0, e7 = ranked(S["per_model"][m]["num"]["effects"]), ranked(e)
    ks = sorted(set(e0) & set(e7))
    sds = [float(np.std([r["p"] for r in rs], ddof=1)) for rs in by.values() if len(rs) > 1]
    S["robustness"][m] = {"n_samples": max(r["sample"] for r in rows), "effects_t07": e,
                          "mean_within_prompt_sd": float(np.mean(sds)) if sds else None,
                          "rank_spearman_t0_vs_t07": float(spearmanr([e0[k] for k in ks], [e7[k] for k in ks]).correlation) if len(ks) >= 4 else None,
                          "top_cue_t0": max(e0, key=e0.get) if e0 else None, "top_cue_t07": max(e7, key=e7.get) if e7 else None}

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
            for j, c in enumerate(CUES):
                r = e.get(f"{pool}:{c}")
                if r:  # cells with n < 10 are drawn faded: too few items to read much into
                    ax.bar(j + (k - (len(models) - 1) / 2) * w, r["cue_effect"], w, color=f"C{k}",
                           alpha=1.0 if r["n"] >= 10 else 0.35, label=m if j == 0 else None,
                           yerr=[[r["cue_effect"] - r["ci"][0]], [r["ci"][1] - r["cue_effect"]]], capsize=2)
        ns = {c: max((S["per_model"][m].get(mode, {}).get("effects", {}).get(f"{pool}:{c}", {}).get("n", 0)
                      for m in models), default=0) for c in CUES}
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xticks(range(len(CUES))); ax.set_xticklabels([f"{CUE_NAME[c]}\n(n={ns[c]})" for c in CUES], rotation=30, ha="right", fontsize=8)
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
