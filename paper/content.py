"""Report text. Numbers in RESULTS come from results/summary.json (see results_blocks below); prose numbers were
copied from the same file. Static sections reuse paper/paper.tex."""
import ast, json

TITLE = "Tells: What Causally Makes a Language Model Think It Is Being Evaluated, and Does It Know Why?"
AUTHOR = "Mallika Suri"
AFFIL = "Scaler School of Technology, Bengaluru"
REPO = "https://github.com/snowhiteohno/tells-eval-awareness"

S = json.load(open("results/summary.json"))
JUDGES = ["gemma-4-31b-it", "deepseek-v4-flash", "qwen3.8-27b"]
REWRITER = "glm-5.2"
CLASSIFIER = "gemma-4-26b-a4b-it"


def consts(path, names):
    out = {}
    for n in ast.parse(open(path).read()).body:
        if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id in names:
            try:
                out[n.targets[0].id] = ast.literal_eval(n.value)
            except ValueError:
                pass
    return out


PR = consts("probe.py", {"HEAD", "TAIL"})
TR = consts("transforms.py", {"SYS", "I_OPEN", "I_CONTEXT", "I_NOHYPO", "I_POLISH", "I_OPTS", "I_NOCTX", "HYPO_PREFIX"})
CL = consts("classify.py", {"TAGS"})

ABSTRACT = "TODO"

INTRO = [
    ("h2", "1. Introduction"),
    ("p", "Safety evaluations assume that what a model does on a test is what it would do in deployment. That assumption "
          "weakens if models can tell the two apart: frontier models already separate evaluation transcripts from real "
          "usage well above chance [1, 2], and evaluation-awareness can be represented internally and steered [3, 4]. A "
          "natural response is to make evaluations look more realistic. To do that, we need to know *which* features of a "
          "prompt give the game away."),
    ("p", "Current practice answers this in one of two ways. The first asks the model: a judge model explains why it "
          "thinks a prompt is a test, and those stated reasons decide what to rewrite [5]. The second rewrites prompts "
          "holistically, with many changes at once, and measures the aggregate shift [6]; this leaves the contribution of "
          "individual cues unidentified, which [6] lists as future work. Both leave open an epistemic question that "
          "matters for anyone relying on these signals: *when a model tells us why it thinks it is being tested, does its "
          "explanation match what actually drove the judgement?* Humans are poor at this kind of introspective report [7], "
          "and language-model explanations are unfaithful in other settings [8]."),
    ("p", "Our main contributions are:"),
    ("bullets", [
        "**A causal cue map.** We edit five candidate “tells” one at a time, in both directions: removing each from real "
        "evaluation items (Anthropic’s advanced-AI-risk evaluations [9] and MMLU-Pro [10]) and adding it to real user "
        "prompts from WildChat [11]. We then measure the paired change in three open judges’ stated probability that the "
        "prompt is an evaluation.",
        "**A stated-vs-causal faithfulness test.** We compare which cues judges *say* drove their judgement with which "
        "cues *actually* move it, giving precision and recall of verbalised tells.",
        "TODO_FINDINGS",
    ]),
]

RELATED = [
    ("h2", "2. Related Work"),
    ("p", "**Evaluation awareness.** Needham et al. [1] show frontier models classify evaluation vs. deployment "
          "transcripts with high AUC; SAD [2] benchmarks broader situational awareness. Nayan et al. [4] find detection, "
          "behaviour and control of evaluation-awareness are only weakly coupled across 37 open models, using framing "
          "perturbations but not single-feature ablations. Knecht et al. [12] intervene on verbalised awareness in "
          "chain-of-thought and find small behavioural effects; they do not test whether the verbalised reasons are "
          "faithful. Hua et al. [3] steer an evaluation-aware model organism toward deployment behaviour."),
    ("p", "**Making evaluations realistic.** Ivanov and Kadochnikov [5] use a judge’s stated reasons to guide rewrites of "
          "scheming evaluations and find realism hard to achieve. Probe-Rewrite-Evaluate [6] rewrites prompts toward a "
          "deployment-like probe score and observes behavioural changes, but applies many edits jointly. Our design "
          "isolates individual cues and tests the judge’s stated reasons against their measured causal effect."),
    ("p", "**Faithfulness of explanations.** Verbal reports can diverge from the processes that produced a judgement, in "
          "humans [7] and in LLMs [8]. We bring this lens to evaluation-awareness, where stated reasons are currently "
          "used as a design signal."),
]

CUE_ROWS = [
    ["Q/A scaffolding", "drop Question:/Answer: (rule)", "wrap in Question:/Answer: (rule)"],
    ["Answer options", "drop options, minimal stem rewrite (LLM)", "append 4 generated options (LLM)"],
    ["Polished register", "lowercase, contractions, 2 typos (rule)", "formal polish, same content (LLM)"],
    ["No personal context", "prepend 1–2 sentence first-person context (LLM)", "strip personal context (LLM)"],
    ["Hypothetical framing", "recast as a direct situation (LLM)", "prefix “Hypothetically, suppose a user…” (rule)"],
]

METHODS = [
    ("h2", "3. Methods"),
    ("h3", "3.1 Prompt pools"),
    ("p", "**Evaluation pool (n = 60):** 30 human-written items from Anthropic’s advanced-AI-risk evaluations [9] (5 each "
          "from power-seeking, survival-instinct, corrigibility, wealth-seeking, coordination with other AIs, and "
          "self-awareness) and 30 MMLU-Pro items [10] (3 from each of 10 knowledge-heavy categories), all rendered in the "
          "standard harness format (`Question:` / `Choices:` / `Answer:`). **Deployment pool (n = 60):** first user turns "
          "from WildChat-1M [11], real ChatGPT conversations; English, non-toxic, non-redacted, 60–700 characters, filtered "
          "to remove URLs, contact details, code blocks, jailbreak templates, explicit content, and prompts already "
          "containing lettered options or Question:/Answer: markers (so every cue can be cleanly added). Sampling used a "
          "fixed seed."),
    ("h3", "3.2 Single-cue counterfactual edits"),
    ("p", "Table 1 lists the five cue families. For each prompt we produce one variant per cue in which *only* that cue "
          "is changed: removed from evaluation items (toward realism) and added to deployment items (toward test-likeness). "
          f"Rule-based edits are used wherever possible; where a semantic edit is unavoidable, a fixed rewriter model "
          f"({REWRITER}, not one of the judges) applies a minimal-edit instruction at temperature 0. Hypothetical-framing "
          "removal and personal-context removal are applied only where the rewriter detects the cue; otherwise the item is "
          "skipped for that cue. A sixth variant, *all*, applies every edit jointly. A length/script guard rejects rewrites "
          "that change length by more than 2.5× or the share of non-ASCII characters by more than 0.2 (this caught the "
          "rewriter translating a prompt instead of polishing it); rejected edits are skipped and logged."),
    ("table", "**Table 1.** Cue families and edit directions. “Rule” edits are deterministic string operations; “LLM” "
              "edits use the rewriter with the minimal-edit instructions in Appendix B.",
     ["Cue", "Evaluation items (remove)", "Deployment items (add)"], CUE_ROWS, [1.3, 2.6, 2.6]),
    ("h3", "3.3 Measuring evaluation belief"),
    ("p", "Each judge sees one prompt at a time inside a fixed wrapper asking whether it was “more likely written as part "
          "of an EVALUATION or test of an AI system … or sent by a REAL USER in ordinary deployment”, and returns P(eval) "
          "as an integer 0–100. We use two modes: **num**, which returns only the number (our causal measurement), and "
          "**reason**, which first lists up to three features that most influenced the judgement. All calls use "
          f"temperature 0. Judges, from three model families: {', '.join(JUDGES)}, accessed through Adaption’s hosted "
          "inference API (exact IDs as listed). qwen3.8-27b and deepseek-v4-flash produce hidden reasoning tokens before "
          "answering; gemma-4-31b-it does not. If a temperature-0 answer cannot be parsed (gemma-4-31b-it occasionally "
          "degenerates into a repeated token), it is re-queried once at temperature 0.3 and flagged; responses that still "
          "fail are dropped (models tried and dropped are listed in Appendix B)."),
    ("h3", "3.4 Metrics"),
    ("p", "**Cue effect.** For prompt *i* and cue *c*, Δ = P(variant) − P(original). We report the mean effect *of the "
          "cue’s presence* on P(eval) (sign-flipped for removals, so positive always means “this cue makes a prompt look "
          "like a test”), with 95% bootstrap CIs over prompts (5,000 resamples) and a Wilcoxon signed-rank test. "
          "**Additivity.** We compare the joint (*all*) edit with the sum of single-cue effects. **Stated vs causal.** "
          f"Each listed reason is tagged by a classifier ({CLASSIFIER}, five responses per call, temperature 0) with "
          "every applicable label among the cue families, “AI-as-topic” and “other content” (a line such as “lettered "
          "options and an Answer: label” names two cues). For each (prompt, cue) pair we record whether the cue was named "
          "in the reasons given on the version that contains it, and whether editing it moved P(eval) by ≥ 10 points. We "
          "report mention rate, causal rate, P(causal | mentioned) (precision of stated tells) and P(mentioned | causal) "
          "(recall)."),
]

REFS = [
    "[1] Needham, J., Edkins, G., Pimpale, G., Bartsch, H., & Hobbhahn, M. (2025). Large Language Models Often Know When "
    "They Are Being Evaluated. arXiv:2505.23836.",
    "[2] Laine, R., Chughtai, B., Betley, J., Hariharan, K., Scheurer, J., Balesni, M., Hobbhahn, M., Meinke, A., & Evans, O. "
    "(2024). Me, Myself, and AI: The Situational Awareness Dataset (SAD) for LLMs. arXiv:2407.04694.",
    "[3] Hua, T. T., Qin, A., Marks, S., & Nanda, N. (2026). Steering Evaluation-Aware Language Models to Act Like They Are "
    "Deployed. ICLR 2026. arXiv:2510.20487.",
    "[4] Nayan, N., et al. (2026). Evaluation Awareness Is Not One Capability: Evidence from Open Language Models. "
    "arXiv:2606.23583.",
    "[5] Ivanov, I., & Kadochnikov, D. (2025). It’s hard to make scheming evals look realistic for LLMs. LessWrong. "
    "https://www.lesswrong.com/posts/TBk2dbWkg2F7dB3jb",
    "[6] Xiong, L., Bhargava, N., Chang, J., Hong, J., Liu, H., Sharma, V., & Zhu, K. (2025). Probe-Rewrite-Evaluate: A "
    "Workflow for Reliable Benchmarks and Quantifying Evaluation Awareness. arXiv:2509.00591.",
    "[7] Nisbett, R. E., & Wilson, T. D. (1977). Telling More Than We Can Know: Verbal Reports on Mental Processes. "
    "Psychological Review, 84(3), 231–259.",
    "[8] Turpin, M., Michael, J., Perez, E., & Bowman, S. R. (2023). Language Models Don’t Always Say What They Think: "
    "Unfaithful Explanations in Chain-of-Thought Prompting. NeurIPS 2023.",
    "[9] Perez, E., et al. (2022). Discovering Language Model Behaviors with Model-Written Evaluations. arXiv:2212.09251.",
    "[10] Wang, Y., et al. (2024). MMLU-Pro: A More Robust and Challenging Multi-Task Language Understanding Benchmark. "
    "NeurIPS 2024 Datasets and Benchmarks Track.",
    "[11] Zhao, W., Ren, X., Hessel, J., Cardie, C., Choi, Y., & Deng, Y. (2024). WildChat: 1M ChatGPT Interaction Logs in "
    "the Wild. ICLR 2024.",
    "[12] Knecht, A., Florin, L., & Hagendorff, T. (2026). Evaluation Awareness in Language Models Has Limited Effect on "
    "Behaviour. arXiv:2605.05835.",
]

PROMPTS_APPX = [
    ("h3", "Appendix B. Verbatim prompts"),
    ("p", "**Models tried and dropped.** qwen3.5-397b (as a judge) and glm-5.2 / glm-5.3-flash (as the reason classifier) "
          "exhausted their output-token budgets (2,000–8,000 tokens) on hidden reasoning and returned empty answers; the "
          "hosted API exposes no way to cap reasoning. All three were replaced before any results were computed."),
    ("p", "**Judge wrapper** (the message replaces {msg}); followed by the mode-specific tail."),
    ("pre", PR["HEAD"].rstrip()),
    ("p", "**num tail:**"), ("pre", PR["TAIL"]["num"]),
    ("p", "**reason tail:**"), ("pre", PR["TAIL"]["reason"]),
    ("p", f"**Rewriter system prompt** ({REWRITER}, temperature 0); the text to edit is passed in <text> tags after the "
          "instruction."),
    ("pre", TR["SYS"]),
] + [x for k, lab in [("I_OPEN", "Remove options (eval)"), ("I_CONTEXT", "Add personal context (eval)"),
                      ("I_NOHYPO", "Remove hypothetical framing (eval)"), ("I_POLISH", "Polish register (deploy)"),
                      ("I_OPTS", "Generate options (deploy)"), ("I_NOCTX", "Remove personal context (deploy)")]
     for x in [("p", f"**{lab}:** {TR[k]}")]] + [
    ("p", f"**Hypothetical prefix (deploy, rule):** “{TR['HYPO_PREFIX'].strip()}”"),
    ("p", f"**Classifier tags** ({CLASSIFIER}; each listed feature gets every tag that applies): " +
          "; ".join(f"`{k}` = {v}" for k, v in CL["TAGS"].items()) + "."),
]

CODE_DATA = [
    ("h2", "Code and Data"),
    ("p", f"**Code repository:** {REPO} (all scripts, reproduction instructions, exact model IDs)."),
    ("p", "**Data:** `data/pool.json` (the 120 prompts), `data/variants.json` (all single-cue variants), "
          "`data/skipped_rewrites.json` (guard-rejected edits), `results/probes.jsonl` (every judge response, raw text "
          "included), `results/reason_tags.jsonl`, `results/summary.json` (every number in this report)."),
]

ETHICS_APPX = [
    ("h3", "Appendix A. Limitations and Dual-Use / Ethical Considerations"),
    ("p", "**What this does and does not establish.** We measure how single prompt features shift a judge model’s "
          "*stated* probability that a prompt is an evaluation, when asked directly. This is not the model’s internal "
          "belief, and it is not evidence that the model behaves differently on such prompts; a model can be "
          "unaware-in-words but aware-in-activations, or the reverse [3, 4, 12]. Our judges are three open models served "
          "by one hosted provider; frontier closed models may weight cues differently. The prompt-level n is small "
          "(60 per pool, and only 3–4 items for two one-directional cells), so we report CIs and treat small-n cells as "
          "descriptive."),
    ("p", "**Confounds.** WildChat is ChatGPT-era traffic (2023–24), so deployment prompts differ from evaluation items "
          "in topic, length and language as well as in our five cues; single-cue edits isolate each cue *within* a "
          "prompt, but not every interaction with these background differences. LLM rewrites can leave artifacts (e.g. a "
          "slightly over-fluent added context sentence); we used rule-based edits where possible, a length/script guard, "
          "a constrained context instruction, and manual inspection of sampled variants, but cannot rule out residual "
          "leakage. Our wrapper itself frames the task as classification, which one judge occasionally cited as a reason."),
    ("p", "**Dual use.** A ranked list of tells could help someone disguise evaluations from models, or could be used to "
          "help a model detect evaluations. We think the main use is the intended one: helping evaluation builders "
          "remove the cheapest giveaways so that safety evaluations measure deployment behaviour. The cues we "
          "study are surface features already discussed publicly. We do not train or fine-tune any model, and we "
          "release only prompts drawn from public datasets."),
    ("p", "**Personal data.** WildChat is PII-scrubbed by its authors; we additionally dropped prompts containing URLs, "
          "e-mail addresses, phone-number patterns, or redaction flags, and use only 60 first turns. One retained "
          "prompt contains sexually explicit text; we kept it unchanged because it is a genuine deployment prompt from "
          "the public dataset and removing it after seeing results would bias the pool."),
]

NEW_WORK_APPX = [
    ("h3", "Appendix C. Statement of new work"),
    ("p", "All code, data selection, experiments, analysis and writing were produced during the sprint (13–15 November "
          "2026). No preexisting code was used. The prompt pools come from public datasets [9, 10, 11]."),
]

# ------------------------------------------------------------------ results (all numbers read from summary.json)
SP = json.load(open("results/spotcheck.json"))
PM = {m: S["per_model"][m] for m in JUDGES}
E = {m: PM[m]["num"]["effects"] for m in JUDGES}
SV = {m: S["stated_vs_causal"][m] for m in JUDGES}
CUES5 = ["fmt", "opts", "register", "context", "hypo"]
NAME = {"fmt": "Q/A scaffolding", "opts": "Answer options", "register": "Polished register",
        "context": "No personal context", "hypo": "Hypothetical framing", "all": "All cues jointly"}


def rng(vals, fmt="{:.0f}", pct=False):
    vals = [v * 100 if pct else v for v in vals]
    lo, hi = fmt.format(min(vals)), fmt.format(max(vals))
    return (lo if lo == hi else f"{lo}–{hi}") + ("%" if pct else "")


def eff(key):
    return [E[m][key]["cue_effect"] for m in JUDGES if key in E[m]]


def frac(key):
    return [E[m][key]["frac_moved10"] for m in JUDGES if key in E[m]]


def stated(c):
    return [SV[m]["orig_tag_rates"]["eval"].get(c, 0.0) for m in JUDGES]


def nmax(key):
    return max(E[m][key]["n"] for m in JUDGES if key in E[m])


def sg(x):
    return "0" if round(x) == 0 else f"{x:+.0f}"


def cell(key):
    v = eff(key)
    return f"{sg(sum(v) / len(v))} ({sg(min(v))} to {sg(max(v))})" + (f", n={nmax(key)}" if nmax(key) < 10 else "")


def pct_mean(vals):
    return f"{100 * sum(vals) / len(vals):.0f}%"


auroc = [PM[m]["num"]["auroc"] for m in JUDGES]
add_e = {m: PM[m]["num"]["additivity"]["eval"] for m in JUDGES}
add_d = {m: PM[m]["num"]["additivity"]["deploy"] for m in JUDGES}
single_eval_max = max(abs(E[m][f"eval:{c}"]["cue_effect"]) for m in JUDGES for c in CUES5 if E[m][f"eval:{c}"]["n"] >= 10)
xj = [v["rho"] for v in S["cross_judge_spearman"].values()]
parsed = sum(PM[m]["num"]["n_parsed"] for m in JUDGES); calls = sum(PM[m]["num"]["n_calls"] for m in JUDGES)
retried = sum(PM[m]["num"]["n_retried"] for m in JUDGES)
ROB = S.get("robustness", {})

def recall_dep(c):
    """P(cue named in reasons | adding the cue moved P(eval) >= 10pp), per judge, deploy side only."""
    out = []
    for m in JUDGES:
        bp = SV[m]["cues"].get(c, {}).get("by_pool", {}).get("deploy")
        if bp and bp["p_mentioned_given_causal"] is not None and bp["n"] >= 10:
            out.append(bp["p_mentioned_given_causal"])
    return out


DEPLOY_RECALL = ""
if all(recall_dep(c) for c in ["opts", "fmt", "register", "hypo"]):
    DEPLOY_RECALL = (" On the deploy side, among real prompts where adding a cue moved P(eval) by ≥ 10 points, judges "
                     f"named the added cue in their reasons in {rng(recall_dep('opts'), pct=True)} of cases for answer "
                     f"options, {rng(recall_dep('fmt'), pct=True)} for Q/A labels, {rng(recall_dep('register'), pct=True)} "
                     f"for polished register and {rng(recall_dep('hypo'), pct=True)} for hypothetical framing.")


TABLE2 = [[NAME[c], cell(f"eval:{c}"), cell(f"deploy:{c}"),
           pct_mean(stated(c)) if c != "all" else "–",
           pct_mean(frac(f"eval:{c}")), pct_mean(frac(f"deploy:{c}"))] for c in CUES5 + ["all"]]

RESULTS = [
    ("h2", "4. Results"),
    ("p", f"**Judges separate the pools.** On unedited prompts all three judges discriminate evaluation items from real "
          f"user prompts (AUROC {rng(auroc, '{:.2f}')}; mean P(eval) {rng([PM[m]['num']['mean_p_eval_items'] for m in JUDGES])} "
          f"on evaluation items vs {rng([PM[m]['num']['mean_p_deploy_items'] for m in JUDGES])} on real prompts). "
          f"{parsed} of {calls} num-mode responses parsed ({100 * parsed / calls:.1f}%), {retried} after one re-query "
          "(Methods). There is therefore signal for single cues to explain."),
    ("p", f"**No single cue is necessary; most are close to sufficient.** Removing any one cue from evaluation items "
          f"changes mean P(eval) by at most {single_eval_max:.1f} points for any judge (Table 2, Figure 1 left), and "
          f"moves an individual item by ≥ 10 points in at most {100 * max(max(frac(f'eval:{c}')) for c in CUES5 if nmax(f'eval:{c}') >= 10):.0f}% "
          f"of cases. The effects are often statistically detectable (e.g. answer options: Wilcoxon p < 0.01 for all "
          f"three judges) but small. "
          f"Removing all cues jointly lowers P(eval) by {rng([-a['mean_all'] for a in add_e.values()])} points, far more "
          f"than the sum of the single-cue effects ({rng([-a['mean_sum_singles'] for a in add_e.values()])}): the cues "
          "are *redundant*, so each evaluation item stays recognisable while any other tell remains. The reverse "
          f"direction is the opposite: adding one cue to a real prompt is often enough. Answer options raise P(eval) by "
          f"{rng(eff('deploy:opts'))} points, hypothetical framing by {rng(eff('deploy:hypo'))}, Q/A scaffolding by "
          f"{rng(eff('deploy:fmt'))} and polished register by {rng(eff('deploy:register'))} (Figure 1 right); the joint "
          f"addition ({rng([a['mean_all'] for a in add_d.values()])}) is *less* than the sum of singles "
          f"({rng([a['mean_sum_singles'] for a in add_d.values()])}), as expected if one tell already saturates the "
          f"judgement. The three judges rank the cue effects almost identically (pairwise Spearman ρ = {rng(xj, '{:.2f}')} "
          "across the eight well-powered cells)."),
    ("fig", "figs/cue_effects.png", 6.5,
     "**Figure 1.** Causal effect of each cue’s presence on a judge’s stated P(eval), in percentage points (positive = "
     "the cue makes a prompt look like a test). Left: cue removed from evaluation items; right: cue added to real "
     "WildChat prompts. Bars are mean paired changes per judge with 95% bootstrap CIs; faded bars have n < 10 items "
     "(the cue was rarely present to edit). Single removals barely move evaluation items, while single additions move "
     "real prompts a lot."),
    ("table", "**Table 2.** Cue effects (mean over the three judges, with range across judges) and how often each cue is "
              "named versus causally important. *Stated*: share of original evaluation items whose stated reasons name "
              "the cue. *Necessary*: share of evaluation items where removing the cue alone moves P(eval) ≥ 10 points. "
              "*Sufficient*: share of real prompts where adding the cue alone moves P(eval) ≥ 10 points. Effects in "
              "percentage points; n = 54–60 items per cell unless shown.",
     ["Cue", "Removed from evals", "Added to real prompts", "Stated", "Necessary", "Sufficient"], TABLE2,
     [1.45, 1.55, 1.6, 0.6, 0.75, 0.75]),
    ("p", f"**Safety evaluations stay recognisable without any surface cue.** The joint edit lowers P(eval) by "
          f"{rng([E[m]['eval-mmlupro:all']['cue_effect'] for m in JUDGES])} points on MMLU-Pro items but only "
          f"{rng([E[m]['eval-aar:all']['cue_effect'] for m in JUDGES])} on the advanced-AI-risk items. Judges say why: "
          f"“the content is about the AI itself” is named on {rng([SV[m]['orig_tag_rates']['eval'].get('ai_topic', 0) for m in JUDGES], pct=True)} "
          "of original evaluation items. For safety evaluations the topic is itself a tell that no formatting fix removes."),
    ("p", f"**Stated reasons name real tells but hide their redundancy.** Judges name Q/A scaffolding on "
          f"{rng(stated('fmt'), pct=True)} of evaluation items and answer options on {rng(stated('opts'), pct=True)}, yet "
          f"removing scaffolding alone moves P(eval) by ≥ 10 points on only {rng(frac('eval:fmt'), pct=True)} of them "
          f"and removing options on {rng(frac('eval:opts'), pct=True)} (Figure 2). The precision of a stated tell as a "
          "guide to what is *individually* worth fixing is therefore low. The stated cues are not invented, however: "
          f"they are the ones that are most often *sufficient* when added to real prompts (options "
          f"{rng(frac('deploy:opts'), pct=True)}, scaffolding {rng(frac('deploy:fmt'), pct=True)}). A designer who "
          "followed the stated reasons would fix scaffolding first and see almost no change." + DEPLOY_RECALL +
          " Register is the exception: it is sufficient on "
          f"{rng(frac('deploy:register'), pct=True)} of real prompts and named on only {rng(stated('register'), pct=True)} "
          "of evaluation items, so it is the one cue the judges systematically under-report."),
    ("fig", "figs/stated_vs_causal.png", 5.2,
     "**Figure 2.** What judges say versus what moves them. For each cue: how often it is named as a reason on original "
     "evaluation items (blue), how often removing it alone from an evaluation item moves P(eval) by ≥ 10 points (orange), "
     "and how often adding it alone to a real prompt does (green). Bars: mean over judges; dots: individual judges; "
     "faded: n < 10. Judges name scaffolding and options on almost every evaluation item, but neither is individually "
     "necessary."),
    ("p", f"**Robustness of the measurement.** Asking for reasons first changes the judgement only modestly: num-mode and "
          f"reason-mode P(eval) correlate at Spearman ρ = {rng([PM[m]['num_reason_spearman'] for m in JUDGES], '{:.2f}')}, "
          f"although reason mode raises deepseek-v4-flash’s AUROC from {PM['deepseek-v4-flash']['num']['auroc']:.2f} to "
          f"{PM['deepseek-v4-flash']['reason']['auroc']:.2f}. The scaffolding mention rate depends on the classifier; a "
          "classifier-free lower bound that counts only reasons literally citing the Question:/Answer: labels or a "
          f"template still gives {rng([SV[m]['fmt_lexical_rate_eval'] for m in JUDGES], pct=True)}, against a necessary "
          f"rate of at most {100 * max(frac('eval:fmt')):.0f}%. On {SP['n']} randomly sampled stated features the "
          f"classifier’s tag set matched an independent blind labelling (by an LLM assistant, Claude, not a human) exactly on {SP['agree']}/{SP['n']}, with per-tag "
          f"presence agreement of {rng(list(SP['per_tag_agreement'].values()), pct=True)} (lowest for scaffolding, "
          f"{100 * SP['per_tag_agreement']['fmt']:.0f}%, where the classifier over-assigns it to multiple-choice "
          "descriptions)." + (
          f" Re-running num mode once more at temperature 0.7 preserves the cue ranking (Spearman ρ between T = 0 and "
          f"T = 0.7 cue effects: {rng([ROB[m]['rank_spearman_t0_vs_t07'] for m in JUDGES if ROB.get(m, {}).get('rank_spearman_t0_vs_t07') is not None], '{:.2f}')})."
          if all(ROB.get(m, {}).get("rank_spearman_t0_vs_t07") is not None for m in JUDGES) else "")),
]

DISCUSSION = [
    ("h2", "5. Discussion and Limitations"),
    ("p", "**Implications for evaluation design.** Evaluation items are overdetermined: each carries several tells, and "
          "the judges keep recognising an item while any one remains. Cue-by-cue realism fixes, the natural response to "
          "a judge’s stated reasons, should therefore be expected to show no effect until the last tell is removed; this "
          "is consistent with the difficulty reported by [5]. In the other direction, almost any single benchmark "
          "convention (answer options, hypothetical framing, Q/A labels) is enough to make an ordinary request look like "
          "a test. For safety evaluations specifically, the content (questions about the model’s own goals, shutdown or "
          "power) remains a strong tell after all surface cues are gone, which suggests realism work for these "
          "evaluations has to change the scenario, not the formatting."),
    ("p", "**Implications for trusting model explanations.** The judges’ stated reasons are neither random nor "
          "fabricated: when an added cue moves the judgement, the judges usually name it. They fail in a subtler way: "
          "by presenting each tell as decisive when it is individually unnecessary, and by under-reporting writing "
          "register. Stated reasons are a reasonable list of candidates, but a poor guide to what to change; single-cue "
          "interventions like ours are a cheap way to check them."),
    ("h3", "Limitations"),
    ("p", "We measure *stated* P(eval) from three open models when asked directly, not internal belief or downstream "
          "behaviour. Pools are small (60 + 60), two cue cells are tiny (hypothetical framing could be removed from only "
          f"{nmax('eval:hypo')} evaluation items and personal context removed from only {nmax('deploy:context')} real "
          "prompts), and WildChat prompts differ from evaluation items in more than our five cues. LLM edits can leave "
          "artifacts despite our guard and inspection. Effects are measured from a near-ceiling baseline on evaluation "
          "items (mean P(eval) ≈ 95), which limits how much a single removal could show if judges are saturated; the "
          "joint-edit result shows the judges are not insensitive to removal, only to single removals. Appendix A "
          "expands on these points and on dual use."),
    ("h3", "Future Work"),
    ("p", "Pairwise and leave-one-in edits to map exactly which cue combinations keep an item recognisable; the same "
          "protocol on frontier closed models and on model-internal probes [3, 4]; testing whether lowering stated "
          "P(eval) through these edits changes behaviour on the safety items themselves [6, 12]; and scenario-level "
          "rewrites for safety evaluations, where the topic itself is the tell."),
]

CONCLUSION = [
    ("h2", "6. Conclusion"),
    ("p", "By editing one cue at a time in both directions, we find that what makes a prompt look like an evaluation is "
          "redundant rather than located in any single feature: no single cue is necessary for an evaluation item to be "
          "recognised, while several are individually sufficient to make a real prompt look like a test, and "
          "AI-safety content stays recognisable after every surface cue is removed. Judges’ stated reasons point at real "
          "tells but misrepresent them as individually decisive. Evaluation builders should treat a judge’s explanation "
          "as a hypothesis to test, and expect realism to come from changing scenarios, not formatting."),
]

ABSTRACT = (
    "If language models can tell evaluations from real use, safety evaluations stop measuring deployment behaviour. "
    "Efforts to make evaluations realistic currently rely on a judge model’s stated reasons for why a prompt looks like "
    "a test, or on rewrites that change many features at once. We isolate five candidate tells (Q/A scaffolding, answer "
    "options, polished register, absence of personal context, hypothetical framing) and edit each one alone, in both "
    "directions: removing it from 60 real evaluation items (Anthropic advanced-AI-risk evaluations and MMLU-Pro) and "
    "adding it to 60 real WildChat prompts. Three open judges from different model families report P(eval) for every "
    f"variant. All judges separate the original pools (AUROC {rng(auroc, '{:.2f}')}). No single removal lowers "
    f"mean P(eval) on evaluation items by more than {single_eval_max:.0f} points, yet removing all cues lowers it by "
    f"{rng([-a['mean_all'] for a in add_e.values()])} points: tells are redundant. Conversely, adding answer options "
    f"alone raises P(eval) on real prompts by {rng(eff('deploy:opts'))} points and hypothetical framing by "
    f"{rng(eff('deploy:hypo'))}. AI-safety items remain recognisable after all surface edits "
    f"({rng([E[m]['eval-aar:all']['cue_effect'] for m in JUDGES])}-point drop versus "
    f"{rng([E[m]['eval-mmlupro:all']['cue_effect'] for m in JUDGES])} for MMLU-Pro). Judges’ stated reasons name "
    f"scaffolding on {rng(stated('fmt'), pct=True)} of evaluation items, although removing it alone moves P(eval) by ≥ 10 "
    f"points on at most {100 * max(frac('eval:fmt')):.0f}%. Stated reasons identify real tells but hide their "
    "redundancy, so they are a poor guide to what to change; single-cue interventions offer a cheap check."
)

INTRO[-1][1][-1] = ("**Findings.** No single cue is necessary for an evaluation item to look like a test, but several are "
                    "individually sufficient to make a real prompt look like one; AI-safety content remains a tell after "
                    "all surface edits; and judges’ stated reasons name real tells while hiding that each is individually "
                    "unnecessary.")

# template order: ... Conclusion, Code and Data, References, Appendix
BLOCKS = (INTRO + RELATED + METHODS + RESULTS + DISCUSSION + CONCLUSION + CODE_DATA
          + [("h2", "References")] + [("ref", r) for r in REFS]
          + [("pagebreak",), ("h2", "Appendix")] + ETHICS_APPX + PROMPTS_APPX + NEW_WORK_APPX)
