"""Report text. Numbers in RESULTS come from results/summary.json (see results_blocks below); prose numbers were
copied from the same file. Static sections reuse paper/paper.tex."""
import ast, json

TITLE = "Tells: What Causally Makes a Language Model Think It Is Being Evaluated — and Does It Know Why?"
AUTHOR = "Mallika Suri"
AFFIL = "Scaler School of Technology, Bengaluru"
REPO = "https://github.com/snowhiteohno/tells-eval-awareness"

S = json.load(open("results/summary.json"))
JUDGES = ["gemma-4-31b-it", "deepseek-v4-flash", "qwen3.8-27b"]
REWRITER = "glm-5.2"


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
        "**A causal cue map.** We edit five candidate “tells” one at a time, in both directions — removing each from real "
        "evaluation items (Anthropic’s advanced-AI-risk evaluations [9] and MMLU-Pro [10]) and adding it to real user "
        "prompts from WildChat [11] — and measure the paired change in three open judges’ stated probability that the "
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
          "answering; gemma-4-31b-it does not."),
    ("h3", "3.4 Metrics"),
    ("p", "**Cue effect.** For prompt *i* and cue *c*, Δ = P(variant) − P(original). We report the mean effect *of the "
          "cue’s presence* on P(eval) (sign-flipped for removals, so positive always means “this cue makes a prompt look "
          "like a test”), with 95% bootstrap CIs over prompts (5,000 resamples) and a Wilcoxon signed-rank test. "
          "**Additivity.** We compare the joint (*all*) edit with the sum of single-cue effects. **Stated vs causal.** "
          f"Each listed reason is tagged by a classifier ({REWRITER}, batched, temperature 0) into one of the cue "
          "families, “AI-as-topic”, or “other content”. For each (prompt, cue) pair we record whether the cue was named "
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
    ("p", "**Classifier tags** (each listed feature gets exactly one): " +
          "; ".join(f"`{k}` = {v}" for k, v in CL["TAGS"].items()) + "."),
]
