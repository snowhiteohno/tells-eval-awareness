# Tells: What Causally Makes a Language Model Think It Is Being Evaluated — and Does It Know Why?

Solo submission to the **AI x Epistemics Research Sprint** (Apart Research, 13–15 Nov 2026), Track 1: Model Epistemics & Decision-Making Evals.
**Author:** Mallika Suri, Scaler School of Technology, Bengaluru.

---

## TL;DR

If a model can tell an evaluation from real use, safety evals stop measuring real behaviour. To make evals realistic we need to know **which features give them away**. Existing work either trusts a judge model's *stated* reasons ([Ivanov & Kadochnikov, 2025](https://www.lesswrong.com/posts/TBk2dbWkg2F7dB3jb)) or rewrites prompts with many changes at once ([Probe-Rewrite-Evaluate, arXiv:2509.00591](https://arxiv.org/abs/2509.00591)).

This project does two things neither does:

1. **Single-cue counterfactuals, in both directions.** Remove one candidate "tell" at a time from real eval items, add it to real user prompts, and measure the paired change in the judge's P(eval).
2. **A stated-vs-causal faithfulness test.** Compare the cues a judge *says* drove its judgement with the cues that *causally* move it.

## Results

> ⏳ Runs in progress. Headline numbers, figures and the report PDF will be added here once `results/summary.json` is final. Every number in the report traces to that file.

## Design

### Prompt pools (`data/pool.json`, 120 prompts, fixed seed)

| Pool | n | Source |
|---|---|---|
| Eval | 30 | Anthropic advanced-AI-risk human-written evals: power-seeking, survival-instinct, corrigible-neutral-HHH, wealth-seeking, coordinate-other-ais, self-awareness (5 each) |
| Eval | 30 | MMLU-Pro, 3 × 10 knowledge-heavy categories, rendered as `Question:` / `Choices:` / `Answer:` |
| Deploy | 60 | WildChat-1M first user turns: English, non-toxic, non-redacted, filtered for URLs, contact details, code blocks, jailbreaks and existing MC/Q-A markers |

### Cues (each edited **alone**; `transforms.py`)

| Cue | Eval items: remove | Deploy items: add |
|---|---|---|
| Q/A scaffolding | drop `Question:`/`Answer:` (rule) | wrap in `Question:`/`Answer:` (rule) |
| Answer options | drop options, minimal stem rewrite (LLM) | append 4 generated options (LLM) |
| Polished register | lowercase, contractions, 2 typos (rule) | formal polish, same content (LLM) |
| No personal context | prepend 1–2 first-person sentences (LLM) | strip personal context (LLM) |
| Hypothetical framing | recast as a direct situation (LLM) | prefix "Hypothetically, suppose a user…" (rule) |
| All | every edit above jointly | every edit above jointly |

LLM edits use a fixed rewriter at temperature 0 with minimal-edit instructions. The rewriter is told the text is data and must never be executed. A length/script guard rejects rewrites that drift (e.g. the rewriter *translating* a prompt instead of polishing it). Rejected and skipped edits are logged in `data/skipped_rewrites.json`.

**Coverage note.** Two cues can only be edited in one direction for most items. Only 4/60 WildChat first turns carry personal context that can be stripped, and only 3/60 eval items are explicitly hypothetical. Those cells have very small n and are reported as such.

### Measurement (`probe.py`)

Each judge sees one prompt inside a fixed wrapper and returns `P(eval)` from 0 to 100, in two modes:
- **num**: the number only. This is the causal measure.
- **reason**: up to 3 features that drove the judgement, then the number. These are the stated tells.

Temperature 0. If a greedy answer can't be parsed (gemma occasionally degenerates into a repetition loop), it is re-queried once at T=0.3 and flagged `retried`.

### Analysis (`classify.py`, `analyze.py`)

- Per-cue paired effect of the cue's *presence* on P(eval), with 95% bootstrap CIs (5,000 resamples) and Wilcoxon signed-rank tests
- AUROC, eval vs deploy originals
- Additivity: joint edit vs sum of single-cue effects
- Agreement between num and reason modes
- **Stated vs causal:** each stated reason is tagged into a cue family (plus `ai_topic` and `content_other`). Reported: mention rate, causal rate (|Δ| ≥ 10 pp), P(causal | mentioned), P(mentioned | causal), and Spearman correlation across cues

## Models

All accessed via Adaption Labs hosted inference (Responses API), with exact model IDs:

| Role | Model ID | Family |
|---|---|---|
| Judge | `gemma-4-31b-it` | Google, non-reasoning |
| Judge | `deepseek-v4-flash` | DeepSeek, reasoning |
| Judge | `qwen3.8-27b` | Alibaba, reasoning |
| Rewriter + classifier | `glm-5.2` | Zhipu, not one of the judges |

`qwen3.5-397b` was tried as a judge and dropped: it spent its whole 2,000-token budget on hidden reasoning and returned no answer.

## Reproduce

```bash
uv venv -p 3.12 .venv
uv pip install -p .venv "adaption>=0.16" openai pandas pyarrow scipy scikit-learn matplotlib python-docx
echo 'ADAPTION_API_KEY=...' > .env        # gitignored; or set BASE_URL + API_KEY for any OpenAI-compatible endpoint

# data (only needed to rebuild data/pool.json, which is committed)
python prep_data.py 30                      # MMLU-Pro subset (needs data/mmlupro_test.parquet)
python build_items.py                       # -> data/pool.json

python transforms.py glm-5.2                # -> data/variants.json
python probe.py gemma-4-31b-it,deepseek-v4-flash,qwen3.8-27b --modes num,reason
python classify.py glm-5.2                  # -> results/reason_tags.jsonl
python analyze.py                           # -> results/summary.json, figs/
python paper/build_report.py                # -> paper/report.docx (Apart template)
```

Every model call is cached in `cache/` with its raw request, response and cost, so reruns are free and auditable. The Adaption key is rate-limited to roughly 25 requests per minute, so a full probe run (~4,300 calls) takes about 3 hours. The whole pipeline cost under $10 of inference.

## Repository layout

```
build_items.py, prep_data.py   build the prompt pools
transforms.py                  single-cue counterfactual variants
probe.py                       P(eval) from each judge, num + reason modes
classify.py                    tag stated reasons into cue families
analyze.py                     effects, CIs, faithfulness metrics, figures
llm.py                         cached Adaption / OpenAI-compatible client
data/                          pool.json, variants.json, skipped_rewrites.json
results/                       probes.jsonl, reason_tags.jsonl, summary.json
figs/                          cue_effects.png, stated_vs_causal.png
paper/                         report builder, content, LaTeX draft, refs.bib, report PDF
```

## Limitations (short version)

- Judges are open hosted models only. A stated P(eval) is not the model's internal belief.
- WildChat is ChatGPT-era traffic, and its prompts carry their own style differences from eval items.
- LLM rewrites may leave artifacts despite the guard and manual spot checks.
- Small n per cue, especially for the one-directional cues above.

The full discussion, including dual-use considerations, is in the report appendix.

## Data, licences and ethics

- Anthropic `model-written-evals`, `advanced-ai-risk/human_generated_evals` (CC-BY-4.0)
- MMLU-Pro, TIGER-Lab (MIT)
- WildChat-1M, AllenAI (ODC-BY). PII-scrubbed upstream and filtered further here. Only 60 prompts are used, stored in `data/pool.json`. One retained prompt contains sexually explicit text. It is kept as-is because it is a real deployment prompt from the public dataset.

**Dual use.** A ranked list of tells could help someone disguise tests from models, or help a model spot them. We expect it mainly helps eval builders make evaluations realistic, and we release only prompts from public datasets.

## Statement of new work

All code, data selection, experiments and writing were done during the sprint (13–15 Nov 2026). No preexisting code was used.
