# Tells: what causally makes an LLM think it's being evaluated — and does it know why?

Submission to the **AI x Epistemics Research Sprint** (Apart Research, 13–15 Nov 2026). Solo project by Mallika Suri (Scaler School of Technology).

**Question.** If models can tell evaluations apart from real use, safety evals stop measuring real behaviour. To make evals realistic we need to know *which* features give them away. Current practice either (a) asks a judge model why it thinks a prompt is a test and trusts its stated reasons, or (b) rewrites prompts with many changes at once. We instead:

1. **Edit one cue at a time, in both directions.** Remove each candidate "tell" from real eval items (Anthropic advanced-AI-risk evals + MMLU-Pro) and add it to real user prompts (WildChat), then measure the paired change in the judge's stated P(eval).
2. **Test whether stated reasons are faithful.** Compare the cues a model *says* drove its judgement with the cues that *causally* move it.

Cues: Q/A scaffolding · multiple-choice options · polished register · absence of personal context · hypothetical framing (plus all edits jointly).

## Results
_Filled in after the runs; see `results/summary.json` and `figs/`._

## Reproduce
```bash
pip install openai adaption pandas pyarrow scipy scikit-learn matplotlib
python prep_data.py 30          # MMLU-Pro subset (needs data/mmlupro_test.parquet)
python build_items.py           # eval + deployment pools -> data/pool.json
export ADAPTION_API_KEY=...     # or BASE_URL + API_KEY for any OpenAI-compatible endpoint
python transforms.py <rewriter-model>          # single-cue variants -> data/variants.json
python probe.py <judge1>,<judge2> --modes num,reason
python classify.py <classifier-model>          # tag stated reasons
python analyze.py                              # -> results/summary.json, figs/
```
All model calls use temperature 0 and are cached in `cache/` (raw request + response), so reruns are free and auditable.

## Data sources
- Anthropic model-written-evals, `advanced-ai-risk/human_generated_evals` (CC-BY-4.0)
- MMLU-Pro (TIGER-Lab, MIT)
- WildChat-1M (AllenAI, ODC-BY); English, non-toxic, non-redacted first turns, additionally filtered to drop URLs, contact details and explicit content. Only 60 prompts are used and they are stored in `data/pool.json`.

## Statement of new work
Everything in this repo was written during the sprint. No preexisting code was used.
