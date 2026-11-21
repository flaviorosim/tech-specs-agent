# Tech-Specs Agent — Benchmark v3 source selection

Goal: prepare **new** questions for an independent, more diverse evaluation. The exporter does **not** generate questions or relevance labels: it selects real, previously unjudged source chunks from the current `chunks.jsonl` so questions can be authored and checked against evidence.

## Installation

No new dependencies. From the repository root, with your existing virtual environment active:

```powershell
python -m pytest tests/test_benchmark_sampling.py -v
python scripts/sample_benchmark_sources.py
```

The second command creates `evaluation/benchmark_v3_sources.json` containing up to 30 chunks from each document (60 total), with a mix of text and tables and coverage across pages. If the output already exists, the script refuses to overwrite it. Pick another `--output` path to regenerate.

**Send `evaluation/benchmark_v3_sources.json` to ChatGPT.** It includes the chunk content required to draft 20–30 grounded questions. You do **not** need to answer or label the questions yourself.

## Sampling constraints

- Reads `data/processed/chunks.jsonl` unchanged.
- Excludes all chunk IDs present in `evaluation/judgments_v2_reviewed.csv`.
- Uses a fixed seed to select chunks reproducibly.
- Avoids obvious TOC entries, very short text, incomplete tables and initial PDF pages.
- Prefers diverse sections and page ranges, and mixes text and tables.
- Does not truncate any selected chunk; chunks longer than 2,500 characters are skipped.
- Includes a SHA-256 of the corpus, for version tracking.

The filtering is heuristic. **A selected chunk is a source candidate, not a verified relevant answer.** The next step is to review candidate passages, write questions that are actually supported by them, and then pool and label retrieved results from all retrievers. Keep some new questions as a holdout set, not used to tune fusion parameters. Inspect original PDFs when tables or figures are ambiguous.
