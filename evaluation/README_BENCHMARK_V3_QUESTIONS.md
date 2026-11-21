# Tech-Specs Agent — Benchmark v3 (source-anchored draft)

## What was created

Based **only** on the supplied `benchmark_v3_sources.json` (60 extracted source candidates, 30 per manufacturer), this draft contains **30 new technical queries**, **15 from each datasheet**, balanced into:

| Split | ATtiny25/45/85 | STM32L432xx | Total |
|---|---:|---:|---:|
| Development | 9 | 9 | 18 |
| Holdout test | 6 | 6 | 12 |
| Total | 15 | 15 | 30 |

Two queries require **two source chunks** and the source-anchored draft has **32 chunk references** in total. The questions cover exact identifiers, conceptual operation, register settings, power consumption, tables, and multi-evidence retrieval.

### Files and placement

Copy these files **individually** into your repository:

- `evaluation/questions_v3_dev.json` — 18 queries for development/tuning. Compatible with existing `evaluate_bm25.py`, `evaluate_vectors.py`, and `evaluate_hybrid.py`.
- `evaluation/questions_v3_test.json` — 12 reserved holdout queries. **Do not use to select hyperparameters.** Run once after freezing the approach.
- `evaluation/benchmark_v3_answer_key.json` — reference answers, original document/page/section, chunk IDs, and verbatim evidence excerpts for all 30 questions.
- `scripts/validate_benchmark_v3.py` — checks question IDs, split balance, exact excerpts, answer-key consistency and unchanged corpus SHA256.
- `evaluation/README_BENCHMARK_V3_QUESTIONS.md` — these instructions.
- `evaluation/benchmark_v3_catalog.md` — human-readable preview of questions, answers, and source references.

Keep your already-generated `evaluation/benchmark_v3_sources.json`, along with `data/processed/chunks.jsonl`, unchanged for this experiment.

## Validation

From the project root with the `.venv` activated:

```powershell
python scripts/validate_benchmark_v3.py
```

Expected output:

```text
Benchmark v3 source validation: PASS
  questions: 30
  development_questions: 18
  holdout_questions: 12
  source_anchors: 32
  multi_source_questions: 2
```

If the corpus was changed since source selection, validation should fail. In that case, re-export the sources and regenerate source anchors before running evaluations.

## Development evaluation — run these first

```powershell
python scripts/evaluate_bm25.py --questions evaluation/questions_v3_dev.json --output data/processed/bm25_v3_dev.json --top-k 5

python scripts/evaluate_vectors.py --questions evaluation/questions_v3_dev.json --output data/processed/vector_v3_dev.json --top-k 5

python scripts/evaluate_hybrid.py --questions evaluation/questions_v3_dev.json --output data/processed/hybrid_v3_dev.json --top-k 5
```

Then pool all top-5 candidates for a **new relevance audit**:

```powershell
python scripts/audit_retrieval.py --bm25 data/processed/bm25_v3_dev.json --vector data/processed/vector_v3_dev.json --hybrid data/processed/hybrid_v3_dev.json --output evaluation/judgments_v3_dev_unreviewed.csv
```

Send `evaluation/judgments_v3_dev_unreviewed.csv` for source-based relevance review, just as in the previous benchmark. Other retrieved chunks may be genuinely relevant even if absent from the initial anchor list.

## After development: holdout evaluation

Only **after** freezing retrieval settings / filters, run:

```powershell
python scripts/evaluate_bm25.py --questions evaluation/questions_v3_test.json --output data/processed/bm25_v3_test.json --top-k 5

python scripts/evaluate_vectors.py --questions evaluation/questions_v3_test.json --output data/processed/vector_v3_test.json --top-k 5

python scripts/evaluate_hybrid.py --questions evaluation/questions_v3_test.json --output data/processed/hybrid_v3_test.json --top-k 5
```

To audit their pooled results:

```powershell
python scripts/audit_retrieval.py --bm25 data/processed/bm25_v3_test.json --vector data/processed/vector_v3_test.json --hybrid data/processed/hybrid_v3_test.json --output evaluation/judgments_v3_test_unreviewed.csv
```

## Methodological cautions

1. **Only source anchors are labeled** in the question JSON files. Existing `Hit@5`, `Recall@5`, and `MRR@5` are **anchor-recovery diagnostics**, not exhaustive relevance estimates. A retrieval may return a different, valid chunk and still get a zero. For multi-evidence questions, `Hit@5` can pass with only one source; check anchor recall / coverage as well.
2. **Question generation was conditioned on extracted chunks**, so the holdout is only a tuning holdout, **not an independently authored external evaluation set**. An eventual independent test should be written without inspecting retrieved text or be collected from real user questions.
3. **All answers were checked against the supplied extracted text, not the manufacturer PDF pages**. Numeric tables and the LSE bypass description deserve manual verification against the source PDFs before treating them as engineering specifications.
4. Keep v2 reports intact. Do not compare 5-question v2 scores and 30-question v3 scores as if they were a controlled ablation: question difficulty and relevance-annotation completeness differ.
5. **Do not rebuild BM25/Qdrant or rewrite the corpus** merely to evaluate these files. The same corpus snapshot is intentionally reused.

## After this step

Revise the pooled dev judgments to establish multiple acceptable chunks per question. Re-evaluate BM25, vector, and RRF using graded relevance (e.g. nDCG@5), inspect failures by question category, then decide whether to improve filtering / indexing / reranking. Keep the test split untouched until the approach is frozen.
