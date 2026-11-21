# Experimental evidence-facet retrieval

This is a controlled, **manually provided** query plan. The program does not automatically decide which entities or facts the question needs. A future LLM planner may generate plans after validation. The interleaved results are *candidate evidence*, not necessarily sufficient or correct answers.

## Installation (files individually)

Place the files in the existing project:

- `facet_retrieval.py` -> `src/tech_specs/facet_retrieval.py`
- `search_facets.py` -> `scripts/search_facets.py`
- `test_facet_retrieval.py` -> `tests/test_facet_retrieval.py`
- `attiny_sleep_facets.json` -> `evaluation/plans/attiny_sleep_facets.json`

Keep the existing `BM25Retriever`, `VectorRetriever`, `HybridRetriever`, corpus, and Qdrant index unchanged. This step adds no dependencies.

## Test

```powershell
python -m pytest tests/test_facet_retrieval.py -v
```

## BM25: coverage experiment

```powershell
python scripts/search_facets.py --method bm25 --plan evaluation/plans/attiny_sleep_facets.json --top-k 6 --expected attiny85_p036_c004 --expected attiny85_p036_c006
```

Both `--expected` IDs are *reference anchors* used only for **post-hoc evaluation**. They do not influence queries, retrieval, or ranking.

## Hybrid comparison

Close other processes using Qdrant Local, then run:

```powershell
python scripts/search_facets.py --method hybrid --plan evaluation/plans/attiny_sleep_facets.json --top-k 6 --expected attiny85_p036_c004 --expected attiny85_p036_c006
```

To save an output report, append:

```powershell
--output data/processed/attiny_sleep_facets_bm25.json
```

Choose a distinct output filename per run. The program refuses to overwrite an existing report.

## Limitations

- The original question “which two peripherals” is under-specified: the source also mentions the Watchdog Timer. This controlled example explicitly names **ADC** and **BOD** to test retrieval coverage.
- Results are round-robin by facet. This is **not RRF**, a semantic relevance model, or automatic evidence verification.
- The benchmark v3 development/holdout data must not be rewritten to hide previously observed misses.
