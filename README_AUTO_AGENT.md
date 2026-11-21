# Tech-Specs Agent: automatic retrieval workflow (LangGraph v0)

This stage is an **agent workflow for candidate retrieval**, not yet a generative Q&A bot.
It uses rule-based planning (no LLM) and existing BM25, vector, or hybrid search.

## Install files in the project

- `src/tech_specs/auto_planner.py` — new
- `src/tech_specs/agent_graph.py` — new
- `scripts/run_agent.py` — new
- `tests/test_auto_agent.py` — new

No changes are necessary in `facet_retrieval.py` or the existing indices.

### Dependency

Inside `pyproject.toml`, under `[project.optional-dependencies]`, add:

```toml
agent = [
    "langgraph>=1.2,<2",
]
```

Keep other extras already in the project. Then install:

```powershell
python -m pip install -e ".[dev,retrieval,vectors,agent]"
```

### Test

```powershell
python -m pytest tests/test_auto_agent.py -v
```

### Run with lexical search

```powershell
python scripts/run_agent.py "How do the ADC and Brown-out Detector contribute to ATtiny25/45/85 sleep-mode power consumption, and what should be disabled if they are not needed?" --method bm25 --expected attiny85_p036_c004 --expected attiny85_p036_c006
```

### Run with hybrid search

```powershell
python scripts/run_agent.py "How do the ADC and Brown-out Detector contribute to ATtiny25/45/85 sleep-mode power consumption, and what should be disabled if they are not needed?" --method hybrid --expected attiny85_p036_c004 --expected attiny85_p036_c006
```

### Other examples

```powershell
python scripts/run_agent.py "What is TIM2 counter resolution on STM32L432KC?" --method bm25
python scripts/run_agent.py "How does the EEPROM work on ATtiny85?" --method bm25 --output data/processed/agent_eeprom_report.json
```

The `--expected` labels are used only for the post-search coverage check, never for planning or retrieval.

## How it works

LangGraph: `START -> plan -> retrieve -> assess -> report -> END`.

- `plan`: recognize explicitly mentioned technical topics (`ADC`, `BOD`, `watchdog`, etc.) and the device; create facets for two or more recognized topics. A single or unknown topic is searched unchanged.
- `retrieve`: delegate to the previously tested `search_facets()` function, which interleaves facet lists and deduplicates chunks.
- `assess`: check whether each facet received candidates. This does **not** verify technical correctness or completeness.
- `report`: list evidence chunk IDs, page numbers, and facet names. It does **not** generate an answer.

**Known limitations:** The rules recognize a small glossary and can miss synonyms or over-split some questions. They do not determine whether retrieved text supports a claim. Cross-device comparisons are deliberately rejected if both device families are mentioned. The answer generation and semantic evidence verification stages come later.

## Reproducibility note

The model development environment here did not have LangGraph installed, and installing it was blocked by network access. The Python files are syntax-checked; planner tests can run without LangGraph. The actual graph integration tests run on your machine once `langgraph` is installed.
