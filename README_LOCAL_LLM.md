# Tech-Specs Agent — Local grounded Q&A (Ollama + LangGraph)

## New files (copy individually into project)

- `src/tech_specs/local_answer.py` — standard-library Ollama client, evidence prompt, citation checks
- `src/tech_specs/answer_graph.py` — new LangGraph workflow, without replacing `agent_graph.py`
- `scripts/ask_local.py` — command line entry point
- `tests/test_local_answer.py` — offline unit and graph tests

Existing BM25, Qdrant, RRF, and facet-based retrieval code is unchanged. No new Python dependencies are required if you have installed LangGraph and existing project extras.

## Install Ollama (Windows)

Download from https://ollama.com/download/windows. Then run:

```powershell
ollama pull qwen3:4b
ollama list
```

If you cannot reach the server, open the Ollama app or launch `ollama serve` in another terminal (only if not already running). To see GPU/CPU allocation, run `ollama ps` during generation. On a smaller-memory setup, `ollama pull qwen3:1.7b` and set `--model qwen3:1.7b`.

## Tests (do NOT require Ollama to be running)

```powershell
python -m pytest tests/test_local_answer.py -v
```

## First end-to-end questions

```powershell
python scripts/ask_local.py "What is TIM2 counter resolution on STM32L432KC?" --method bm25
python scripts/ask_local.py "How do the ADC and Brown-out Detector contribute to ATtiny25/45/85 sleep-mode power consumption, and what should be disabled if they are not needed?" --method bm25
python scripts/ask_local.py "How do the ADC and Brown-out Detector contribute to ATtiny25/45/85 sleep-mode power consumption, and what should be disabled if they are not needed?" --method hybrid
```

For a weaker GPU / lower latency, try `--model qwen3:1.7b --top-k 4`. Save the graph state with `--output data/processed/local_tim2.json` (will not overwrite existing output).

## Design and limitations

Graph: `START → plan → retrieve → assess → [generate if candidate exists for every facet] → validate citations → report → END`.

- Ollama is contacted at **127.0.0.1** only. There is no cloud API key or Python Ollama SDK.
- The LLM only receives bounded snippets of selected source chunks. `--num-ctx 4096` and `--num-predict 500` are defaults to limit memory and latency.
- A citation must be `[[chunk_id]]` and identify a retrieved chunk; every planned facet must have at least one *cited* selected chunk. Answers missing/using invalid citations are withheld.
- **Citation correctness is not semantic correctness.** An LLM can cite a real chunk for an incorrect statement; checking factual support requires further work. The rule-based facet planner also cannot detect all needed subquestions.
- Datasheet contents must not be interpreted as instructions to the LLM. Prompt mitigations help but are not a formal defense against all prompt injection.

Official API: https://docs.ollama.com/api/chat
