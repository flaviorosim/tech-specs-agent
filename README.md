# Tech Specs Agent

An Agentic Retrieval-Augmented Generation (RAG) system
for querying technical hardware documentation.

## Goals

- Extract structured content from technical PDFs.
- Implement hybrid retrieval using BM25 and embeddings.
- Build an agentic workflow using LangGraph.
- Evaluate retrieval and answer quality.

## Setup

Requires Python 3.11+.

```powershell
python -m pip install -e .
``` 

## Download documentation


```powershell
python scripts/download_docs.py
``` 

Official PDF sources are defined in scripts/download_docs.py.