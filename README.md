# IBM RAG and Agentic AI Capstone Project

This repository contains hands-on lab work for IBM's RAG and Agentic AI capstone, focused on transforming unstructured restaurant descriptions into structured data that can later be used for retrieval and agent workflows.

## Project Structure

- `pyproject.toml`: Python project metadata and dependencies.
- `Lab-Structure-Unstructured-Restaurant-Data-with-an-LLM/main.ipynb`: Notebook used for the extraction workflow.
- `Lab-Structure-Unstructured-Restaurant-Data-with-an-LLM/California-Culinary-Map.txt`: Unstructured source text.
- `Lab-Structure-Unstructured-Restaurant-Data-with-an-LLM/structured_restaurant_data.json`: Structured output dataset (210 records).
- `Lab-Structure-Unstructured-Restaurant-Data-with-an-LLM/M1L1_structure_for_loop.png`: Supporting lab image.

## Environment

This project targets Python `3.11.*` and currently uses:

- `ibm-watsonx-ai==1.4.7`
- `matplotlib==3.10.7`
- `numpy==2.3.4`

Install dependencies with your preferred toolchain. For example:

```bash
pip install -e .
```

## Quick Start

1. Open `Lab-Structure-Unstructured-Restaurant-Data-with-an-LLM/main.ipynb`.
2. Run notebook cells in order to process the unstructured restaurant corpus.
3. Review the generated structured data in `Lab-Structure-Unstructured-Restaurant-Data-with-an-LLM/structured_restaurant_data.json`.

## Current Lab Output

- Source corpus: California restaurant narratives in plain text.
- Structured result: JSON records with fields such as name, location, cuisine/style, signatures, ratings, and price range.
- Dataset size: 210 restaurant entries.