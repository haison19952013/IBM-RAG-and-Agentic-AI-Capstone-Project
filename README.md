# IBM RAG and Agentic AI Capstone Project

This repository contains hands-on lab work for IBM's RAG and Agentic AI capstone. It includes notebook-based workflows for structuring unstructured text and processing multimodal recipe data with LLMs.

## Project Structure

- `pyproject.toml`: Python project metadata and dependencies.
- `README.md`: Project overview and usage guide.

### Lab 1: Structure Unstructured Restaurant Data

- `Lab-Structure-Unstructured-Restaurant-Data-with-an-LLM/main.ipynb`: Main notebook for extracting structured fields from raw restaurant text.
- `Lab-Structure-Unstructured-Restaurant-Data-with-an-LLM/California-Culinary-Map.txt`: Unstructured source corpus.
- `Lab-Structure-Unstructured-Restaurant-Data-with-an-LLM/structured_restaurant_data.json`: Structured output dataset (210 records).
- `Lab-Structure-Unstructured-Restaurant-Data-with-an-LLM/M1L1_structure_for_loop.png`: Supporting lab image.

### Lab 2: Process Multimodal Data with LLMs

- `Lab-Process-Multimodal-Data-with-LLMs/main.ipynb`: Main multimodal processing notebook.
- `Lab-Process-Multimodal-Data-with-LLMs/Recipes.json`: Base recipe dataset.
- `Lab-Process-Multimodal-Data-with-LLMs/Synthetic-User-Reviews.json`: Base user review dataset.
- `Lab-Process-Multimodal-Data-with-LLMs/augmented_food_recipe.json`: Augmented recipe output.
- `Lab-Process-Multimodal-Data-with-LLMs/augmented_user_reviews.json`: Augmented review output.
- `Lab-Process-Multimodal-Data-with-LLMs/M1L2_caption_all_recipes.png`: Supporting lab image.

### Python Script Export

- `py_files/Lab-Structure-Unstructured-Restaurant-Data-with-an-LLM.py`: Script version of the structured-data lab workflow.

## Environment

This project targets Python `3.11.*` and currently uses:

- `ibm-watsonx-ai==1.4.7`
- `matplotlib==3.10.7`
- `numpy==2.3.4`

Install dependencies with your preferred toolchain:

```bash
pip install -e .
```

## Quick Start

1. Open one of the lab notebooks:
	 - `Lab-Structure-Unstructured-Restaurant-Data-with-an-LLM/main.ipynb`
	 - `Lab-Process-Multimodal-Data-with-LLMs/main.ipynb`
2. Run cells in order.
3. Review generated JSON outputs in the same lab folder.

## Outputs

- Restaurant structuring output: 210 entries in `Lab-Structure-Unstructured-Restaurant-Data-with-an-LLM/structured_restaurant_data.json`.
- Multimodal augmentation outputs:
	- `Lab-Process-Multimodal-Data-with-LLMs/augmented_food_recipe.json`
	- `Lab-Process-Multimodal-Data-with-LLMs/augmented_user_reviews.json`