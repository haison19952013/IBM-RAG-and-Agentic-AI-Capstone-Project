# IBM RAG and Agentic AI Capstone Project

This repository contains hands-on lab work for IBM's RAG and Agentic AI capstone. It includes notebook-based workflows for structuring unstructured text and processing multimodal recipe data with LLMs.

## Project Structure

```text
.
├── pyproject.toml
├── README.md
├── Lab-Process-Multimodal-Data-with-LLMs/
│   ├── main.ipynb
│   ├── Recipes.json
│   ├── Synthetic-User-Reviews.json
│   ├── augmented_food_recipe.json
│   ├── augmented_user_reviews.json
│   └── M1L2_caption_all_recipes.png
├── Lab-Structure-Unstructured-Restaurant-Data-with-an-LLM/
│   ├── main.ipynb
│   ├── California-Culinary-Map.txt
│   ├── structured_restaurant_data.json
│   └── M1L1_structure_for_loop.png
└── py_files/
    ├── Lab-Process-Multimodal-Data-with-LLMs.py
	└── Lab-Structure-Unstructured-Restaurant-Data-with-an-LLM.py
```

- pyproject.toml: Python project metadata and dependencies.
- README.md: Project overview and usage guide.
- Lab-Structure-Unstructured-Restaurant-Data-with-an-LLM/main.ipynb: Notebook for extracting structured fields from raw restaurant text.
- Lab-Process-Multimodal-Data-with-LLMs/main.ipynb: Notebook for multimodal data processing and augmentation.
- py_files/Lab-Process-Multimodal-Data-with-LLMs.py: Script export of the multimodal workflow.
- py_files/Lab-Structure-Unstructured-Restaurant-Data-with-an-LLM.py: Script export of the structured-data workflow.

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