# IBM RAG and Agentic AI Capstone Project

Hands-on lab repository for IBM's RAG and Agentic AI capstone.

The project walks through three connected workflows:

1. Structure unstructured restaurant text with an LLM.
2. Enrich recipe and review data with multimodal (image + text) processing.
3. Manage structured restaurant records through a command-line interface.

## Repository Layout

```text
.
├── pyproject.toml
├── README.md
├── Lab-Structure-Unstructured-Restaurant-Data-with-an-LLM/
│   ├── main.ipynb
│   ├── California-Culinary-Map.txt
│   ├── structured_restaurant_data.json
│   ├── README.md
│   └── M1L1_structure_for_loop.png
├── Lab-Process-Multimodal-Data-with-LLMs/
│   ├── main.ipynb
│   ├── Recipes.json
│   ├── Synthetic-User-Reviews.json
│   ├── augmented_food_recipe.json
│   ├── augmented_user_reviews.json
│   ├── README.md
│   └── M1L2_caption_all_recipes.png
├── Lab-Build-a-Command-Line-Data-Management-UI-for-Restaurant-Data/
│   ├── main.py
│   ├── README.md
│   └── M1L3_new_data_entry_process.png
└── py_files/
		├── Lab-Structure-Unstructured-Restaurant-Data-with-an-LLM.py
		├── Lab-Process-Multimodal-Data-with-LLMs.py
		└── Lab-Build-a-Command-Line-Data-Management-UI-for-Restaurant-Data.py
```

## Environment

- Python: `3.11.*`
- Dependencies:
	- `ibm-watsonx-ai==1.4.7`
	- `matplotlib==3.10.7`
	- `numpy==2.3.4`

Install dependencies:

```bash
pip install -e .
```

## How To Run

### Notebook Labs

Open and run these notebooks cell-by-cell:

- `Lab-Structure-Unstructured-Restaurant-Data-with-an-LLM/main.ipynb`
- `Lab-Process-Multimodal-Data-with-LLMs/main.ipynb`

### CLI Lab

Run the command-line restaurant data manager:

```bash
python Lab-Build-a-Command-Line-Data-Management-UI-for-Restaurant-Data/main.py
```

## Key Outputs

- Structured restaurant dataset:
	- `Lab-Structure-Unstructured-Restaurant-Data-with-an-LLM/structured_restaurant_data.json`
- Multimodal augmented datasets:
	- `Lab-Process-Multimodal-Data-with-LLMs/augmented_food_recipe.json`
	- `Lab-Process-Multimodal-Data-with-LLMs/augmented_user_reviews.json`

## Notes

- Each lab folder has a dedicated `README.md` with assignment details and expected deliverables.
- `py_files/` contains `.py` exports of the lab workflows.