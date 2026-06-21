import glob
import json
import shutil
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import torch
from PIL import Image
from langchain_chroma import Chroma
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer
from transformers import CLIPModel, CLIPProcessor


def log(message: str) -> None:
    print(f"[INFO] {message}")


def download_and_extract_images(base_dir: Path) -> list[str]:
    zip_url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/5_Rr6ohviItzucyWk6nkrw/synthetic-recipe-images.zip"
    zip_path = base_dir / "synthetic-recipe-images.zip"
    image_dir = base_dir / "recipe_images"

    if not zip_path.exists():
        log("Downloading synthetic image dataset...")
        urlretrieve(zip_url, zip_path)

    if not image_dir.exists():
        log("Extracting image dataset...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(base_dir)

    image_paths = sorted(glob.glob(str(image_dir / "**/*.png"), recursive=True))
    log(f"Images found: {len(image_paths)}")
    return image_paths


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def embed_texts(model: SentenceTransformer, texts: list[str], batch_size: int = 64) -> np.ndarray:
    return model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,
    ).astype(np.float32)


def embed_images(
    model: CLIPModel,
    processor: CLIPProcessor,
    image_paths: list[str],
    device: str,
    batch_size: int = 16,
) -> np.ndarray:
    vectors = []
    with torch.no_grad():
        for i in range(0, len(image_paths), batch_size):
            batch = image_paths[i : i + batch_size]
            images = [Image.open(p).convert("RGB") for p in batch]
            inputs = processor(images=images, return_tensors="pt").to(device)
            features = model.get_image_features(**inputs)
            features = features / features.norm(dim=-1, keepdim=True)
            vectors.append(features.cpu().numpy().astype(np.float32))
    return np.vstack(vectors)


def build_documents(restaurants: list[dict], recipes: list[dict], image_paths: list[str]):
    article_docs = []
    for i, restaurant in enumerate(restaurants):
        name = str(restaurant.get("name", "")).strip()
        if not name:
            continue

        text = (
            f"Restaurant: {name}\n"
            f"Cuisine: {restaurant.get('food_style', '')}\n"
            f"Location: {restaurant.get('location', '')}"
        )

        article_docs.append(
            Document(
                page_content=text.strip(),
                metadata={
                    "doc_id": f"rest_{i}",
                    "cuisine": restaurant.get("food_style"),
                    "location": restaurant.get("location"),
                    "source": "restaurant",
                },
            )
        )

    image_docs = []
    for i, (image_path, recipe) in enumerate(zip(image_paths, recipes)):
        image_docs.append(
            Document(
                page_content=recipe.get("name", f"recipe image {i}"),
                metadata={
                    "doc_id": f"img_{i}",
                    "image_path": image_path,
                    "source": "recipe_image",
                    "recipe_id": recipe.get("id"),
                    "cuisine": recipe.get("cuisine"),
                },
            )
        )

    log(f"Article documents: {len(article_docs)}")
    log(f"Image documents: {len(image_docs)}")
    return article_docs, image_docs


def build_vector_indexes(article_docs: list[Document], image_docs: list[Document]) -> None:
    text_model = SentenceTransformer("all-MiniLM-L6-v2")
    log("Text embedder ready")

    device = "cpu"
    clip_name = "openai/clip-vit-base-patch32"
    clip_model = CLIPModel.from_pretrained(clip_name).to(device)
    clip_processor = CLIPProcessor.from_pretrained(clip_name, use_fast=True)
    clip_model.eval()
    log("Image embedder ready")

    db_dir = (Path.home() / "chroma_multimodal").resolve()
    if db_dir.exists():
        shutil.rmtree(db_dir)

    article_vectors = embed_texts(text_model, [d.page_content for d in article_docs])
    article_db = Chroma(collection_name="restaurant_articles", persist_directory=str(db_dir))
    article_db._collection.upsert(
        ids=[d.metadata["doc_id"] for d in article_docs],
        embeddings=article_vectors.tolist(),
        documents=[d.page_content for d in article_docs],
        metadatas=[d.metadata for d in article_docs],
    )
    log("Article DB ready")

    image_vectors = embed_images(
        model=clip_model,
        processor=clip_processor,
        image_paths=[d.metadata["image_path"] for d in image_docs],
        device=device,
    )
    image_db = Chroma(collection_name="food_images", persist_directory=str(db_dir))
    image_db._collection.upsert(
        ids=[d.metadata["doc_id"] for d in image_docs],
        embeddings=image_vectors.tolist(),
        documents=[d.page_content for d in image_docs],
        metadatas=[d.metadata for d in image_docs],
    )
    log("Image DB ready")
    log("Multimodal vector index construction complete")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]

    # Prefer existing outputs in lab folders; fall back to repo root if needed.
    restaurant_json = repo_root / "Lab-Structure-Unstructured-Restaurant-Data-with-an-LLM" / "structured_restaurant_data.json"
    recipe_json = repo_root / "Lab-Process-Multimodal-Data-with-LLMs" / "augmented_food_recipe.json"

    if not restaurant_json.exists():
        restaurant_json = repo_root / "structured_restaurant_data.json"
    if not recipe_json.exists():
        recipe_json = repo_root / "augmented_food_recipe.json"

    image_paths = download_and_extract_images(repo_root)
    restaurants = load_json(restaurant_json)
    recipes = load_json(recipe_json)

    log(f"Loaded restaurants: {len(restaurants)}")
    log(f"Loaded recipes: {len(recipes)}")

    article_docs, image_docs = build_documents(restaurants, recipes, image_paths)
    build_vector_indexes(article_docs, image_docs)


if __name__ == "__main__":
    main()
