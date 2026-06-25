# Assignment Overview: Similarity Retrieval with Metadata Filtering

**Estimated time needed:** 5 minutes

## Introduction

After constructing persistent multimodal vector indexes for restaurant articles and food images, the next step is to use those indexes to perform retrieval. In this lab, you will operationalize those indexes by implementing similarity-based retrieval with optional metadata filtering.

The goal of this assignment is to move from static indexing to dynamic query-time retrieval.

You will:

- Validate the persisted vector databases
- Reinitialize embedding models for query encoding
- Implement reusable retrieval functions
- Perform top-k similarity search
- Apply metadata constraints
- Execute image-to-image visual similarity retrieval
- Analyze how filtering affects retrieval behavior

This lesson transforms your indexed database into an interactive retrieval system.

## Screenshot Requirement for the Final Project

During the course of the lab, you will be instructed to take a screenshot of the final code cell and its output. Name the file `M2L2_similarity_retrieval.jpg`.

This screenshot will later serve as a component for the Final Project.

## Why This Lab Matters

While similarity search retrieves semantically relevant results, real-world systems often require additional control. In this lab, you combine similarity search with metadata filtering to improve precision and enforce structured constraints. This approach reflects how production retrieval systems balance relevance with control when handling user queries.

## From Indexed Data to Query Processing

A vector index is inert until queried. Retrieval requires:

- Query encoding using the same embedding models used during indexing
- Nearest neighbor search in embedding space
- Interpretation and formatting of returned results

In this lab, queries are embedded using:

- MiniLM (384-dimensional) for text-to-article retrieval
- CLIP (512-dimensional) for image-to-image retrieval

Ensuring that query embeddings are generated using the same models as indexing is a fundamental requirement in vector retrieval systems.

## Design Pattern: Hybrid Retrieval

This lab introduces a key architectural concept: hybrid retrieval.

Pure similarity search retrieves the closest vectors in the embedding space. However, production systems often require structured constraints.

For example:

- Only restaurants in a specific location
- Only a specific cuisine
- Only certain content sources

The lab implements a filter-first retrieval pattern:

- Apply metadata constraints using Chroma's `where` parameter.
- Perform similarity search within the filtered candidate set.

This pattern improves precision and controllability while preserving semantic relevance.

## Retrieval Function Abstraction

The lab defines reusable retrieval functions such as:

- `retrieve_articles()`
- `retrieve_images_by_image()`

These functions:

- Encode queries
- Execute vector search
- Return structured outputs
- Abstract away nested Chroma result formats

This modularity reflects production design principles where retrieval logic is encapsulated and reusable.

## Experimental Design: Three Retrieval Scenarios

The lab includes three structured experiments:

### Demo 1: Article Similarity Search (No Filter)

This establishes baseline behavior:

- Top-k nearest neighbor search
- Ranking purely by cosine distance
- No structural constraints

This experiment demonstrates raw semantic retrieval.

### Demo 2: Article Similarity + Metadata Filter

By applying: `where_filter = {"location": "Pasadena"}`

You observe:

- Reduced candidate pool
- Constrained ranking
- Potential trade-offs in recall

This experiment illustrates how structural filtering reshapes the retrieval space.

### Demo 3: Image Similarity Search (Image → Image)

Using CLIP embeddings, you:

- Encode a query image
- Perform nearest neighbor search in image space
- Optionally apply metadata filters

This demonstrates perceptual similarity retrieval independent of textual semantics.

## Observing Retrieval Dynamics

Through these experiments, you analyze:

- How filtering changes result sets
- How ranking shifts under constraints
- How candidate pools shrink
- How similarity behaves differently across modalities

This analytical perspective is essential for understanding production retrieval tuning.

## What You Will Do in the Lab

In the upcoming lab, you will:

- Validate the Chroma persistence directory
- Confirm vector counts
- Initialize embedding models
- Implement retrieval utilities
- Execute text-based similarity search
- Apply metadata filtering
- Perform visual similarity search
- Analyze the impact of filters on ranking behavior

## Key Considerations

As you complete this lesson, reflect on:

- Why similarity alone is insufficient in structured domains
- How metadata filtering shapes precision
- How embedding alignment affects retrieval quality
- How hybrid retrieval balances recall and controllability

These are core principles in production multimodal retrieval systems.

## Summary

In this reading, you learned that:

- Retrieval activates persistent vector indexes
- Query embeddings must match indexing embeddings
- Hybrid similarity + metadata filtering improves controllability
- Visual similarity retrieval extends multimodal capability
- Observing retrieval behavior is critical for tuning systems

You are now ready to implement hybrid similarity retrieval and analyze its behavior.

