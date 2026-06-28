# Assignment Overview: Multimodal Similarity Fusion and Retrieval Ranking

**Estimated time:** 5 minutes

## Introduction

Earlier, you performed similarity retrieval independently within each modality. Articles and images were retrieved and ranked separately.

In this lab, you will take the next step by combining those independent retrieval results into a unified multimodal ranking pipeline.

Real-world RAG systems rarely rely on a single evidence source. Instead, they retrieve heterogeneous candidates and fuse them into a single ranked result set.

You will:

- Verify persisted vector databases
- Embed queries into both MiniLM and CLIP spaces
- Convert cosine-style distances into similarity scores
- Normalize scores across modalities
- Implement weighted fusion
- Apply optional metadata constraints
- Analyze how weight tuning affects ranking behavior

This lab completes the retrieval architecture.

## Screenshot Requirement for the Final Project

During the course of the lab, you will be instructed to take a screenshot of the final code cell and its output.

Name the file `M2L3_multimodal_fusion_results.jpg`.

This screenshot will later serve as a component for the Final Project.

## Why This Lab Matters

You are learning how to combine retrieval results from multiple modalities into a single, unified ranking system. This is essential in real-world applications where relevant information exists across both text and images.

By implementing normalization and weighted fusion, you will understand how to fairly compare independent embedding spaces, balance modality influence, and produce more complete and accurate retrieval outcomes.

## The Cross-Modal Scoring Problem

MiniLM and CLIP produce similarity scores in independent embedding spaces:

- 384-dimensional text space
- 512-dimensional multimodal space

Their raw similarity values are not directly comparable.

If combined without adjustment, one modality may dominate due to scale differences rather than relevance.

This lesson addresses that calibration problem.

## Design Pattern: Late Fusion

The lab implements late fusion, which follows this workflow:

- Retrieve top-k article candidates
- Retrieve top-k image candidates
- Convert distances to similarity scores
- Normalize similarity scores within each modality
- Apply a weighted combination
- Produce a unified ranked list

Late fusion preserves modality specialization while enabling unified comparison.

## Score Transformation and Normalization

Chroma returns distance values where smaller values indicate higher similarity.

The lab transforms these into similarity scores using:

`similarity = 1 - distance`

Then it applies min-max normalization per modality.

Normalization ensures:

- Comparable score ranges
- Fair cross-modal ranking
- Stable fusion behavior

These transformations are implemented in `_to_similarity()` and `_minmax()`.

## Weighted Multimodal Fusion

The `fuse_rank()` function:

- Accepts configurable weights (`w_text`, `w_img`)
- Computes weighted fused scores
- Merges candidate pools
- Sorts by fused ranking

This design enables tunable retrieval behavior.

Weight tuning directly controls modality influence.

## Experimental Structure

The lab includes three experiments:

### Demo 1: Multimodal Fusion (No Filters)

Balanced weights:

- Combined article and image evidence
- Unified ranking without constraints

### Demo 2: Fusion with Metadata Filters

Adds:

- Location constraint (articles)
- Source constraint (images)

This demonstrates precision-oriented multimodal retrieval.

### Demo 3: Weight Tuning

Compares:

- Text-heavy configuration
- Image-heavy configuration

Learners observe ranking shifts under different weight settings.

This illustrates controllable multimodal retrieval behavior.

## What You Will Do in the Lab

In the upcoming lab, you will:

- Validate vector collections
- Initialize embedding models
- Implement cross-modal retrieval functions
- Convert and normalize similarity scores
- Implement weighted fusion
- Apply metadata filters
- Compare ranking behavior across weight configurations
- Analyze fused ranking outputs

## Key Considerations

As you complete this lesson, consider:

- Why normalization is necessary before fusion
- How weight tuning affects ranking dominance
- How metadata constraints interact with fusion
- Why late fusion is modular and scalable

These design principles mirror real-world multimodal search systems.

## Summary

In this reading, you learned that:

- Independent modality ranking is insufficient for unified retrieval
- Distance-to-similarity conversion enables consistent scoring
- Normalization ensures fair cross-modal comparison
- Weighted fusion integrates heterogeneous evidence
- Metadata filtering improves precision
- Fusion completes the multimodal retrieval pipeline

You are now ready to implement a production-style multimodal fusion and reranking workflow.