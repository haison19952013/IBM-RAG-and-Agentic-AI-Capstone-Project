# Assignment Overview: Process Multimodal Data with LLMs

Estimated time: 5 minutes

## Introduction

Building on the structured JSON knowledge base you created earlier, this lab extends your GenAI data pipeline beyond text by incorporating visual data. You will work with food recipe images, recipe metadata, and user visit history, and use vision-enabled LLMs to convert visual information into meaningful textual descriptions that enrich your existing structured JSON data.

This lab focuses on extending GenAI-powered data pipelines beyond text by using image captioning to enrich structured JSON data, enabling more comprehensive and explainable recommendation systems.

## Screenshot Requirement for the Final Project

During the course of the lab, you will be instructed to take a screenshot of the Python code, clearly showing your implementation and the output for a particular task.

Name the screenshot: M1L2_caption_all_recipes.jpg

This screenshot will later serve as a component for the Final Project.

## Background

The dataset produced in this lab strengthens the AI-powered multimodal restaurant recommendation system by enabling cross-modal retrieval and improving ranking across text and image data. The additional semantic signals support more effective agent reasoning and contribute to more accurate, personalized, and explainable recommendations in later modules.

## Why This Lab Matters

You are learning how multimodal LLMs can translate visual inputs into structured, machine-readable knowledge. By designing prompts for vision models and integrating generated captions into existing JSON data, you will gain hands-on experience building scalable multimodal data workflows that complement and enhance text-based understanding.

## What You Will Do in This Lab

You will follow a step-by-step workflow to enrich existing structured JSON data with multimodal insights using vision-enabled LLMs.

### Explore Multimodal Recipe and User Visit Data

First, you will begin by loading and inspecting food recipe JSON files and associated images, as well as user visit history data that references visual content. This step helps you understand how images are linked to textual metadata and how visual context can add valuable semantic information to existing structured records.

### Design Prompts for Image Captioning

Next, you will design prompt templates for vision-enabled LLMs to generate concise, informative captions for food images. These prompts guide the model to focus on relevant visual attributes, such as ingredients, presentation, and cooking style, and when applicable, incorporate contextual information from user reviews.

### Validate Captions with Small Tests

Before scaling image captioning across the dataset, you will perform unit tests on sample images to validate prompt effectiveness and output quality. This step ensures that generated captions are accurate, consistent, and suitable for integration into structured JSON formats.

### Augment Structured Data with Multimodal Insights

Finally, you will apply the image captioning pipeline to all images and merge the generated descriptions into the existing recipe and user visit JSON data. This process produces an enriched multimodal knowledge file that combines textual and visual understanding in a unified, machine-accessible structure.

## Deliverables

In this lab, you will complete the following deliverables:

- Explore and inspect multimodal recipe and user visit data.
- Design prompt templates for vision-based image captioning.
- Validate image caption outputs using small tests.
- Integrate generated image captions into structured JSON files.
- Save an enriched multimodal knowledge file for downstream applications.

## Conclusion

By the end of this lab, you will have extended a text-based knowledge base into a multimodal structured dataset using vision-capable LLMs. This workflow demonstrates how GenAI can bridge images and text in real-world data engineering pipelines, laying the foundation for richer, more intelligent recommendation systems.