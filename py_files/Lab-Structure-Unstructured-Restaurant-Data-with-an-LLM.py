# %% [markdown]
# <p style="text-align:center">
#     <a href="https://skills.network" target="_blank">
#     <img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/assets/logos/SN_web_lightmode.png" width="200" alt="Skills Network Logo"  />
#     </a>
# </p>
# 

# %% [markdown]
# # **Lab: Structure Unstructured Restaurant Data with an LLM**
# 

# %% [markdown]
# Estimated time needed: **45** minutes
# 

# %% [markdown]
# ## **Scenario**
# 

# %% [markdown]
# ### Background
# 
# You are a Data Engineer at a leading AI startup building a next-generation *Connoisseur Companion*. Unlike traditional recommendation systems that rely on coarse 1–5 star ratings, the organization aims to understand the **why** behind user preferences, capturing the vibe of a restaurant, dietary considerations, and standout **“hero dishes”** hidden inside unstructured, multimodal data.
# 
# ### The challenge
# 
# The organization has acquired a large, messy dataset spanning multiple modalities:
# 
# * **Restaurant descriptions** and **user reviews** (raw TXT)
# * **Food recipes** with structured metadata and images (JSON + JPEG)
# * **User restaurant visit histories** (JSON with URLs)
# 
# However, this data is not immediately usable:
# 
# 1. **Unstructured text** lacks a consistent schema, making search and retrieval inefficient.
# 2. **Images and URLs** are opaque to traditional data pipelines and must be transformed into searchable, quantitative representations.
# 3. The data spans multiple modalities with no unified representation.
# 
# To build a high-quality, explainable recommendation engine, your first task is to transform this **heterogeneous, multimodal data** into a **structured knowledge base**. This includes extracting semantic signals from **text**, generating representations for **images** and **URLs**, and organizing everything into a coherent, query-friendly JSON format.
# 

# %% [markdown]
# ## **Objectives**
# 

# %% [markdown]
# In this lab, you will write a Python program that will:
# 
# * Load raw restaurant descriptions and review data from unstructured TXT files
# * Use a multimodal/LLM-based pipeline to extract structured attributes (e.g., cuisine type, ambiance, dietary options, and signature dishes)
# * Convert the extracted information into a well-defined JSON schema suitable for indexing and search
# 

# %% [markdown]
# ## **Important: About the lab environment**
# 

# %% [markdown]
# Please be aware that sessions for this lab environment are not persisted. Every time you connect to this lab, a new environment is created for you. Any data you may have saved in the earlier session would get lost. Plan to complete these labs in a single session, to avoid losing your data.
# 

# %% [markdown]
# ## **Screenshot requirement for this lab**
# 

# %% [markdown]
# You will be prompted to take a screenshot and save it on your own device. You will need this screenshot either to answer graded quiz questions or to upload as your submission for the Final Project at the end of this course. You can use various free screen-grabbing tools or your operating system's shortcut keys to do this (for example, `Alt+PrintScreen` on Windows and `Command+shift+4` on Mac).
# 
# **Note**: The screenshot can be saved with either the **.jpg** or **.png** extension.
# 

# %% [markdown]
# ----
# 

# %% [markdown]
# ## **Set up the lab environment**
# 

# %% [markdown]
# For this lab, you will be using the following libraries:
# 
# * [`numpy`](https://numpy.org/) for numerical operations and handling array-based data during preprocessing and analysis
# 
# * [`matplotlib`](https://matplotlib.org/) for basic data visualization and plotting, useful for inspecting distributions or intermediate results
# 
# * [`json`](https://docs.python.org/3/library/json.html) for parsing, constructing, and serializing structured JSON representations extracted from unstructured text
# 
# * **IBM Watsonx AI SDK** (`ibm-watsonx-ai`) for interacting with foundation models hosted on IBM watsonx.ai:
# 
#   * `Credentials` to securely authenticate with the watsonx.ai service
#   * `ModelInference` to invoke foundation models for text understanding and information extraction
#   * `GenTextParamsMetaNames` to configure text generation and extraction parameters
#   * `ModelTypes` and `DecodingMethods` to select the appropriate model and control inference behavior
# 
# These libraries together enable you to transform raw restaurant description text into structured, machine-readable knowledge using GenAI-powered workflows.
# 

# %% [markdown]
# ### Install required libraries
# 
# Run the following code block to install all required libraries:
# 

# %% [markdown]
# ### Import required libraries
# 
# It is recommended to import all required libraries in one place (here):
# 

# %%
import numpy as np
import matplotlib.pyplot as plt
import json
import os

# IBM WatsonX imports
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.foundation_models.utils.enums import (
    ModelTypes,
    DecodingMethods,
)

# Libraries and codes to suppress warnings generated by your code:
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
warnings.filterwarnings('ignore')

# %% [markdown]
# ----
# 

# %% [markdown]
# ## **Exercise 1: Load the data and display the texts**
# 

# %% [markdown]
# Great! You now have the **California-Culinary-Map.txt** file in your folder. Before diving into the data, it is important to load and explore it first to gain a high-level understanding of its structure and content. This exercise gives you the opportunity to inspect the raw data and prepare it for transformation into a more structured, machine-accessible format.
# 

# %% [markdown]
# ### Step 1: Load the data
# 
# In Step 1, you will load the text data file and explore the restaurant contents. Make sure you use the correct file path that links to the file California-Culinary-Map.txt. 
# 

# %%
### Your Code Here:
### 1.1: Define the file_path to the text file
file_path = "California-Culinary-Map.txt"  # Replace with your actual text file name or path

### 1.2: Open the text file
with open(file_path, 'r', encoding='utf-8') as file:
    data = file.read()

### 1.3: Print the first 100 characters of the restaurant data.
print(data[:100])

# %% [markdown]
# ### Step 2: Split the restaurant paragraphs into a Python list
# 
# In the previous step, you may have noticed that the text file consists of multiple paragraphs, each describing a single restaurant. In Step 2, you will split these paragraphs into a list, allowing you to work with and manage each restaurant’s data more effectively.
# 

# %%
### Your Code Here:
### 2.1: Split the restaurant paragraphs into list (hint: use .split)
restaurant_list = data.split('\n\n')  # Assuming paragraphs are separated by double newlines

### 2.2: Since the first item is the dataset name, we remove it
restaurant_list = restaurant_list[1:]

### 2.3: Print out the number of restaurants we have
print(f"Number of restaurants: {len(restaurant_list)}")

### 2.4: Print out the first item to have a closer look at the content
print(restaurant_list[0])


# %% [markdown]
# ----
# 

# %% [markdown]
# ## **Exercise 2: Define the LLM**
# 

# %% [markdown]
# You may have observed that each restaurant description includes the following key attributes: 
# 1. Restaurant name
# 2. Location
# 3. Restaurant type
# 4. Food style
# 5. Rating
# 6. Price range
# 7. Signature dishes
# 8. Specialties
# 9. Shortcomings
# 
# This information is essential for uniquely identifying and describing a restaurant.
# 
# In this lab, you will use a large language model (LLM) to organize these attributes into a structured JSON format. As a first step, you will define the base LLM that will extract and structure the information from the unstructured text.
# 

# %% [markdown]
# ### Step 1: Implement the LLM function
# 

# %% [markdown]
# You will use **IBM Granite-3-8b-instruct** as the base LLM for this lab, as it provides strong text understanding capabilities while remaining cost-efficient:
# 

# %%
def llm_model(system_msg, prompt_txt):
    model_id = "ibm/granite-4-h-small"
    project_id = "skills-network"

    credentials = Credentials(
        url="https://us-south.ml.cloud.ibm.com"
    )

    ### 1.1: Define the model by ModelInference
    model = ModelInference(
        model_id=model_id,
        credentials=credentials,
        project_id=project_id
    )

    ### 1.2: Define the messages
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": prompt_txt}
    ]

    ### 1.3: Get the final response output and return it
    response = model.chat(
        messages=messages,
        params={
            "temperature": 0.7,
            "max_tokens": 512,
            "top_p": 0.9,
            "frequency_penalty": 0,
            "presence_penalty": 0
        }
    )

    # Extract the generated text from the response
    output_text = response["choices"][0]["message"]["content"]
    return output_text

# %% [markdown]
# ### Step 2: Test your llm_model()
# 

# %% [markdown]
# Test your LLM function with the following inputs:
# 

# %%
import time

def safe_llm_call(system_msg, prompt_txt, retries=3):
    for i in range(retries):
        try:
            return llm_model(system_msg, prompt_txt)
        except Exception:
            time.sleep(2)
    return "Failed after retries"

# %%
system_msg = "You are a helpful assistant."
prompt_txt = "Which place is warmer in winter? Hawaii or Greenland?"
print(safe_llm_call(system_msg, prompt_txt))

# %% [markdown]
# ----
# 

# %% [markdown]
# ## **Exercise 3: Prompt Engineering**
# 

# %% [markdown]
# To make an LLM behave reliably and produce outputs that meet your requirements, careful prompt design is essential. Here is a recap of key prompt engineering techniques:
# 
# * **Clear and specific instructions**: Explicitly state the task, expected output format, and constraints to reduce ambiguity.
# * **Structured output guidance**: Provide schemas, templates, or examples (e.g., JSON formats) to encourage consistent, machine-readable responses.
# * **Few-shot prompting**: Include representative input–output examples to guide the model toward the desired behavior.
# * **Role prompting**: Assign the model a specific role (e.g., “You are a data extraction assistant”) to shape its reasoning and tone.
# * **Constraint-based prompting**: Define what the model should and should not do to improve precision and reliability.
# * **Iterative refinement**: Evaluate outputs and progressively refine prompts to improve performance over time.
# 
# This exercise focuses on one-shot prompting, a special case of few-shot prompting. You will define a function that generates the prompt templates.
# 

# %% [markdown]
# ### Step 1: Define the template
# 

# %% [markdown]
# In **Step 1**, you will implement a function that generates prompts for the LLM, given an input restaurant description paragraph.
# 
# You will apply the **one-shot prompting** technique, which includes a single example of the desired output within the prompt to guide the model’s response.
# 
# An example output is provided, corresponding to the second restaurant paragraph in your `restaurant_list` variable. Using this example, you will design a prompt template that enables the model to consistently transform *any* input restaurant description into the required structured JSON format.
# 
# **Note:** For the price range field, instruct the LLM to convert dollar signs (e.g., \\\\\\\\\\\\\\\\\\\\\\$, \\$\\$, $$$) into an integer representing the number of dollar symbols.
# 

# %%
EXAMPLE_RESTAURANT_PARAGRAPH = restaurant_list[1] #use the second restaurant paragraph as the example
EXAMPLE_OUTPUT = """
    {{
    "name": "Mar de Cortez",
    "location": "Santa Monica",
    "type": "casual taqueria",
    "food_style": "Baja-style seafood",
    "rating": 4.2,
    "price_range": 1,
    "signatures": [
        "beer-battered snapper tacos",
        "zesty octopus ceviche"
    ],
    "vibe": "salt-air energy",
    "environment": "a premier sun-drenched spot for open-air dining near the pier."
    "shortcomings": []
    }}
"""

### Design your prompt here
def restaurant_data_structure_prompt_generation(restaurant_paragraph):
    base_system_msg = f"""
    You are a data extraction assistant. Your task is to extract structured information from restaurant description paragraphs and return it as a valid JSON object.

    Follow these rules strictly:
    - Return only the JSON object, with no additional text, explanation, or markdown formatting.
    - Extract only information explicitly stated in the description; do not infer or fabricate details.
    - For "price_range", count the number of dollar signs (e.g., "$" = 1, "$$" = 2, "$$$" = 3) and return it as an integer.
    - For "rating", return a float.
    - For "signatures", return a list of strings representing signature dishes or drinks mentioned.
    - For "shortcomings", return a list of any mentioned negatives or complaints; return an empty list if none.
    - Use the exact JSON field names and structure shown in the example.
    """

    base_user_prompt = f"""
    Task:
    Extract structured restaurant data from the description below and return a JSON object with these fields:
    "name", "location", "type", "food_style", "rating", "price_range", "signatures", "vibe", "environment", "shortcomings".

    Restaurant description:
    {restaurant_paragraph}

    Example:
    Input Restaurant Description: {EXAMPLE_RESTAURANT_PARAGRAPH}
    Output:
    {EXAMPLE_OUTPUT}

    Now extract the data from the restaurant description above and return only the JSON object.
    """
    return base_system_msg, base_user_prompt

# %% [markdown]
# ### Step 2: Test your prompts
# 

# %% [markdown]
# Run the following code to verify that your prompt produces the expected output:
# 

# %%
# Unit test:
restaurant_paragraph = restaurant_list[0]
base_system_msg, base_user_prompt = restaurant_data_structure_prompt_generation(restaurant_paragraph=restaurant_paragraph)

test_response = llm_model(system_msg=base_system_msg, prompt_txt=base_user_prompt)
print(test_response)

# %% [markdown]
# ### Step 3: Validate the LLM outputs
# 

# %% [markdown]
# LLMs are not always perfectly reliable and may produce errors, even when strict rules are specified in the prompt. To ensure that the generated outputs strictly conform to the required JSON format, you will learn how to define a function that formally validates the LLM output. 
# 
# Here, you are provided with the function and the test. You don't have to implement anything. Run the following code to validate the test response you obtained in Step 2.
# 

# %%
# Validation
from pydantic import BaseModel, Field, ValidationError
from typing import List, Optional

### 3.1. Define the schema
class Restaurant(BaseModel):
    name: str
    location: str
    type: str
    food_style: str
    rating: Optional[float] = None
    price_range: Optional[int] = None
    signatures: List[str] = Field(default_factory=list)
    vibe: Optional[str] = None
    environment: str
    shortcomings: List[str] = Field(default_factory=list)


### 3.2. Use the validation method to validate the test_response from the unit test
try:
    restaurant_data = Restaurant.model_validate_json(test_response)
    print(f"Success! Validated: {restaurant_data.name}")
except ValidationError as e:
    print(f"Validation failed: {e.json()}")

# %% [markdown]
# ----
# 

# %% [markdown]
# ## **Exercise 4: Structure all the restaurant data**
# 

# %% [markdown]
# ### Step 1: Define prompts to have an LLM that auto repairs outputs to JSON format
# 

# %% [markdown]
# As mentioned earlier, LLMs are not perfect. While you previously defined a schema to validate the output, validation alone does not fix incorrect results. To address this, introduce an additional LLM “expert” whose sole responsibility is to automatically repair and correct outputs that do not conform to the required JSON format.
# 
# In this step, you will define the prompt template function for this LLM. This function takes:
# - `candidate_json_output`: The candidate json output
# - `error_message`: The error message generated by the schema validation if a mistake is detected
# 
# In your prompts, you should tell LLM:
# - The original wrong output by feeding `candidate_json_output`
# - The guidance on correction by feeding `error_message`
# 

# %%
def JSON_auto_repair_prompts(candidate_json_output, error_message):
    auto_repair_system_msg = """
    You are a JSON repair expert. Your sole responsibility is to fix malformed or invalid JSON outputs so they conform to the required schema.

    Follow these rules strictly:
    - Return only the corrected JSON object, with no additional text, explanation, or markdown formatting.
    - Do not change values that are already correct — only fix what is broken.
    - Use the error message as guidance to identify and correct the specific issue.
    - Ensure the output is valid, parseable JSON.
    """

    auto_repair_prompt = f"""
    The following JSON output failed schema validation.

    Invalid JSON output:
    {candidate_json_output}

    Validation error:
    {error_message}

    Fix the JSON so it passes validation and return only the corrected JSON object.
    """
    return auto_repair_system_msg, auto_repair_prompt

# %% [markdown]
# ### Step 2: Run the for loop to go over all the restaurant data in the list (approximately 20 minutes)
# 

# %% [markdown]
# You will structure each restaurant description paragraph into a JSON-formatted output by iterating through the dataset using a for loop.
# 
# Although using an LLM to auto repair the response format is not the best practice, here you are guaranteed that the generation and repair tasks here are simple enough for LLMs to accomplish.
# 

# %%
### Your Code Here:
structured_restaurant_lists = []
for i, restaurant_paragraph in enumerate(restaurant_list):
    ### 2.1: Produce your initial output
    system_msg, user_prompt = restaurant_data_structure_prompt_generation(restaurant_paragraph)
    raw_output = llm_model(system_msg, user_prompt)

    ### 2.2: Validation and Auto Correction loop on the output
    while True:
        try:
            restaurant_data = Restaurant.model_validate_json(raw_output)
            break  # valid — exit the loop
        except ValidationError as e:
            error_message = e.json()
            repair_system_msg, repair_prompt = JSON_auto_repair_prompts(raw_output, error_message)
            raw_output = llm_model(repair_system_msg, repair_prompt)

    ### 2.3: Append your finalized response to the structured_restaurant_lists
    structured_restaurant_lists.append(restaurant_data)

    # A manual progress bar
    if (i+1) % 20 == 0:
        print(f'{i+1} out of {len(restaurant_list)} is done')

print('ALL DONE!!')

# %% [markdown]
# Take a screenshot of the Python code, clearly showing your implementation and the output from Step 2. Name the screenshot `M1L1_structure_for_loop.jpg`. 
# 

# %% [markdown]
# ### Step 3: Save the list to a JSON file
# 

# %% [markdown]
# You are almost there! First, print the 50th item in your structured_restaurant_lists.
# 

# %%
### Your Code Here:
### Print the 50th item in the structured_restaurant_lists
print(structured_restaurant_lists[49].model_dump())

# %% [markdown]
# Then, save your processed data to your folder by running the following code!
# 

# %%
structured_restaurant_lists_json = [response.model_dump() for response in structured_restaurant_lists]

#For each item in the restaurant list, assign it with an itemId to be consistent with the one in the user review data:
for i, response in enumerate(structured_restaurant_lists_json):
    response['itemId'] = 1000001 + i
    structured_restaurant_lists_json[i] = response
    
filename = 'structured_restaurant_data.json'
with open(filename, 'w', encoding='utf-8') as f:
    json.dump(structured_restaurant_lists_json, f, indent=4)

# %% [markdown]
# You can download the saved JSON file to your local folder. Be sure to keep it in a safe location, as you will need it for future assignments.
# 

# %% [markdown]
# ----
# 

# %% [markdown]
# ## **Conclusion**
# 

# %% [markdown]
# You have successfully applied GenAI tools to transform the unstructured text data into a well-structured JSON file!
# 

# %% [markdown]
# ## Authors
# 

# %% [markdown]
# [Jianping (Mike) Ye](https://www.linkedin.com/in/jianping-ye/)
# 

# %% [markdown]
# <!--## Changelog
# |Date (YYYY-MM-DD)|Version|Changed By|Change Description|
# |-|-|-|-|
# |2026-02-02|1|Jianping Ye|Create lab|
# |2026-02-09|2|Jojy John|ID Review|-->
# 

# %% [markdown]
# Copyright IBM Corporation. All rights reserved.
# 


