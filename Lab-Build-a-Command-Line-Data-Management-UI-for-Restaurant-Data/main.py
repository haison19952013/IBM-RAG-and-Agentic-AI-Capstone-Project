from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from pydantic import BaseModel, Field, ValidationError
from typing import List, Optional
import json
import os
import shutil
import io
import unittest
from unittest.mock import patch

FILEPATH = 'structured_restaurant_data.json'
BACKUP_PATH = 'structured_restaurant_data.json.bak'
EXAMPLE_RESTAURANT_PARAGRAPH = 'Down in **Santa Monica**, **Mar de Cortez** serves as a **sun-drenched**, **casual taqueria** specializing in **Baja-style seafood**. With a **4.2/5** rating, it captures the salt-air energy of the coast through its signature beer-battered snapper tacos and zesty octopus ceviche, making it a premier spot for open-air dining near the pier. Price range: $'

EXAMPLE_OUTPUT = """
    {
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
    "environment": "a premier sun-drenched spot for open-air dining near the pier.",
    "shortcomings": []
    }
"""

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


def llm_model(system_msg, prompt_txt, params=None):
    model_id = "ibm/granite-4-h-small"
    project_id = "skills-network"

    credentials = Credentials(
        url="https://us-south.ml.cloud.ibm.com"
    )

    model = ModelInference(
        model_id=model_id,
        credentials=credentials,
        project_id=project_id
    )

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": prompt_txt}
    ]

    if params is None:
        params = {
            "temperature": 0.7,
            "max_tokens": 512,
            "top_p": 0.9,
            "frequency_penalty": 0,
            "presence_penalty": 0
        }

    response = model.chat(
        messages=messages,
        params=params
    )

    output_text = response["choices"][0]["message"]["content"]
    return output_text


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


def new_data_entry_process(paragraph, itemId):
    # Step 1: Generate initial structured output from paragraph
    system_msg, user_prompt = restaurant_data_structure_prompt_generation(paragraph)
    raw_output = llm_model(system_msg, user_prompt)

    # Step 2: Validate and auto-repair loop
    while True:
        try:
            restaurant_data = Restaurant.model_validate_json(raw_output)
            break
        except ValidationError as e:
            error_message = e.json()
            repair_system_msg, repair_prompt = JSON_auto_repair_prompts(raw_output, error_message)
            raw_output = llm_model(repair_system_msg, repair_prompt)

    # Step 3: Convert to dict and assign itemId
    restaurant_dict = restaurant_data.model_dump()
    restaurant_dict['itemId'] = itemId

    return restaurant_dict


def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_data(file_path, backup_path, data):
    # Back up before overwriting
    if os.path.exists(file_path):
        shutil.copy(file_path, backup_path)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)


def show_restaurant_card(res, index):
    print(f"\n{'='*40}")
    print(f"  [{index}] {res.get('name', 'N/A')}")
    print(f"{'='*40}")
    for key, value in res.items():
        print(f"  {key:15}: {value}")
    print(f"{'='*40}")


def manage_restaurants(file_path=FILEPATH, backup_path=BACKUP_PATH):
    while True:
        data = load_data(file_path)
        print(f"\n🏨 RESTAURANT DATABASE | Records: {len(data)}")
        print("1. Browse All (Names)")
        print("2. View Detailed Record")
        print("3. Add New Restaurant")
        print("4. Edit Restaurant Info")
        print("5. Delete Restaurant")
        print("6. Exit")

        choice = input("\nAction: ")

        if choice == '1':
            print("\n--- Current Listings ---")
            for i, res in enumerate(data):
                print(f"  [{i}] {res.get('name', 'N/A')}")

        elif choice == '2':
            raw = input("Enter record index: ")
            try:
                idx = int(raw)
                if 0 <= idx < len(data):
                    show_restaurant_card(data[idx], idx)
                else:
                    print("Invalid index.")
            except ValueError:
                print("Invalid index.")

        elif choice in ['3', '4', '5']:
            # Strict Security Warning
            print("\n❗ SECURITY WARNING: You are entering write-mode.")
            print("Changes will be saved to the database immediately.")
            confirm = input("Are you sure? (type 'yes' to proceed): ").lower()
            if confirm != 'yes':
                print("Operation cancelled.")
                continue

            if choice == '3':  # ADD NEW DATA
                itemId = 1000000 + len(data) + 1

                paragraph = input("Enter the new restaurant description:\n> ")
                new_record = new_data_entry_process(paragraph, itemId)
                data.append(new_record)
                save_data(file_path, backup_path, data)
                print("✅ Restaurant added.")

            elif choice == '4':  # EDIT DATA
                raw = input("Enter record index to edit: ")
                try:
                    idx = int(raw)
                    if 0 <= idx < len(data):
                        print(f"Editing [{idx}] {data[idx].get('name', 'N/A')}")
                        print("(Press Enter to keep the current value.)")
                        for key in data[idx].keys():
                            current = data[idx][key]
                            new_val = input(f"  {key} [{current}]: ")
                            if new_val.strip() == '':
                                continue  # user skipped — keep existing value
                            # Try to preserve the original type
                            try:
                                if isinstance(current, int):
                                    data[idx][key] = int(new_val)
                                elif isinstance(current, float):
                                    data[idx][key] = float(new_val)
                                elif isinstance(current, list):
                                    data[idx][key] = [v.strip() for v in new_val.split(',')]
                                else:
                                    data[idx][key] = new_val
                            except ValueError:
                                data[idx][key] = new_val
                        save_data(file_path, backup_path, data)
                        print("✅ Record updated.")
                    else:
                        print("Invalid index.")
                except ValueError:
                    print("Invalid index.")

            elif choice == '5':  # DELETE DATA
                raw = input("Enter record index to delete: ")
                try:
                    idx = int(raw)
                    if 0 <= idx < len(data):
                        removed = data.pop(idx)
                        save_data(file_path, backup_path, data)
                        print(f"✅ Deleted: {removed.get('name', 'N/A')}")
                    else:
                        print("Invalid index.")
                except ValueError:
                    print("Invalid index.")

        elif choice == '6':  # EXIT
            print("Goodbye!")
            break

        else:
            print("Invalid input.")


class TestRestaurantDatabase(unittest.TestCase):

    def setUp(self):
        """Create a temporary clean database for testing."""
        self.test_file = 'structured_restaurant_data_unit_test.json'
        self.test_file_backup = 'structured_restaurant_data_unit_test.json.bak'
        self.initial_data = [{"name": "Test Cafe", "location": "Test City"}]
        with open(self.test_file, 'w') as f:
            json.dump(self.initial_data, f)

    def tearDown(self):
        """Clean up the test file after tests."""
        if os.path.exists(self.test_file):
            os.remove(self.test_file)
        if os.path.exists(self.test_file_backup):
            os.remove(self.test_file_backup)

    @patch('builtins.input')
    @patch('sys.stdout', new_callable=io.StringIO)
    def test_add_and_delete_restaurant_success(self, mock_stdout, mock_input):
        """
        Test Scenario: Add a new restaurant.
        Inputs: '3' (Add), 'yes' (Confirm), 'New Burger Joint', '6' (Exit)
        """
        mock_restaurant = 'The Copper Sprout is a high-concept, Modern Appalachian farm-to-table destination that blends an industrial-chic aesthetic with rustic forest charm, featuring reclaimed wood and amber lighting to create a sophisticated yet cozy vibe. Priced in the $$ category, the menu celebrates seasonal foraging and local heritage, headlined by signature dishes like Cast-Iron Smoked Trout with pickled fiddlehead ferns and hand-foraged Wild Mushroom Risotto with aged goat cheese. The experience is designed to be intimate and earthy, making it a premier spot for those seeking high-quality, smokehouse-influenced cuisine in a refined, atmospheric setting.'
        mock_input.side_effect = ['3', 'yes', mock_restaurant, '6']

        # Run the app
        try:
            manage_restaurants(self.test_file, self.test_file_backup)
        except SystemExit:
            pass  # Handle exit if your script uses sys.exit()

        # Check if the data was actually saved
        with open(self.test_file, 'r') as f:
            data = json.load(f)

        print(data)
        self.assertEqual(len(data), 2)
        self.assertIn("✅ Restaurant added.", mock_stdout.getvalue())

        mock_input.side_effect = ['5', 'yes', 1, '6']

        # Run the app
        try:
            manage_restaurants(self.test_file, self.test_file_backup)
        except SystemExit:
            pass  # Handle exit if your script uses sys.exit()

        # Check if the data was actually saved
        with open(self.test_file, 'r') as f:
            data = json.load(f)

        print(data)
        self.assertEqual(len(data), 1)

    @patch('builtins.input')
    @patch('sys.stdout', new_callable=io.StringIO)
    def test_delete_security_cancel(self, mock_stdout, mock_input):
        """
        Test Scenario: Try to delete but say 'no' to security warning.
        Inputs: '5' (Delete), 'no' (Cancel), '6' (Exit)
        """
        mock_input.side_effect = ['5', 'no', '6']

        manage_restaurants(self.test_file, self.test_file_backup)

        with open(self.test_file, 'r') as f:
            data = json.load(f)

        self.assertEqual(len(data), 1)  # Data should remain unchanged
        self.assertIn("Operation cancelled.", mock_stdout.getvalue())


if __name__ == "__main__":
    unittest.main()  # Unit Test
    # manage_restaurants(FILEPATH, BACKUP_PATH)  # Actual UI Call
