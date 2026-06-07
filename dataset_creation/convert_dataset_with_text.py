import json
import datasets
from datasets import Dataset
from collections import Counter

with open("dataset_text_action/dataset.json", "r") as f:
    raw_data = json.load(f)

PROMPT = """Look at the image.

Describe the current situation in 1-2 short sentences.
Then choose the best action to reach the goal.

Action IDs:
0 = left
1 = right
2 = forward

Use exactly this format:
State: ...
Plan: ...
Action: 0/1/2"""

images_paths = []
texts_data = []

for entry in raw_data:
    images_paths.append(entry["image_path"])

    conversation = [
        {
            "user": PROMPT,
            "assistant": entry["assistant"]
        }
    ]

    texts_data.append(conversation)

dataset = Dataset.from_dict({
    "images": images_paths,
    "texts": texts_data
})

dataset = dataset.cast_column("images", datasets.Image())

print(dataset)

counter = Counter()

for ex in dataset:
    answer = ex["texts"][0]["assistant"]

    if "Action: 0" in answer:
        counter["0"] += 1
    elif "Action: 1" in answer:
        counter["1"] += 1
    elif "Action: 2" in answer:
        counter["2"] += 1

print("Action distribution:", counter)

dataset.save_to_disk("my_expert_dataset_text_action")

print("Dataset created.")