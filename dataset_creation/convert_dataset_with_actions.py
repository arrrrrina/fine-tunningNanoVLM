import json
from collections import Counter

import datasets
from datasets import Dataset


DATASET_JSON_PATH = "dataset_high_resolution/dataset.json"
OUTPUT_DATASET_PATH = "my_expert_dataset_action_ids"

action_to_id = {
    "left": "0",
    "right": "1",
    "forward": "2",
}

prompt = (
    "Look at the image. "
    "Action IDs: 0 = left, 1 = right, 2 = forward. "
    "Choose the best action to reach the goal. "
    "Answer with only one number."
)

with open(DATASET_JSON_PATH, "r") as f:
    raw_data = json.load(f)

image_paths = []
conversations = []

for entry in raw_data:
    action_text = entry["action"].strip().lower()

    if action_text not in action_to_id:
        continue

    image_paths.append(entry["image_path"])

    conversations.append(
        [
            {
                "user": prompt,
                "assistant": action_to_id[action_text],
            }
        ]
    )

dataset = Dataset.from_dict(
    {
        "images": image_paths,
        "texts": conversations,
    }
)

dataset = dataset.cast_column("images", datasets.Image())

action_counter = Counter()

for example in dataset:
    action_counter[example["texts"][0]["assistant"]] += 1

print(dataset)
print("Action distribution:", action_counter)

dataset.save_to_disk(OUTPUT_DATASET_PATH)

print("Dataset was successfully created.")