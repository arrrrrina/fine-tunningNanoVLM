import gymnasium as gym
import minigrid
from minigrid.wrappers import RGBImgObsWrapper
from PIL import Image
import os
import json

os.makedirs("dataset_text_action/images", exist_ok=True)

env = gym.make("MiniGrid-Empty-Random-6x6-v0", render_mode="rgb_array")
env = RGBImgObsWrapper(env, tile_size=42)

action_to_id = {
    0: "0",  # left
    1: "1",  # right
    2: "2",  # forward
}

dir_to_text = {
    0: "right",
    1: "down",
    2: "left",
    3: "up",
}

def find_goal_pos(env):
    grid = env.unwrapped.grid

    for i in range(grid.width):
        for j in range(grid.height):
            cell = grid.get(i, j)
            if cell is not None and cell.type == "goal":
                return (i, j)

    raise ValueError("Goal not found")


def get_expert_action(env):
    agent_pos = env.unwrapped.agent_pos
    agent_dir = env.unwrapped.agent_dir

    goal_pos = find_goal_pos(env)

    ax, ay = agent_pos
    gx, gy = goal_pos

    if ax != gx:
        target_dir = 0 if gx > ax else 2

        if agent_dir == target_dir:
            return 2
        elif (agent_dir + 1) % 4 == target_dir:
            return 1
        else:
            return 0

    elif ay != gy:
        target_dir = 1 if gy > ay else 3

        if agent_dir == target_dir:
            return 2
        elif (agent_dir + 1) % 4 == target_dir:
            return 1
        else:
            return 0

    return 2


def make_description(env, action_id):
    agent_pos = env.unwrapped.agent_pos
    agent_dir = env.unwrapped.agent_dir
    goal_pos = find_goal_pos(env)

    ax, ay = agent_pos
    gx, gy = goal_pos

    dx = gx - ax
    dy = gy - ay

    if dx > 0:
        horizontal = "to the right"
    elif dx < 0:
        horizontal = "to the left"
    else:
        horizontal = "aligned horizontally"

    if dy > 0:
        vertical = "below"
    elif dy < 0:
        vertical = "above"
    else:
        vertical = "aligned vertically"

    action_text = {
        0: "turn left",
        1: "turn right",
        2: "move forward",
    }[action_id]

    description = (
        f"State: The agent is facing {dir_to_text[agent_dir]}, and the goal is {horizontal} and {vertical} relative to the agent.\n"
        f"Plan: The agent should reduce the distance to the green goal by choosing the next useful movement.\n"
        f"Action: {action_to_id[action_id]}"
    )

    return description


num_episodes = 1000
dataset = []
step_counter = 0

for episode in range(num_episodes):
    obs, info = env.reset()

    done = False
    truncated = False

    while not (done or truncated):
        img_array = obs["image"]
        img = Image.fromarray(img_array)

        img_path = f"dataset_text_action/images/step_{step_counter}.png"
        img.save(img_path)

        action = get_expert_action(env)
        assistant_answer = make_description(env, action)

        dataset.append({
            "image_path": img_path,
            "assistant": assistant_answer,
            "action": action_to_id[action],
        })

        obs, reward, done, truncated, info = env.step(action)
        step_counter += 1

print(f"Collected {step_counter} examples from {num_episodes} episodes.")

with open("dataset_text_action/dataset.json", "w") as f:
    json.dump(dataset, f, indent=4)