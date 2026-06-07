import os
import json

import gymnasium as gym
import minigrid
from minigrid.wrappers import RGBImgObsWrapper
from PIL import Image


DATASET_DIR = "dataset_high_resolution"
IMAGES_DIR = os.path.join(DATASET_DIR, "images")
os.makedirs(IMAGES_DIR, exist_ok=True)

env = gym.make("MiniGrid-Empty-Random-6x6-v0", render_mode="rgb_array")
env = RGBImgObsWrapper(env, tile_size=42)

action_to_text = {
    0: "left",
    1: "right",
    2: "forward",
}


def find_goal_position(env):
    grid = env.unwrapped.grid

    for x in range(grid.width):
        for y in range(grid.height):
            cell = grid.get(x, y)
            if cell is not None and cell.type == "goal":
                return x, y

    raise RuntimeError("Goal cell was not found in the environment.")


def turn_towards(agent_dir, target_dir):
    if agent_dir == target_dir:
        return 2  # forward

    if (agent_dir + 1) % 4 == target_dir:
        return 1  # right

    return 0  # left


def get_expert_action(env):
    """
    Heuristic expert policy for MiniGrid EmptyEnv.

    The policy finds the goal position and follows a shortest Manhattan-style
    path in an empty room. It first aligns the agent with the goal along the
    x-axis and then along the y-axis.
    """
    agent_x, agent_y = env.unwrapped.agent_pos
    agent_dir = env.unwrapped.agent_dir

    goal_x, goal_y = find_goal_position(env)

    if agent_x != goal_x:
        target_dir = 0 if goal_x > agent_x else 2
        return turn_towards(agent_dir, target_dir)

    if agent_y != goal_y:
        target_dir = 1 if goal_y > agent_y else 3
        return turn_towards(agent_dir, target_dir)

    return 2  # fallback action


def collect_dataset(num_episodes=1000):
    dataset = []
    step_counter = 0

    for _ in range(num_episodes):
        obs, info = env.reset()
        done = False
        truncated = False

        while not (done or truncated):
            image = Image.fromarray(obs["image"])
            image_path = os.path.join(IMAGES_DIR, f"step_{step_counter}.png")
            image.save(image_path)

            action = get_expert_action(env)

            dataset.append(
                {
                    "image_path": image_path,
                    "prompt": "What should the agent do next to reach the goal?",
                    "action": action_to_text[action],
                }
            )

            obs, reward, done, truncated, info = env.step(action)
            step_counter += 1

    return dataset, step_counter


dataset, num_steps = collect_dataset(num_episodes=1000)

with open(os.path.join(DATASET_DIR, "dataset.json"), "w") as f:
    json.dump(dataset, f, indent=4)

print(f"Collected {num_steps} expert transitions.")