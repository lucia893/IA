# evaluate_2048.py
from __future__ import annotations
import numpy as np
from train import play_one_game


def evaluate(agent,
             episodes: int = 30,
             seed: int = 999) -> dict:

    rewards = []
    max_tiles = []
    wins = []

    for ep in range(episodes):
        stats = play_one_game(agent, seed + ep)
        rewards.append(stats["reward"])
        max_tiles.append(stats["max_tile"])
        wins.append(stats["win"])

    rewards = np.array(rewards)
    max_tiles = np.array(max_tiles)
    wins = np.array(wins)

    return {
        "reward_mean": float(rewards.mean()),
        "reward_std": float(rewards.std()),
        "max_tile_mean": float(max_tiles.mean()),
        "max_tile_std": float(max_tiles.std()),
        "win_rate": float(wins.mean())
    }
