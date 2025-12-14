from __future__ import annotations
import numpy as np
from typing import Dict
from GameBoard import GameBoard
from agent import Agent

TARGET_TILE = 2048


def check_win(board: GameBoard) -> bool:
    return board.get_max_tile() >= TARGET_TILE


def play_one_game(agent: Agent, seed: int | None = None) -> Dict[str, float]:
    """
    Juega una partida completa de 2048.
    Devuelve métricas coherentes:
        - reward: suma total de fichas (score)
        - max_tile
        - moves
        - win (1 si >= 2048, sino 0)
    """
    if seed is not None:
        np.random.seed(seed)

    board = GameBoard()
    done = False
    moves = 0

    while not done:
        action = agent.play(board)
        no_moves = board.play(action)
        done = no_moves or check_win(board)
        moves += 1

    max_tile = int(board.get_max_tile())
    reward = float(np.sum(board.grid))
    win = 1 if max_tile >= TARGET_TILE else 0

    return {
        "reward": reward,
        "max_tile": max_tile,
        "moves": moves,
        "win": win,
    }


def moving_avg(arr, w=50):
    out = []
    for i in range(len(arr)):
        window = arr[max(0, i-w+1):i+1]
        out.append(sum(window)/len(window))
    return out


def train(agent: Agent,
          episodes: int = 50,
          seed: int = 42,
          wandb_run=None):

    rewards, max_tiles, moves_list, wins = [], [], [], []

    for ep in range(episodes):
        stats = play_one_game(agent, seed + ep)

        rewards.append(stats["reward"])
        max_tiles.append(stats["max_tile"])
        moves_list.append(stats["moves"])
        wins.append(stats["win"])

        ma50 = moving_avg(rewards, 50)[-1]

        if wandb_run is not None:
            wandb_run.log({
                "episode": ep,
                "reward": rewards[-1],
                "ma50_reward": ma50,          # igual que MA50 en CartPole
                "max_tile": max_tiles[-1],
                "moves": moves_list[-1],
                "win": wins[-1],
            })

    return {
        "rewards": np.array(rewards),
        "max_tiles": np.array(max_tiles),
        "moves": np.array(moves_list),
        "wins": np.array(wins)
    }
