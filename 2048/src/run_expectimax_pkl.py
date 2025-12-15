import pickle
from datetime import datetime
import numpy as np
from GameBoard import GameBoard
from ExpectiMaxAgent import ExpectimaxAgent


BEST_PARAMS = dict(
    depth=4,
    empty_weight=30000.0,
    smooth_weight=1.0,
    max_tile_weight=0.5,
    mono_weight=5.0,
    corner_weight=50.0,
    value_weight=0.0001,
    p_two=0.9,
)


def run_one_game(params, seed=None):
    if seed is not None:
        np.random.seed(seed)

    board = GameBoard()
    agent = ExpectimaxAgent(**params)

    done = False
    moves = 0
    total_reward = 0.0 

    while not done:
        action = agent.play(board)
        done = board.play(action)
        moves += 1
        total_reward = board.get_max_tile()

    max_tile = board.get_max_tile()
    return {
        "max_tile": float(max_tile),
        "moves": int(moves),
        "reward_proxy": float(total_reward),
    }


def main():
    n_games = 10
    results = []

    for i in range(n_games):
        res = run_one_game(BEST_PARAMS, seed=42 + i)
        res["game_idx"] = i
        results.append(res)
        print(f"Juego {i+1}: max_tile={res['max_tile']}, moves={res['moves']}")

    max_tiles = np.array([r["max_tile"] for r in results], dtype=float)
    moves = np.array([r["moves"] for r in results], dtype=float)

    summary = {
        "params": BEST_PARAMS,
        "results": results,
        "mean_max_tile": float(max_tiles.mean()),
        "std_max_tile": float(max_tiles.std()),
        "mean_moves": float(moves.mean()),
        "std_moves": float(moves.std()),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }

    filename = "expectimax_depth4_10runs.pkl"
    with open(filename, "wb") as f:
        pickle.dump(summary, f)

    print("\nResumen guardado en:", filename)
    print("mean_max_tile:", summary["mean_max_tile"])
    print("mean_moves   :", summary["mean_moves"])


if __name__ == "__main__":
    main()
