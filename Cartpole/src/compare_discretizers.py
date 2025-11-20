from __future__ import annotations
import time
import numpy as np

from src.env import make_env
from src.discretizers import (
    DiscretizerUniform,
    DiscretizerHeuristic,
    DiscretizerDataDriven,
    learn_data_driven_cuts,
)
from src.agents import QLearningAgent
from src.train import train
from src.evaluate import evaluate


def build_agent_with_disc(disc_name: str, env, alpha=0.1, gamma=0.99,
                          eps_start=1.0, eps_end=0.02, eps_decay=0.995):
    if disc_name == "uniform":
        disc = DiscretizerUniform(bins_per_feature=(6, 6, 12, 12))
    elif disc_name == "heur":
        disc = DiscretizerHeuristic()
    elif disc_name == "data":
        cuts = learn_data_driven_cuts(env,
                                      bins_per_feature=(6, 6, 12, 12),
                                      episodes=50,
                                      seed=123)
        disc = DiscretizerDataDriven(cuts)
    else:
        raise ValueError("disc_name debe ser 'uniform', 'heur' o 'data'")

    agent = QLearningAgent(
        n_actions=env.action_space.n,
        discretizer=disc,
        alpha=alpha,
        gamma=gamma,
        epsilon_start=eps_start,
        epsilon_end=eps_end,
        epsilon_decay=eps_decay,
    )
    return agent


def main():
    disc_names = ["uniform", "heur", "data"]
    episodes = 800
    base_seed = 42

    for disc_name in disc_names:
        print(f"\n=== Discretizador: {disc_name} ===")
        env = make_env("CartPole-v1", seed=base_seed, render=False)
        agent = build_agent_with_disc(disc_name, env)

        t0 = time.time()
        out = train(agent, env, episodes=episodes, seed=base_seed, render=False)
        dt = time.time() - t0

        rewards = out["rewards"]
        ma = out["ma"]

        print(f"Tiempo de entrenamiento: {dt:.1f}s")
        print(f"Última recompensa: {rewards[-1]:.1f}")
        print(f"Media móvil final (50 eps): {ma[-1]:.1f}")

        mu, sd, _ = evaluate(agent, env, episodes=20, seed=base_seed + 999)
        print(f"Eval greedy (20 eps): mean={mu:.1f} ± {sd:.1f}")

        env.close()


if __name__ == "__main__":
    main()
