from __future__ import annotations
from src.agents import QLearningAgent, StochasticQLearningAgent, DoubleQLearningAgent
import time
import numpy as np
import wandb

from src.env import make_env
from src.discretizers import (
    DiscretizerUniform,
    DiscretizerHeuristic,
    DiscretizerDataDriven,
    learn_data_driven_cuts,
)
from src.agents import DoubleQLearningAgent, QLearningAgent, StochasticQLearningAgent
from src.train import train
from src.evaluate import evaluate


def build_discretizer_from_cfg(cfg, env):
    if cfg.disc == "uniform":
        return DiscretizerUniform(bins_per_feature=(6, 6, 12, 12))
    elif cfg.disc == "heur":
        return DiscretizerHeuristic()
    elif cfg.disc == "data":
        cuts = learn_data_driven_cuts(
            env,
            bins_per_feature=(6, 6, 12, 12),
            episodes=cfg.disc_data_episodes,
            seed=cfg.seed,
        )
        return DiscretizerDataDriven(cuts)
    else:
        raise ValueError("disc debe ser 'uniform', 'heur' o 'data'")


def build_agent_from_cfg(cfg, env, discretizer):
    common_kwargs = dict(
        alpha=cfg.alpha,
        gamma=cfg.gamma,
        epsilon_start=cfg.eps_start,
        epsilon_end=cfg.eps_end,
        epsilon_decay=cfg.eps_decay,
    )

    if cfg.agent == "ql":
        return QLearningAgent(
            n_actions=env.action_space.n,
            discretizer=discretizer,
            **common_kwargs,
        )
    elif cfg.agent == "sql":
        return StochasticQLearningAgent(
            n_actions=env.action_space.n,
            discretizer=discretizer,
            k_subset=cfg.k_subset,
            **common_kwargs,
        )
    elif cfg.agent == "double":
        return DoubleQLearningAgent(
            n_actions=env.action_space.n,
            discretizer=discretizer,
            **common_kwargs,
    )
    else:
        raise ValueError("agent debe ser 'ql' o 'sql'")


def main():
    run = wandb.init(
        project="IA-Cartpole",
        config={
            "agent": "ql",
            "disc": "uniform",
            "alpha": 0.1,
            "gamma": 0.99,
            "eps_start": 1.0,
            "eps_end": 0.02,
            "eps_decay": 0.995,
            "k_subset": 1,
            "episodes": 100,
            "seed": 42,
            "disc_data_episodes": 50,
        },
    )
    cfg = wandb.config

    env = make_env("CartPole-v1", seed=cfg.seed, render=False)
    discretizer = build_discretizer_from_cfg(cfg, env)
    agent = build_agent_from_cfg(cfg, env, discretizer)

    t0 = time.time()
    out = train(
        agent,
        env,
        episodes=cfg.episodes,
        seed=cfg.seed,
        render=False,
        wandb_run=None,  
    )
    dt = time.time() - t0

    rewards = out["rewards"]
    ma = out["ma"]

    for ep, (r, m) in enumerate(zip(rewards, ma)):
        wandb.log({
            "episode": ep,
            "reward": float(r),
            "ma50": float(m),
        })

    mu, sd, _ = evaluate(agent, env, episodes=20, seed=cfg.seed + 1234)

    wandb.log({
        "final_reward": float(rewards[-1]),
        "final_ma50": float(ma[-1]),
        "train_time_s": dt,
        "eval_mean": mu,
        "eval_std": sd,
    })

    env.close()
    run.finish()


if __name__ == "__main__":
    main()
