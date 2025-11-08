# src/mbd/train.py
from __future__ import annotations
import time, csv, itertools
from collections import deque
from dataclasses import dataclass, asdict
from typing import Callable, Dict, Any, Iterable
import numpy as np

def moving_avg(x: Iterable[float], w: int = 50) -> np.ndarray:
    x = list(x)
    if not x: return np.array([])
    out, s = [], 0.0
    for i, v in enumerate(x):
        s += v
        if i >= w:
            s -= x[i - w]
        out.append(s / min(i + 1, w))
    return np.array(out, dtype=float)

def train(agent,
          env,
          episodes: int = 800,
          seed: int = 42,
          render: bool = False) -> Dict[str, np.ndarray]:
    """
    Entrena un agente tabular (QL o StochQL) en el env dado.
    Devuelve dict con recompensas por episodio y media móvil.
    """
    rewards = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        done, total = False, 0.0
        while not done:
            if render: env.render()
            a, s = agent.select_action(obs)
            obs2, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            s2 = agent.discretizer(obs2)
            agent.update(s, a, r, s2, done)
            obs = obs2
            total += r
        agent.decay_epsilon()
        rewards.append(total)
    return {
        "rewards": np.array(rewards, dtype=float),
        "ma": moving_avg(rewards, 50)
    }

@dataclass
class GSConfig:
    agent_name: str        # 'ql' o 'sql'
    disc_name: str         # 'uniform' o 'heur'
    alpha: float
    gamma: float
    eps_start: float
    eps_end: float
    eps_decay: float
    k_subset: int = 1      # solo aplica si agent_name == 'sql'
    episodes: int = 800
    seed: int = 42
    runs: int = 1          # cuántas repeticiones por config

@dataclass
class GSResult:
    agent_name: str
    disc_name: str
    alpha: float
    gamma: float
    eps_start: float
    eps_end: float
    eps_decay: float
    k_subset: int
    episodes: int
    seed: int
    run_index: int
    train_time_s: float
    last_reward: float
    last_ma50: float
    eval_mean: float
    eval_std: float

def grid_search(env_factory: Callable[[], Any],
                agent_factory: Callable[..., Any],
                evaluate_fn: Callable[..., tuple[float, float, np.ndarray]],
                param_grid: Dict[str, Iterable],
                base: GSConfig,
                csv_out: str | None = None) -> list[GSResult]:
    """
    Barrido simple de hiperparámetros.
    - env_factory(): retorna un env listo (llamar uno por corrida).
    - agent_factory(...): construye el agente con params (ver main.py helpers).
    - evaluate_fn(agent, env, episodes, seed): eval greedy -> (mean, std, arr)
    - param_grid: dict con iterables de valores a combinar.
    - base: valores base (GSConfig) que se pisan con las combinaciones.
    - csv_out: si se pasa, guarda resultados línea a línea.
    """
    keys = list(param_grid.keys())
    combos = list(itertools.product(*[param_grid[k] for k in keys]))

    results: list[GSResult] = []
    if csv_out:
        with open(csv_out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=[*GSResult.__annotations__.keys()])
            w.writeheader()

    for combo in combos:
        cfg_dict = asdict(base)
        for k, v in zip(keys, combo):
            cfg_dict[k] = v
        cfg = GSConfig(**cfg_dict)

        for run_idx in range(cfg.runs):
            env = env_factory()
            agent = agent_factory(cfg.agent_name, env.action_space.n, cfg)

            t0 = time.time()
            out = train(agent, env, episodes=cfg.episodes, seed=cfg.seed + 1000*run_idx, render=False)
            dt = time.time() - t0

            last_reward = float(out["rewards"][-1])
            last_ma50 = float(out["ma"][-1])

            mu, sd, _ = evaluate_fn(agent, env, episodes=20, seed=cfg.seed + 5000*run_idx)

            res = GSResult(
                agent_name=cfg.agent_name,
                disc_name=cfg.disc_name,
                alpha=cfg.alpha,
                gamma=cfg.gamma,
                eps_start=cfg.eps_start,
                eps_end=cfg.eps_end,
                eps_decay=cfg.eps_decay,
                k_subset=cfg.k_subset,
                episodes=cfg.episodes,
                seed=cfg.seed + 1000*run_idx,
                run_index=run_idx,
                train_time_s=dt,
                last_reward=last_reward,
                last_ma50=last_ma50,
                eval_mean=mu,
                eval_std=sd
            )
            results.append(res)

            if csv_out:
                with open(csv_out, "a", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=[*GSResult.__annotations__.keys()])
                    w.writerow(asdict(res))

            env.close()

    return results
