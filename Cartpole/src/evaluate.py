# src/mbd/evaluate.py
from __future__ import annotations
import numpy as np

def evaluate(agent,
             env,
             episodes: int = 20,
             seed: int = 999,
             greedy: bool = True) -> tuple[float, float, np.ndarray]:
    """
    Evalúa el agente en 'episodes' episodios.
    Si greedy=True fuerza eps=0 (sin exploración).
    Retorna (mean, std, scores).
    """
    eps_backup = agent.eps
    if greedy:
        agent.eps = 0.0

    scores = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        done, total = False, 0.0
        while not done:
            a, _ = agent.select_action(obs)
            obs, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            total += r
        scores.append(total)

    if greedy:
        agent.eps = eps_backup

    scores = np.array(scores, dtype=float)
    return scores.mean(), scores.std(), scores

def evaluate_with_success(agent,
                          env,
                          episodes: int = 50,
                          seed: int = 2025,
                          success_threshold: float = 500.0) -> dict:
    """
    Evalúa greedy y reporta:
      - mean, std, scores
      - success_rate: % de episodios con score >= threshold (500 pasos por defecto)
    """
    mu, sd, scores = evaluate(agent, env, episodes=episodes, seed=seed, greedy=True)
    success = float(np.mean(scores >= success_threshold))
    return {
        "mean": mu,
        "std": sd,
        "success_rate": success,
        "scores": scores
    }
