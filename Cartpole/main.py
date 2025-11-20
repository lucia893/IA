import argparse
import pickle
import time
import os
from collections import defaultdict
import numpy as np

from src.env import make_env
from src.discretizers import DiscretizerUniform, DiscretizerHeuristic
from src.agents import QLearningAgent, StochasticQLearningAgent
from src.train import train, moving_avg
from src.evaluate import evaluate


def build_discretizer(name: str):
    """Construye el discretizador elegido por parámetro."""
    if name == "uniform":
        return DiscretizerUniform(bins_per_feature=(6, 6, 12, 12))
    elif name == "heur":
        return DiscretizerHeuristic()
    raise ValueError("discretizador debe ser 'uniform' o 'heur'")


def build_agent(name: str, n_actions: int, discretizer, args):
    """Construye el agente según el tipo elegido."""
    kwargs = dict(
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon_start=args.eps_start,
        epsilon_end=args.eps_end,
        epsilon_decay=args.eps_decay,
    )
    if name == "ql":
        return QLearningAgent(n_actions=n_actions, discretizer=discretizer, **kwargs)
    elif name == "sql":
        return StochasticQLearningAgent(
            n_actions=n_actions,
            discretizer=discretizer,
            k_subset=args.k_subset,
            **kwargs,
        )
    raise ValueError("agent debe ser 'ql' o 'sql'")


def main():
    parser = argparse.ArgumentParser(description="CartPole Q-Learning / Stochastic Q-Learning")

    # modos
    parser.add_argument("--mode", choices=["train", "eval"], required=True)
    parser.add_argument("--agent", choices=["ql", "sql"], default="ql")
    parser.add_argument("--disc", choices=["uniform", "heur"], default="uniform")

    # hiperparámetros
    parser.add_argument("--episodes", type=int, default=800)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--eps-start", type=float, default=1.0)
    parser.add_argument("--eps-end", type=float, default=0.02)
    parser.add_argument("--eps-decay", type=float, default=0.995)
    parser.add_argument("--k-subset", type=int, default=1, help="solo para SQL")

    # otros parámetros
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--render", action="store_true", help="muestra ventana del CartPole")
    parser.add_argument("--model-out", type=str, default="models/model.pkl")
    parser.add_argument("--model-in", type=str, default="models/model.pkl")

    args = parser.parse_args()

    # crear entorno y discretizador
    env = make_env("CartPole-v1", seed=args.seed, render=args.render)
    discretizer = build_discretizer(args.disc)
    agent = build_agent(args.agent, env.action_space.n, discretizer, args)

    # modo entrenamiento
    if args.mode == "train":
        print(f"\nEntrenando [{args.agent.upper()}] con discretizador [{args.disc}] ...")
        start_time = time.time()

        out = train(agent, env, episodes=args.episodes, seed=args.seed, render=args.render)
        elapsed = time.time() - start_time

        rewards = out["rewards"]
        ma = out["ma"]

        print(f"\nEntrenamiento finalizado en {elapsed:.1f}s")
        print(f"Última recompensa: {rewards[-1]:.1f}")
        print(f"Media móvil final (50 eps): {ma[-1]:.1f}")

        # crear carpeta models si no existe
        os.makedirs("models", exist_ok=True)

        # guardar modelo (Q-table + metadatos)
        payload = {
            "Q": dict(agent.Q),
            "agent": args.agent,
            "disc": args.disc,
            "alpha": args.alpha,
            "gamma": args.gamma,
            "eps_end": args.eps_end,
            "k_subset": getattr(args, "k_subset", 1),
        }
        with open(args.model_out, "wb") as f:
            pickle.dump(payload, f)

        print(f"\nModelo guardado en: {args.model_out}")

        # evaluación greedy final
        mu, sd, _ = evaluate(agent, env, episodes=20, seed=args.seed + 1234)
        print(f"\nEvaluación greedy: mean={mu:.1f} ± {sd:.1f}")

    # modo evaluación
    elif args.mode == "eval":
        print(f"\nEvaluando modelo: {args.model_in}")
        with open(args.model_in, "rb") as f:
            data = pickle.load(f)

        # reconstruir agente y su Q-table
        agent_loaded = build_agent(
            data.get("agent", "ql"), env.action_space.n, discretizer, args
        )
        agent_loaded.Q = defaultdict(lambda: np.zeros(env.action_space.n, dtype=np.float32))
        for k, v in data["Q"].items():
            agent_loaded.Q[k] = np.array(v, dtype=np.float32)

        mu, sd, scores = evaluate(agent_loaded, env, episodes=20, seed=args.seed + 9999)
        print(f"\nEvaluación greedy: mean={mu:.1f} ± {sd:.1f}")
        print("Scores individuales:", scores)

        # render opcional durante eval
        if args.render:
            print("\nReproduciendo episodio renderizado (greedy)...")
            obs, _ = env.reset(seed=args.seed)
            done = False
            agent_loaded.eps = 0.0  # modo greedy
            import time as t
            while not done:
                env.render()
                a, _ = agent_loaded.select_action(obs)
                obs, _, terminated, truncated, _ = env.step(a)
                done = terminated or truncated
                t.sleep(0.01)  # pequeña pausa para ver la animación
            env.close()


if __name__ == "__main__":
    main()
