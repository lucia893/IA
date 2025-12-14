import wandb
from datetime import datetime

from ExpectiMaxAgent import ExpectimaxAgent
from MiniMaxAgent import MinimaxAgent
from train import train
from evaluate import evaluate


def build_agent_from_cfg(cfg):
    if cfg.agent == "expectimax":
        return ExpectimaxAgent(
            depth=cfg.depth,
            empty_weight=cfg.empty_weight,
            smooth_weight=cfg.smooth_weight,
            max_tile_weight=cfg.max_tile_weight,
            mono_weight=cfg.mono_weight,
            corner_weight=cfg.corner_weight,
            value_weight=cfg.value_weight,
            p_two=cfg.p_two,
        )
    elif cfg.agent == "minimax":
        return MinimaxAgent(
            depth=cfg.depth,
            empty_weight=cfg.empty_weight,
            smooth_weight=cfg.smooth_weight,
            max_tile_weight=cfg.max_tile_weight,
            mono_weight=cfg.mono_weight,
            corner_weight=cfg.corner_weight,
        )
    else:
        raise ValueError("Agente no válido")


def main():

    run = wandb.init(
        project="IA-2048",
        config={
            "agent": "expectimax",
            "depth": 4,
            "empty_weight": 30000.0,
            "smooth_weight": 1.0,
            "max_tile_weight": 0.5,
            "mono_weight": 5.0,
            "corner_weight": 50.0,
            "value_weight": 0.0001,
            "p_two": 0.9,
            "episodes": 50,
            "seed": 42,
            "eval_episodes": 30,
        }
    )
    cfg = wandb.config

    agent = build_agent_from_cfg(cfg)

    start = datetime.now()
    out = train(agent, episodes=cfg.episodes, seed=cfg.seed, wandb_run=run)
    dt = (datetime.now() - start).total_seconds()

    eval_stats = evaluate(agent, episodes=cfg.eval_episodes, seed=cfg.seed + 1234)

    wandb.log({
        "final_reward": float(out["rewards"][-1]),
        "final_max_tile": int(out["max_tiles"][-1]),
        "train_win_rate": float(out["wins"].mean()),
        "train_time_s": dt,
        **eval_stats
    })

    run.finish()


if __name__ == "__main__":
    main()
