# src/mbd/env.py
import math
import numpy as np
import gymnasium as gym

# Rangos razonables para "recortar" la observación continua (x, xdot, theta, thetadot)
RANGES = {
    "x": (-2.4, 2.4),                               # posición del carro
    "xdot": (-3.0, 3.0),                            # velocidad del carro
    "theta": (-12 * math.pi/180, 12 * math.pi/180), # ángulo del palo (±12°)
    "thetadot": (-3.5, 3.5),                        # velocidad angular
}

def clip_obs(obs: np.ndarray) -> np.ndarray:
    x, xdot, th, thdot = obs
    x     = np.clip(x,     *RANGES["x"])
    xdot  = np.clip(xdot,  *RANGES["xdot"])
    th    = np.clip(th,    *RANGES["theta"])
    thdot = np.clip(thdot, *RANGES["thetadot"])
    return np.array([x, xdot, th, thdot], dtype=np.float32)

def make_env(env_id: str = "CartPole-v1", seed: int | None = None, render: bool = False):
    """
    Crea el entorno Gymnasium y setea la semilla (si se pasa).
    Retorna (env, reset) listo para usar.
    """
    render_mode = "human" if render else None
    env = gym.make(env_id, render_mode=render_mode)
    if seed is not None:
        env.reset(seed=seed)
        np.random.seed(seed)
    return env
