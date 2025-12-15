import math
import numpy as np
import gymnasium as gym

RANGES = {
    "x": (-2.4, 2.4),                              
    "xdot": (-3.0, 3.0),                            
    "theta": (-12 * math.pi/180, 12 * math.pi/180), 
    "thetadot": (-3.5, 3.5),                       
}

def clip_obs(obs: np.ndarray) -> np.ndarray:
    x, xdot, th, thdot = obs
    x     = np.clip(x,     *RANGES["x"])
    xdot  = np.clip(xdot,  *RANGES["xdot"])
    th    = np.clip(th,    *RANGES["theta"])
    thdot = np.clip(thdot, *RANGES["thetadot"])
    return np.array([x, xdot, th, thdot], dtype=np.float32)

def make_env(env_id: str = "CartPole-v1", seed: int | None = None, render: bool = False):
    render_mode = "human" if render else None
    env = gym.make(env_id, render_mode=render_mode)
    if seed is not None:
        env.reset(seed=seed)
        np.random.seed(seed)
    return env
