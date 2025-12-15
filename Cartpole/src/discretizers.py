import numpy as np
from .env import RANGES, clip_obs

class DiscretizerUniform:
   
    def __init__(self, bins_per_feature=(6, 6, 12, 12)): 
        bx, bxd, bth, bthd = bins_per_feature            
        self.cuts = [
            np.linspace(*RANGES["x"], bx + 1)[1:-1],
            np.linspace(*RANGES["xdot"], bxd + 1)[1:-1],
            np.linspace(*RANGES["theta"], bth + 1)[1:-1],
            np.linspace(*RANGES["thetadot"], bthd + 1)[1:-1],
        ]

    def __call__(self, obs: np.ndarray) -> tuple[int, int, int, int]:
        o = clip_obs(obs)
        idxs = [int(np.digitize(v, c)) for v, c in zip(o, self.cuts)]
        return tuple(idxs)

class DiscretizerHeuristic:
    
    def __init__(self):
        def centered_edges(a: float, b: float, n_cortes: int):
            xs = np.linspace(0, 1, n_cortes + 1)[1:-1]   
            xs = (xs - 0.5) * 2                          
            xs = np.sign(xs) * (np.abs(xs) ** 1.5)       
            return np.interp(xs, [-1, 1], [a, b])

        self.cuts = [
            np.linspace(*RANGES["x"], 4 + 1)[1:-1],                 
            np.linspace(*RANGES["xdot"], 4 + 1)[1:-1],              
            centered_edges(*RANGES["theta"], 14),                   
            centered_edges(*RANGES["thetadot"], 14),                
        ]

    def __call__(self, obs: np.ndarray) -> tuple[int, int, int, int]:
        o = clip_obs(obs)
        idxs = [int(np.digitize(v, c)) for v, c in zip(o, self.cuts)]
        return tuple(idxs)
    
class DiscretizerDataDriven:
   
    def __init__(self, cuts: list[np.ndarray]):
        self.cuts = cuts

    def __call__(self, obs: np.ndarray) -> tuple[int, int, int, int]:
        o = clip_obs(obs)
        idxs = [int(np.digitize(v, c)) for v, c in zip(o, self.cuts)]
        return tuple(idxs)


def learn_data_driven_cuts(env,
                           bins_per_feature=(6, 6, 12, 12),
                           episodes: int = 50,
                           seed: int = 123) -> list[np.ndarray]:
   
    all_obs = []

    rng = np.random.RandomState(seed)
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        while not done:
            all_obs.append(obs)
            action = env.action_space.sample()
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

    obs_arr = np.array(all_obs)  

    cuts: list[np.ndarray] = []
    for i, n_bins in enumerate(bins_per_feature):
        qs = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]
        feat_values = obs_arr[:, i]
        feat_cuts = np.quantile(feat_values, qs)
        cuts.append(feat_cuts)

    return cuts

