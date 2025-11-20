import numpy as np
from .env import RANGES, clip_obs

# discretizar implica definir un intervalo de valores para cada feature (el mismo que ya existe, continuo; es decir, el rango en el que se puede mover el 
# carro va a ser el mismo que este) y dentro de ese intervalo que deifnimmos, hacemos "cortes" que dividen ese intervalo en "bins" (sub-intervalos).
# ahora las prgeuntas son, ¿cuántos bins por feature? y qué determina esa elección?

# En RL, el estado es toda la información que describe la situación actual del entorno.
# En CartPole, el estado es un vector con 4 features (variables):
# x → posición del carrito
# ẋ (xdot) → velocidad del carrito
# θ (theta) → ángulo del palo
# θ̇ (thetadot) → velocidad angular del palo
# Son las características del sistema físico que el algoritmo necesita para tomar decisiones.

# en que se basa la cantidad de bins? en un trade off entre la resolución necesaria para aprender una buena política (muy pocos bins, va a ser poco preciso)
# y la complejidad computacional (muchos bins, va a aprender muuy lento)

class DiscretizerUniform:
    """
    Discretización UNIFORME por bins por feature (VARIANTE A).
    Parámetro típico: (6, 6, 12, 12)
    """
    def __init__(self, bins_per_feature=(6, 6, 12, 12)): # definimos esa cantidad de bins para cada feature porque el angulo y velocidad angular
        bx, bxd, bth, bthd = bins_per_feature            # tienden a necesitar más resolución para capturar el comportamiento dinámico del sistema
        # Cortes equiespaciados; tomamos los internos (sin extremos)
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
    """
    Discretización HEURÍSTICA (VARIANTE B):
    - Menos bins en x y xdot.
    - Más densidad de cortes cerca de 0 para theta y thetadot.
    """
    def __init__(self):
        def centered_edges(a: float, b: float, n_cortes: int):
            # genera cortes "curvados" para concentrarlos cerca de 0
            xs = np.linspace(0, 1, n_cortes + 1)[1:-1]   # (0,1) sin extremos
            xs = (xs - 0.5) * 2                          # [-1, 1]
            xs = np.sign(xs) * (np.abs(xs) ** 1.5)       # curva más densa cerca de 0
            return np.interp(xs, [-1, 1], [a, b])

        self.cuts = [
            np.linspace(*RANGES["x"], 4 + 1)[1:-1],                 # x: 3 cortes (4 bins)
            np.linspace(*RANGES["xdot"], 4 + 1)[1:-1],              # xdot: 3 cortes
            centered_edges(*RANGES["theta"], 14),                    # theta: 14 cortes (~15 bins)
            centered_edges(*RANGES["thetadot"], 14),                 # thetadot: 14 cortes
        ]

    def __call__(self, obs: np.ndarray) -> tuple[int, int, int, int]:
        o = clip_obs(obs)
        idxs = [int(np.digitize(v, c)) for v, c in zip(o, self.cuts)]
        return tuple(idxs)
    
class DiscretizerDataDriven:
    """
    Discretización DATA-DRIVEN (VARIANTE C):
    Aprende los cortes a partir de observaciones reales del entorno.
    Los cortes se eligen como cuantiles de la distribución observada,
    de forma que cada bin reciba aproximadamente la misma cantidad de muestras.
    """
    def __init__(self, cuts: list[np.ndarray]):
        # cuts[i] son los cortes para la feature i
        self.cuts = cuts

    def __call__(self, obs: np.ndarray) -> tuple[int, int, int, int]:
        o = clip_obs(obs)
        idxs = [int(np.digitize(v, c)) for v, c in zip(o, self.cuts)]
        return tuple(idxs)


def learn_data_driven_cuts(env,
                           bins_per_feature=(6, 6, 12, 12),
                           episodes: int = 50,
                           seed: int = 123) -> list[np.ndarray]:
    """
    Recorre el entorno con una política aleatoria para recolectar observaciones
    y construye cortes por feature usando cuantiles.
    Esto NO entrena al agente; solo mira cómo se mueve el sistema.
    """
    all_obs = []

    rng = np.random.RandomState(seed)
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        while not done:
            all_obs.append(obs)
            # acción aleatoria
            action = env.action_space.sample()
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

    obs_arr = np.array(all_obs)  # shape: (N, 4) en CartPole

    cuts: list[np.ndarray] = []
    for i, n_bins in enumerate(bins_per_feature):
        # cuantiles internos (sin 0 ni 1)
        qs = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]
        feat_values = obs_arr[:, i]
        feat_cuts = np.quantile(feat_values, qs)
        cuts.append(feat_cuts)

    return cuts

