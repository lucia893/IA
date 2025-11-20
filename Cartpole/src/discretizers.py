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

class DiscretizerUniform:
    """
    Discretización UNIFORME por bins por feature (VARIANTE A).
    Parámetro típico: (6, 6, 12, 12)
    """
    def __init__(self, bins_per_feature=(6, 6, 12, 12)):
        bx, bxd, bth, bthd = bins_per_feature
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
