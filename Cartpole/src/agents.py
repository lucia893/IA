# src/mbd/agents.py
from collections import defaultdict, deque
import numpy as np

class QLearningAgent:
    """
    Agente Q-Learning tabular con política epsilon-greedy.
    - Q[s] es un vector de tamaño n_actions con los valores Q(s, a).
    - 'discretizer' transforma obs continua -> estado discreto (tupla).
    """
    def __init__(self,
                 n_actions: int,
                 discretizer,
                 alpha: float = 0.1,
                 gamma: float = 0.99,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.02,
                 epsilon_decay: float = 0.995):
        self.n_actions = n_actions
        self.discretizer = discretizer
        self.alpha = alpha
        self.gamma = gamma
        self.eps = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.Q = defaultdict(lambda: np.zeros(n_actions, dtype=np.float32))

    def select_action(self, obs: np.ndarray) -> tuple[int, tuple]:
        s = self.discretizer(obs)
        # epsilon-greedy
        if np.random.rand() < self.eps:
            return np.random.randint(self.n_actions), s
        return int(np.argmax(self.Q[s])), s

    def update(self, s: tuple, a: int, r: float, s_next: tuple, done: bool):
        # target estándar: r + γ * max_a' Q[s', a'] (o solo r si done)
        if done:
            target = r
        else:
            target = r + self.gamma * np.max(self.Q[s_next])
        self.Q[s][a] += self.alpha * (target - self.Q[s][a])

    def decay_epsilon(self):
        self.eps = max(self.epsilon_end, self.eps * self.epsilon_decay)

class StochasticQLearningAgent(QLearningAgent):
    """
    Igual a QLearningAgent pero el 'target' toma el max sobre un SUBCONJUNTO
    aleatorio de acciones (tamaño k_subset), en vez de todas las acciones.
    - En CartPole hay 2 acciones; k=2 ≈ Q-Learning estándar.
    - k=1 sirve para demostrar la técnica del paper (submuestreo del espacio de acciones).
    """
    def __init__(self, n_actions: int, discretizer, k_subset: int = 1, **kwargs):
        super().__init__(n_actions, discretizer, **kwargs)
        self.k_subset = max(1, min(k_subset, n_actions))

    def update(self, s: tuple, a: int, r: float, s_next: tuple, done: bool):
        if done:
            target = r
        else:
            idx = np.random.choice(self.n_actions, size=self.k_subset, replace=False)
            target = r + self.gamma * np.max(self.Q[s_next][idx])
        self.Q[s][a] += self.alpha * (target - self.Q[s][a])


class DoubleQLearningAgent:
    def __init__(
        self,
        n_actions: int,
        discretizer,
        alpha: float,
        gamma: float,
        epsilon_start: float,
        epsilon_end: float,
        epsilon_decay: float,
    ):
        self.n_actions = n_actions
        self.discretizer = discretizer
        self.alpha = alpha
        self.gamma = gamma
        self.eps = epsilon_start
        self.eps_end = epsilon_end
        self.eps_decay = epsilon_decay

        # Dos tablas Q independientes
        self.Q1 = defaultdict(lambda: np.zeros(n_actions, dtype=np.float32))
        self.Q2 = defaultdict(lambda: np.zeros(n_actions, dtype=np.float32))

        # Para compatibilidad con código que espera .Q (por ejemplo main.py)
        self.Q = self.Q1

    def select_action(self, obs):
        """ε-greedy usando Q1 + Q2 para explotar."""
        s = self.discretizer(obs)
        if np.random.rand() < self.eps:
            a = np.random.randint(self.n_actions)
        else:
            q_sum = self.Q1[s] + self.Q2[s]
            a = int(np.argmax(q_sum))
        return a, s

    def update(self, s, a, r, s2, done: bool):
        """Actualiza una de las dos tablas al azar (Double Q-Learning)."""
        if np.random.rand() < 0.5:
            # Actualizar Q1 usando argmax en Q1 y valor de Q2
            q = self.Q1
            q_next_sel = self.Q1
            q_next_val = self.Q2
        else:
            # Actualizar Q2 usando argmax en Q2 y valor de Q1
            q = self.Q2
            q_next_sel = self.Q2
            q_next_val = self.Q1

        if done:
            target = r
        else:
            a_star = int(np.argmax(q_next_sel[s2]))
            target = r + self.gamma * q_next_val[s2][a_star]

        q[s][a] += self.alpha * (target - q[s][a])

    def decay_epsilon(self):
        self.eps = max(self.eps_end, self.eps * self.eps_decay)
