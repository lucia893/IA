from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt

def plot_learning(rewards: np.ndarray,
                  moving: np.ndarray,
                  title: str = "Aprendizaje",
                  xlabel: str = "Episodio",
                  ylabel: str = "Recompensa"):
    plt.figure()
    plt.plot(rewards, label="reward por episodio")
    if moving is not None and len(moving) == len(rewards):
        plt.plot(moving, label="media móvil (50)")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_compare_curves(series: list[tuple[np.ndarray, np.ndarray, str]],
                        title: str = "Comparación de curvas",
                        xlabel: str = "Episodio",
                        ylabel: str = "Recompensa"):
    """
    series: lista de (rewards, moving_avg, etiqueta)
    """
    plt.figure()
    for rewards, moving, label in series:
        if moving is not None and len(moving) == len(rewards):
            plt.plot(moving, label=f"{label} (MA50)")
        else:
            plt.plot(rewards, label=label)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_bars(labels: list[str],
              means: list[float],
              stds: list[float] | None = None,
              title: str = "Comparación (mean ± std)",
              xlabel: str = "",
              ylabel: str = "Puntaje (greedy)"):
    x = np.arange(len(labels))
    plt.figure()
    if stds is None:
        plt.bar(x, means)
    else:
        plt.bar(x, means, yerr=stds, capsize=4)
    plt.xticks(x, labels, rotation=15)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.show()
