from agent import agent
from GameBoard import GameBoard
import numpy as np


class RandomAgent(agent):
  def init(self):
    pass

  def play(self, board:GameBoard):
    return np.random.randint(0, 4)

  def heuristic_utility(self, board: GameBoard):
    return 0
