from agent import agent
from GameBoard import GameBoard
import math
import numpy as np
import random


class ExpectimaxAgent(agent):
    def __init__(self, depth=3,
                 empty_weight=30000.0,
                 smooth_weight=1.0,
                 max_tile_weight=0.5,
                 mono_weight=5.0,
                 corner_weight=50.0,
                 value_weight=0.0001,
                 p_two=0.9):
        """
        depth: profundidad máxima del árbol (plies).
        p_two: probabilidad de que la nueva ficha sea un 2 (por defecto 0.9).
        El resto son pesos de la heurística.
        """
        self.depth = depth
        self.empty_weight = empty_weight
        self.smooth_weight = smooth_weight
        self.max_tile_weight = max_tile_weight
        self.mono_weight = mono_weight
        self.corner_weight = corner_weight
        self.value_weight = value_weight
        self.p_two = p_two

        self.nodes_expanded = 0


    def play(self, board: GameBoard) -> int:
        """
        Decide qué movimiento hacer (0=LEFT, 1=UP, 2=RIGHT, 3=DOWN)
        usando Expectimax.
        """
        moves = board.get_available_moves()
        if not moves:
            return 0  

        best_move = None
        best_value = -math.inf
        self.nodes_expanded = 0

        for move in moves:
            child = board.clone()
            child.move(move)

            value = self._chance_value(child, depth=self.depth - 1)
            if value > best_value or best_move is None:
                best_value = value
                best_move = move

        if best_move is None:
            best_move = random.choice(moves)

        return best_move

    def heuristic_utility(self, board: GameBoard) -> float:
        """
        Evalúa el tablero combinando:
        - cantidad de casillas vacías
        - suavidad del tablero
        - valor de la ficha máxima
        - monotonicidad (filas/columnas ordenadas)
        - bonus por tener el máximo en una esquina
        - valor total del tablero (suma de cuadrados)
        """
        mat = np.array(board.grid, dtype=float)

        empty_cells = np.sum(mat == 0)
        empty_score = empty_cells * self.empty_weight

        max_tile = np.max(mat)
        max_tile_score = max_tile * self.max_tile_weight

        smooth_score = self._compute_smoothness(mat) * self.smooth_weight

        mono_score = self._compute_monotonicity(mat) * self.mono_weight

        corner_score = self._corner_max_bonus(mat) * self.corner_weight

        value_score = np.sum(mat ** 2) * self.value_weight

        return (empty_score +
                smooth_score +
                max_tile_score +
                mono_score +
                corner_score +
                value_score)

    def _max_value(self, board: GameBoard, depth: int) -> float:
        """
        Nodo MAX: turno del jugador.
        """
        self.nodes_expanded += 1

        moves = board.get_available_moves()
        if depth == 0 or not moves:
            return self.heuristic_utility(board)

        value = -math.inf

        for move in moves:
            child = board.clone()
            child.move(move)

            child_value = self._chance_value(child, depth - 1)
            value = max(value, child_value)

        if value == -math.inf:
            return self.heuristic_utility(board)
        return value

    def _chance_value(self, board: GameBoard, depth: int) -> float:
        self.nodes_expanded += 1

        empty_cells = board.get_available_cells()
        if depth == 0 or not empty_cells:
            return self.heuristic_utility(board)

        n = len(empty_cells)
        if n == 0:
            return self.heuristic_utility(board)

        p_two = self.p_two
        p_four = 1.0 - p_two

        expected_value = 0.0

        for (i, j) in empty_cells:
            child2 = board.clone()
            child2.insert_tile((i, j), 2)
            v2 = self._max_value(child2, depth - 1)
            expected_value += (p_two / n) * v2

            child4 = board.clone()
            child4.insert_tile((i, j), 4)
            v4 = self._max_value(child4, depth - 1)
            expected_value += (p_four / n) * v4

        return expected_value


    def _compute_smoothness(self, mat: np.ndarray) -> float:
        smoothness = 0.0

        log_mat = np.zeros_like(mat)
        non_zero = mat > 0
        log_mat[non_zero] = np.log2(mat[non_zero])

        for i in range(4):
            for j in range(3):
                if log_mat[i, j] > 0 and log_mat[i, j + 1] > 0:
                    smoothness -= abs(log_mat[i, j] - log_mat[i, j + 1])

        for i in range(3):
            for j in range(4):
                if log_mat[i, j] > 0 and log_mat[i + 1, j] > 0:
                    smoothness -= abs(log_mat[i, j] - log_mat[i + 1, j])

        return smoothness

    def _compute_monotonicity(self, mat: np.ndarray) -> float:
        mono = 0.0

        log_mat = np.zeros_like(mat)
        non_zero = mat > 0
        log_mat[non_zero] = np.log2(mat[non_zero])

        for i in range(4):
            current_row = log_mat[i, :]
            inc = 0.0  
            dec = 0.0  
            for j in range(3):
                if current_row[j] > 0 and current_row[j + 1] > 0:
                    diff = current_row[j + 1] - current_row[j]
                    if diff > 0:
                        inc += diff
                    else:
                        dec -= diff   
            mono -= min(inc, dec)

        for j in range(4):
            current_col = log_mat[:, j]
            inc = 0.0
            dec = 0.0
            for i in range(3):
                if current_col[i] > 0 and current_col[i + 1] > 0:
                    diff = current_col[i + 1] - current_col[i]
                    if diff > 0:
                        inc += diff
                    else:
                        dec -= diff
            mono -= min(inc, dec)

        return mono

    def _corner_max_bonus(self, mat: np.ndarray) -> float:
       
        max_tile = np.max(mat)
        corners = [
            mat[0, 0],
            mat[0, 3],
            mat[3, 0],
            mat[3, 3],
        ]
        return 1.0 if max_tile in corners else 0.0
