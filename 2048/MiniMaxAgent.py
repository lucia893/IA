from Agent import Agent
from GameBoard import GameBoard
import math
import numpy as np
import random

# MinimaxAgent.py

from Agent import Agent
from GameBoard import GameBoard
import math
import numpy as np
import random


class MinimaxAgent(Agent):
    def __init__(self, depth=3,
                 empty_weight=10000.0,
                 smooth_weight=1.0,
                 max_tile_weight=1.0,
                 mono_weight=1000.0,
                 corner_weight=2000.0):
        """
        depth: profundidad máxima del árbol (en 'plies', niveles de jugador+oponente).
        empty_weight: peso para la cantidad de casillas vacías.
        smooth_weight: peso para la suavidad del tablero.
        max_tile_weight: peso para el valor de la ficha máxima.
        """
        self.depth = depth
        self.empty_weight = empty_weight
        self.smooth_weight = smooth_weight
        self.max_tile_weight = max_tile_weight
        self.mono_weight = mono_weight
        self.corner_weight = corner_weight

        # Para medir impacto de alpha-beta
        self.nodes_expanded = 0

    def play(self, board: GameBoard) -> int:
        """
        Decide qué movimiento hacer (0=LEFT, 1=UP, 2=RIGHT, 3=DOWN).
        Implementa la raíz del árbol Minimax.
        """
        moves = board.get_available_moves()
        if not moves:
            # Si no hay movimientos posibles, devolvemos algo por contrato (por ejemplo 0)
            return 0

        best_move = None
        best_value = -math.inf

        # Reiniciamos contador de nodos antes de la búsqueda (para estadísticas)
        self.nodes_expanded = 0

        for move in moves:
            # Simulamos el movimiento del jugador
            child = board.clone()
            child.move(move)

            # Llamamos al nivel MIN (oponente) con profundidad-1
            value = self._min_value(child,
                                    depth=self.depth - 1,
                                    alpha=-math.inf,
                                    beta=math.inf)

            if value > best_value or best_move is None:
                best_value = value
                best_move = move

        # Si por algún motivo no encontramos mejor, elegimos uno válido al azar
        if best_move is None:
            best_move = random.choice(moves)

        # Podrías imprimir self.nodes_expanded aquí para debug/experimentos
        # print("Nodos expandidos:", self.nodes_expanded)

        return best_move

    def _max_value(self, board: GameBoard, depth: int,
                   alpha: float, beta: float) -> float:
        """
        Nodo MAX de Minimax: turno del jugador.
        """
        self.nodes_expanded += 1

        # Condiciones de corte: profundidad 0 o sin movimientos
        moves = board.get_available_moves()
        if depth == 0 or not moves:
            return self.heuristic_utility(board)

        value = -math.inf

        for move in moves:
            child = board.clone()
            child.move(move)

            value = max(value, self._min_value(child, depth - 1, alpha, beta))

            alpha = max(alpha, value)
            # Poda alpha-beta
            if alpha >= beta:
                break

        # Si por alguna razón no pudimos mejorar value (por ejemplo todos los moves fueron inválidos),
        # devolvemos la heurística del estado actual
        if value == -math.inf:
            return self.heuristic_utility(board)

        return value

    def _min_value(self, board: GameBoard, depth: int,
                   alpha: float, beta: float) -> float:
        """
        Nodo MIN de Minimax: el 'oponente' elige la peor forma de agregar una ficha.
        """
        self.nodes_expanded += 1

        empty_cells = board.get_available_cells()

        # Si no hay dónde agregar fichas o profundidad 0, usamos heurística
        if depth == 0 or not empty_cells:
            return self.heuristic_utility(board)

        value = math.inf

        # El oponente puede elegir, para cada celda vacía, poner un 2 o un 4.
        for (i, j) in empty_cells:
            # Probamos poner un 2
            child2 = board.clone()
            child2.insert_tile((i, j), 2)
            value = min(value, self._max_value(child2, depth - 1, alpha, beta))
            beta = min(beta, value)
            if alpha >= beta:
                return value  # poda

            # Probamos poner un 4
            child4 = board.clone()
            child4.insert_tile((i, j), 4)
            value = min(value, self._max_value(child4, depth - 1, alpha, beta))
            beta = min(beta, value)
            if alpha >= beta:
                return value  # poda

        # Si por alguna razón no se actualizó value, devolvemos heurística
        if value == math.inf:
            return self.heuristic_utility(board)

        return value

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

        # 1) Espacios vacíos
        empty_cells = np.sum(mat == 0)
        empty_score = empty_cells * self.empty_weight

        # 2) Ficha máxima
        max_tile = np.max(mat)
        max_tile_score = max_tile * self.max_tile_weight

        # 3) Suavidad
        smooth_score = self._compute_smoothness(mat) * self.smooth_weight

        # 4) Monotonicidad
        mono_score = self._compute_monotonicity(mat) * self.mono_weight

        # 5) Max tile en esquina
        corner_score = self._corner_max_bonus(mat) * self.corner_weight

        # 6) Valor total del tablero (suma de cuadrados)
        value_score = np.sum(mat ** 2) * self.value_weight

        return (empty_score +
                smooth_score +
                max_tile_score +
                mono_score +
                corner_score +
                value_score)

    def _compute_smoothness(self, mat: np.ndarray) -> float:
        """
        Suavidad: sumatoria negativa de diferencias en log2 entre celdas vecinas.
        Tableros más "parejos" (fichas similares juntas) son mejores.
        """
        smoothness = 0.0

        log_mat = np.zeros_like(mat)
        non_zero = mat > 0
        log_mat[non_zero] = np.log2(mat[non_zero])

        # vecinos horizontales
        for i in range(4):
            for j in range(3):
                if log_mat[i, j] > 0 and log_mat[i, j + 1] > 0:
                    smoothness -= abs(log_mat[i, j] - log_mat[i, j + 1])

        # vecinos verticales
        for i in range(3):
            for j in range(4):
                if log_mat[i, j] > 0 and log_mat[i + 1, j] > 0:
                    smoothness -= abs(log_mat[i, j] - log_mat[i + 1, j])

        return smoothness

    def _compute_monotonicity(self, mat: np.ndarray) -> float:
        """
        Monotonicidad: premiar filas/columnas donde los valores cambian
        de forma más o menos ordenada (no suben y bajan caóticamente).
        """
        mono = 0.0

        log_mat = np.zeros_like(mat)
        non_zero = mat > 0
        log_mat[non_zero] = np.log2(mat[non_zero])

        # Filas
        for i in range(4):
            current_row = log_mat[i, :]
            inc = 0.0  # tendencia creciente
            dec = 0.0  # tendencia decreciente
            for j in range(3):
                if current_row[j] > 0 and current_row[j + 1] > 0:
                    diff = current_row[j + 1] - current_row[j]
                    if diff > 0:
                        inc += diff
                    else:
                        dec -= diff   # diff < 0
            mono -= min(inc, dec)

        # Columnas
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
        """
        Da 1 si la ficha máxima está en una esquina, 0 si no.
        Luego se multiplica por corner_weight en la heurística.
        """
        max_tile = np.max(mat)
        corners = [
            mat[0, 0],
            mat[0, 3],
            mat[3, 0],
            mat[3, 3],
        ]
        return 1.0 if max_tile in corners else 0.0

    def _compute_monotonicity(self, mat: np.ndarray) -> float:
        """
        Monotonicidad: premiar filas/columnas donde los valores cambian de forma
        más o menos ordenada (no suben y bajan caóticamente).
        """
        mono = 0.0

        log_mat = np.zeros_like(mat)
        non_zero = mat > 0
        log_mat[non_zero] = np.log2(mat[non_zero])

        # Filas
        for i in range(4):
            current_row = log_mat[i, :]
            # dos direcciones posibles: izquierda->derecha o derecha->izquierda
            inc = 0.0  # "creciente"
            dec = 0.0  # "decreciente"
            for j in range(3):
                if current_row[j] > 0 and current_row[j + 1] > 0:
                    diff = current_row[j + 1] - current_row[j]
                    if diff > 0:
                        inc += diff
                    else:
                        dec -= diff  # diff negativo, restamos negativo

            # elegimos la dirección que tenga menor "ruptura"
            mono -= min(inc, dec)

        # Columnas
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

        # mono será negativo cuando haya mucha "ruptura" en el orden
        # al multiplicarlo por mono_weight (positivo) en la heurística,
        # monoticidad alta (poca ruptura) contribuye más.
        return mono

    def _corner_max_bonus(self, mat: np.ndarray) -> float:
        """
        Da 1 si la ficha máxima está en una esquina, 0 si no.
        Así luego lo multiplicamos por corner_weight.
        """
        max_tile = np.max(mat)
        corners = [
            mat[0, 0],
            mat[0, 3],
            mat[3, 0],
            mat[3, 3],
        ]
        return 1.0 if max_tile in corners else 0.0
