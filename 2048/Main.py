from datetime import datetime
from GameBoard import GameBoard
from Agent import Agent
from MiniMaxAgent import MinimaxAgent
from ExpectiMaxAgent import ExpectimaxAgent

def check_win(board: GameBoard):
    return board.get_max_tile() >= 2048


int_to_string = ['UP', 'DOWN', 'LEFT', 'RIGHT']

if __name__ == '__main__':
    agent: Agent
    board: GameBoard
    agent = ExpectimaxAgent(
        depth=4,
        empty_weight=30000.0,
        smooth_weight=1.0,
        max_tile_weight=0.5,
        mono_weight=5.0,
        corner_weight=50.0,
        value_weight=0.0001,
        p_two=0.9
    )
    board = GameBoard()
    done = False
    moves = 0
    board.render()
    start = datetime.now()
    while not done:
        action = agent.play(board)
        print('Next Action: "{}"'.format(
            int_to_string[action]), ',   Move: {}'.format(moves))
        done = board.play(action)
        done = done or check_win(board)
        board.render()
        moves += 1

    print('\nTotal time: {}'.format(datetime.now() - start))
    print('\nTotal Moves: {}'.format(moves))
    if check_win(board):
        print("WON THE GAME!!!!!!!!")
    else:
        print("BOOOOOOOOOO!!!!!!!!!")

    max_tile = board.get_max_tile()
    print("Max tile alcanzado:", max_tile)