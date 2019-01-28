import numpy as np

"""
the play method of Player give original board ,player and return action in canonical board
this is correct for play from mcts too
"""

class RandomPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board, player):
        canonicalBoard = self.game.getCanonicalForm(board, player)
        a = np.random.randint(self.game.getActionSize())
        valids = self.game.getValidMoves(canonicalBoard, 1)
        while valids[a]!=1:
            a = np.random.randint(self.game.getActionSize())
        return a


class HumanHexPlayer():
    def __init__(self, game):
        self.game = game

    def getCanonicalPosition(self, pos, player):
        if player == 1:
            return pos

        n = self.game.n
        x, y = pos
        board = np.zeros(shape=(n, n))
        board[x][y] = 1
        board = self.game.getCanonicalForm(board, -1)
        loc = np.where(board == -1)
        x, y = loc[0][0], loc[1][0]
        return (x, y)

    def play(self, board, player):
        # display(board)
        canonicalBoard = self.game.getCanonicalForm(board, player)
        valid = self.game.getValidMoves(canonicalBoard, 1)
        # for i in range(len(valid)):
        #     if not valid[i]:
        #         print(int(i/self.game.n), int(i%self.game.n))        
        while True:
            a = input()

            x, y = [int(x) for x in a.split(' ')]
            x, y = self.getCanonicalPosition((x, y), player)
            a = self.game.n * x + y
            if valid[a]:
                break
            else:
                print('Invalid')

        return a


class GreedyHexPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board):
        pass
