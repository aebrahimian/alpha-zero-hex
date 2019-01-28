import Arena
from MCTS import MCTS
from hex.HexGame import HexGame, display
from hex.HexPlayers import *
from hex.pytorch.NNet import NNetWrapper as NNet

import numpy as np
from utils import *

"""
use this script to play any two agents against each other, or play manually with
any agent.
"""

g = HexGame(6)

# all players
rp = RandomPlayer(g).play
hp = HumanHexPlayer(g).play

# nnet players
n1 = NNet(g)
n1.load_checkpoint('./pretrained_models/hex/pytorch/dev/6x100/','best.pth.tar')
args1 = dotdict({'numMCTSSims': 50, 'cpuct':1.0})
mcts1 = MCTS(g, n1, args1)
n1p = lambda x, player: np.argmax(mcts1.getActionProb(x, player, temp=0))

# n2 = NNet(g)
# n2.load_checkpoint('./pretrained_models/hex/pytorch/','old.pth.tar')
# args2 = dotdict({'numMCTSSims': 50, 'cpuct':1.0})
# mcts2 = MCTS(g, n2, args2)
# n2p = lambda x, player: np.argmax(mcts2.getActionProb(x, player, temp=0))

arena = Arena.Arena(n1p, rp, g, display=display)
num = 10
print(arena.playGames(num, verbose=True))
# print('sim count all', mcts1.sim_count, 'avg game', mcts1.sim_count/num, 'avg turn', mcts1.sim_count/(num*32))