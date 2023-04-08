"""
Contains the Alphazero class for training a model to play TicTacToe or ConnectFour
"""

# Importing dependencies
from alphamontecarlo import AlphaMCTS


# Class Definition of AlphaZero
class AlphaZero:
    """
    The model is ResNet while the optimizer would be Adam
    """
    def __init__(self, model, optimizer, game, args):
        self.model = model
        self.optimizer = optimizer
        self.game = game
        self.args = args
        self.mcts = AlphaMCTS(game, args, model)
