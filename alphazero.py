"""
Contains the Alphazero class for training a model to play TicTacToe or ConnectFour
"""
import torch

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

    def selfPlay(self):
        pass

    def train(self, memory):
        pass

    """
    For each iteration create training data for one cycle, train the model and
    store the state of the model
    """
    def learn(self):
        for iteration in range(self.args['num_iterations']):
            memory = []
            self.model.eval()

            for selfPlay_iteration in range(self.args['num_selfPlay_iterations']):
                memory += self.selfPlay()

            self.model.train()
            for epoch in range(self.args['num_epochs']):
                self.train(memory)

            torch.save(self.model.state_dict(), f"model_{iteration}.pt")
            torch.save(self.optimizer.state_dict(), f"optimizer_{iteration}.pt")
