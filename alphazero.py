"""
Contains the Alphazero class for training a model to play TicTacToe or ConnectFour
"""
import torch

# Importing dependencies
from alphamontecarlo import AlphaMCTS
import numpy as np


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

    """
    Model plays against itself
    """
    def selfPlay(self):
        memory = []
        player = 1
        state = self.game.get_initial_state()

        while True:
            neutral_state = self.game.change_perspective(state, player)
            action_probs = self.mcts.search(neutral_state)

            memory.append((neutral_state, action_probs, player))

            action = np.random.choice(self.game.action_size, p=action_probs)

            state = self.game.get_next_state(state, action, player)

            value, is_terminated = self.game.check_win_and_termination(state, action)

            if is_terminated:
                returnMemory = []

                for hist_neutral_state, hist_action_probs, hist_player in memory:
                    hist_outcome = value if hist_player == player else self.game.get_opponent_value(value)
                    returnMemory.append(
                        (
                            self.game.get_encoded_state(hist_neutral_state),
                            hist_action_probs,
                            hist_outcome
                        )
                    )

                return returnMemory

            player = self.game.get_opponent(player)

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
