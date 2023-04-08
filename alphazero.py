"""
Contains the Alphazero class for training a model to play TicTacToe or ConnectFour
"""
# Importing dependencies
from alphamontecarlo import AlphaMCTS
import numpy as np
from tqdm import trange
from random import shuffle
import torch
import torch.nn.functional as F


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

            temperature_action_probs = action_probs ** (1 / self.args['temperature'])
            temperature_action_probs /= np.sum(temperature_action_probs)

            action = np.random.choice(self.game.action_size, p=temperature_action_probs)

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

    """
    Training method for the model where batches are shuffled
    """
    def train(self, memory):
        shuffle(memory)

        for batchIdx in range(0, len(memory), self.args['batch_size']):
            sample = memory[batchIdx:min(len(memory) - 1, batchIdx + self.args['batch_size'])]
            state, policy_targets, value_targets = zip(*sample)

            state, policy_targets, value_targets = np.array(state), np.array(policy_targets), \
                np.array(value_targets).reshape(-1, 1)

            state = torch.tensor(state, dtype=torch.float32, device=self.model.device)
            policy_targets = torch.tensor(policy_targets, dtype=torch.float32, device=self.model.device)
            value_targets = torch.tensor(value_targets, dtype=torch.float32, device=self.model.device)

            output_policy, output_value = self.model(state)

            policy_loss = F.cross_entropy(output_policy, policy_targets)
            value_loss = F.mse_loss(output_value, value_targets)

            loss = policy_loss + value_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    """
    For each iteration create training data for one cycle, train the model and
    store the state of the model
    """
    def learn(self):
        for iteration in range(self.args['num_iterations']):
            memory = []
            self.model.eval()

            for _ in trange(self.args['num_selfPlay_iterations']):
                memory += self.selfPlay()

            self.model.train()
            for _ in trange(self.args['num_epochs']):
                self.train(memory)

            torch.save(self.model.state_dict(), f"models/{self.game}/model_{iteration}.pt")
            torch.save(self.optimizer.state_dict(), f"models/optimizer_{iteration}.pt")
