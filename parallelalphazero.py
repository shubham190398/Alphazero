"""
Contains the AlphazeroParallel class for training a model to play TicTacToe or ConnectFour
with games run parallely to speed up training
"""

# Importing dependencies
import numpy as np
from tqdm import trange
from random import shuffle
import torch
import torch.nn.functional as F
from alphamontecarlo import AlphaNode


# Defining an Alpha MCTS Parallel Class
class AlphaMCTSParallel:
    def __init__(self, game, args, model):
        self.game = game
        self.args = args
        self.model = model

    """
    The search function should perform the three steps of Alpha Monte Carlo Tree Search:
    1. Selection
    2. Expansion
    3. Backpropagation
    This is wrapped with no grad to ensure we don't change gradients accidentally
    """
    @torch.no_grad()
    def search(self, states, spGames):
        policy, _ = self.model(
            torch.tensor(self.game.get_encoded_state(states), device=self.model.device)
        )

        policy = torch.softmax(policy, axis=1).cpu().numpy()

        # Create noise for each state with size parameter
        policy = (1 - self.args['dirichlet_epsilon']) * policy + self.args['dirichlet_epsilon'] * \
                 np.random.dirichlet([self.args['dirichlet_alpha']] * self.game.action_size, size=policy.shape[0])

        for i, spg in enumerate(spGames):
            spg_policy = policy[i]
            valid_moves = self.game.get_valid_moves(states[i])
            spg_policy *= valid_moves

            spg_policy /= np.sum(spg_policy)

            spg.root = AlphaNode(self.game, self.args, states[i], visit_count=1)
            spg.root.expand(spg_policy)

        for search in range(self.args['num_searches']):

            for spg in spGames:
                spg.node = None
                node = spg.root

                # Selection
                while node.is_fully_expanded():
                    node = node.select()

                value, is_terminated = self.game.check_win_and_termination(node.state, node.action_taken)
                value = self.game.get_opponent_value(value)

                # Do backpropagation if you reach a terminal node
                if is_terminated:
                    node.backpropagate(value)

                else:
                    spg.node = node

            expandable_spGames = [mappingIdx for mappingIdx in range(len(spGames))
                                  if spGames[mappingIdx].node is not None]

            if len(expandable_spGames) > 0:
                states = np.stack([spGames[mappingIdx].node.state for mappingIdx in expandable_spGames])
                policy, value = self.model(
                    torch.tensor(self.game.get_encoded_state(states), device=self.model.device)
                )
                policy = torch.softmax(policy, axis=1).cpu().numpy()
                value = value.cpu().numpy()

            for i, mappingIdx in enumerate(expandable_spGames):
                # Expand along expandable games
                spg_policy, spg_value = policy[i], value[i]
                node = spGames[mappingIdx].node
                valid_moves = self.game.get_valid_moves(node.state)
                spg_policy *= valid_moves
                spg_policy /= np.sum(spg_policy)

                # Expansion
                node.expand(spg_policy)
                node.backpropagate(spg_value)


# Class Definition of AlphaZero
class AlphaZeroParallel:
    """
    The model is ResNet while the optimizer would be Adam
    """
    def __init__(self, model, optimizer, game, args):
        self.model = model
        self.optimizer = optimizer
        self.game = game
        self.args = args
        self.mcts = AlphaMCTSParallel(game, args, model)

    """
    Model plays against itself
    """
    def selfPlay(self):
        return_memory = []
        player = 1
        spGames = [SPG(self.game) for spg in range(self.args['num_parallel_games'])]

        while len(spGames) > 0:
            states = np.stack([spg.state for spg in spGames])

            neutral_states = self.game.change_perspective(states, player)
            self.mcts.search(neutral_states, spGames)

            # Get the probabilities for the different actions
            action_probs = np.zeros(self.game.action_size)
            for child in root.children:
                action_probs[child.action_taken] = child.visit_count
            action_probs /= np.sum(action_probs)
            return action_probs

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

            for _ in trange(self.args['num_selfPlay_iterations'] // self.args['num_parallel_games']):
                memory += self.selfPlay()

            self.model.train()
            for _ in trange(self.args['num_epochs']):
                self.train(memory)

            torch.save(self.model.state_dict(), f"models/{self.game}/model_{iteration}.pt")
            torch.save(self.optimizer.state_dict(), f"models/optimizer_{iteration}.pt")


# Class Definition for Self Playing Game
class SPG:
    def __init__(self, game):
        self.state = game.get_initial_state()
        self.memory = []
        self.root = None
        self.node = None
