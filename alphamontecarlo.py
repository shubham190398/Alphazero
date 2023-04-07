import numpy as np
import torch
import math


# Definition for a node for the AlphaMCTS algorithm
class AlphaNode:
    """
    The node has a parent, a child, visit count, value sum, the game and its state,
    and the action taken. It doesn't need expandable moves any longer as it expands in all directions
    """
    def __init__(self, game, args, state, parent=None, action_taken=None, prior=0):
        self.game = game
        self.args = args
        self.state = state
        self.parent = parent
        self.action_taken = action_taken
        self.prior = prior

        self.children = []

        self.visit_count = 0
        self.value_sum = 0

    # Find if the node is fully expanded or not
    def is_fully_expanded(self):
        return len(self.children) > 0

    # Select a child node
    def select(self):
        best_child = None
        best_ucb = -np.inf

        for child in self.children:
            ucb = self.get_ucb(child)
            if ucb > best_ucb:
                best_child = child
                best_ucb = ucb

        return best_child

    """
    Get the UCB Score given by Q(s,a) + C*P(s,a)*((Sigma*N(s,b))**0.5)/(1 + N(s,a)
    We take 1-q_value because a bad q_value for the child implies a good q_value for the parent
    """
    def get_ucb(self, child):
        if not child.visit_count:
            q_value = 0
        else:
            q_value = 1 - ((child.value_sum / child.visit_count) + 1) / 2
        return q_value + self.args['C'] * (math.sqrt(self.visit_count) / (1 + child.visit_count)) * child.prior

    # Expand the nodes in all directions depending on the policy and its probabilities
    def expand(self, policy):
        for action, prob in enumerate(policy):
            if prob > 0:
                """
                The child will always think he is player 1. Whenever we need to switch players, we will flip
                the board state instead.
                """
                child_state = self.state.copy()
                child_state = self.game.get_next_state(child_state, action, 1)
                child_state = self.game.change_perspective(child_state, -1)

                # Add child Node
                child = AlphaNode(self.game, self.args, child_state, self, action, prob)
                self.children.append(child)

    # Backpropagate the results
    def backpropagate(self, value):
        self.value_sum += value
        self.visit_count += 1

        value = self.game.get_opponent_value(value)

        if self.parent is not None:
            self.parent.backpropagate(value)


# Defining an Alpha MCTS Class
class AlphaMCTS:
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
    def search(self, state):
        root = AlphaNode(self.game, self.args, state)

        for search in range(self.args['num_searches']):
            node = root

            # Selection
            while node.is_fully_expanded():
                node = node.select()

            value, is_terminated = self.game.check_win_and_termination(node.state, node.action_taken)
            value = self.game.get_opponent_value(value)

            # Encode the tensor and do expansion
            if not is_terminated:
                policy, value = self.model(
                    torch.tensor(self.game.get_encoded_state(node.state)).unsqueeze(0)
                )
                policy = torch.softmax(policy, axis=1).squeeze(0).cpu().numpy()

                # Prevent expansion along invalid moves
                valid_moves = self.game.get_valid_moves(node.state)
                policy *= valid_moves
                policy /= np.sum(policy)

                value = value.item()

                # Expansion
                node.expand(policy)

            # Backpropagation
            node.backpropagate(value)

            # Get the probabilities for the different actions
            action_probs = np.zeros(self.game.action_size)
            for child in root.children:
                action_probs[child.action_taken] = child.visit_count
            action_probs /= np.sum(action_probs)
            return action_probs
