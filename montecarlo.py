import numpy as np
import math


# Definition for a node for the MCTS algorithm
class Node:
    """
    The node has a parent, a child, expandable nodes, visit count, value sum, the game and its state,
    and the action taken
    """
    def __init__(self, game, args, state, parent=None, action_taken=None):
        self.game = game
        self.args = args
        self.state = state
        self.parent = parent
        self.action_taken = action_taken

        self.children = []
        self.expandable_moves = game.get_valid_moves(state)

        self.visit_count = 0
        self.value_sum = 0

    # Find if the node is fully expanded or not
    def is_fully_expanded(self):
        return np.sum(self.expandable_moves) == 0 and len(self.children) > 0

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
    Get the UCB Score given by Q(s,a) + C*((ln(N)/n)**(0.5))
    We take 1-q_value because a bad q_value for the child implies a good q_value for the parent
    """
    def get_ucb(self, child):
        q_value = 1 - ((child.value_sum / child.visit_count) + 1) / 2
        return q_value + self.args['C'] * math.sqrt(math.log(self.visit_count) / child.visit_count)

    # Expand the nodes
    def expand(self):
        action = np.random.choice(np.where(self.expandable_moves == 1)[0])
        self.expandable_moves[action] = 0

        """
        The child will always think he is player 1. Whenever we need to switch players, we will flip
        the board state instead.
        """
        child_state = self.state.copy()
        child_state = self.game.get_next_state(child_state, action, 1)
        child_state = self.game.change_perspective(child_state, -1)

        # Add child Node
        child = Node(self.game, self.args, child_state, self, action)
        self.children.append(child)
        return child

    # Simulate the game result
    def simulate(self):
        value, is_terminated = self.game.check_win_and_termination(self.state, self.action_taken)
        value = self.game.get_opponent_value(value)

        if is_terminated:
            return value

        rollout_state = self.state.copy()
        rollout_player = 1

        while True:
            valid_moves = self.game.get_valid_moves(rollout_state)
            action = np.random.choice(np.where(valid_moves == 1)[0])
            rollout_state = self.game.get_next_state(rollout_state, action, rollout_player)
            value, is_terminated = self.game.check_win_and_termination(rollout_state, action)

            if is_terminated:
                if rollout_player == -1:
                    value = self.game.get_opponent_value(value)
                return value

            rollout_player = self.game.get_opponent(rollout_player)

    # Backpropagate the results
    def backpropagate(self, value):
        self.value_sum += value
        self.visit_count += 1

        value = self.game.get_opponent_value(value)

        if self.parent is not None:
            self.parent.backpropagate(value)


# Definition for the Monte Carlo Tree Search
class MCTS:
    def __init__(self, game, args):
        self.game = game
        self.args = args

    """
    The search function should perform the four steps of a Monte Carlo Tree Search:
    1. Selection
    2. Expansion
    3. Simulation
    4. Backpropagation
    """
    def search(self, state):
        root = Node(self.game, self.args, state)

        for search in range(self.args['num_searches']):
            node = root

            # Selection
            while node.is_fully_expanded():
                node = node.select()

            value, is_terminated = self.game.check_win_and_termination(node.state, node.action_taken)
            value = self.game.get_opponent_value(value)

            # Do expansion and simulation if node is not terminal
            if not is_terminated:
                # Expansion
                node = node.expand()

                # Simulation
                value = node.simulate()

            # Backpropagation
            node.backpropagate(value)
