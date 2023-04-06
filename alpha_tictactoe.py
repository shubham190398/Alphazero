"""
Machine trained to play TicTacToe with Alpha Monte Carlo Tree Search
"""

# Importing dependencies
import numpy as np


# Defining the Tic Tac Toe board
class TicTacToe:
    def __init__(self):
        self.row_count = 3
        self.column_count = 3
        self.action_size = self.row_count * self.column_count

    # The board is initialized with zeros
    def get_initial_state(self):
        return np.zeros((self.row_count, self.column_count))
