# Importing dependencies
import numpy as np


# Defining the TicTacToe Board
class TicTacToe:
    def __init__(self):
        self.row_count = 3
        self.column_count = 3
        self.action_size = self.row_count * self.column_count

    def __repr__(self):
        return "TicTacToe"

    # The board is initialized with zeros
    def get_initial_state(self):
        return np.zeros((self.row_count, self.column_count))

    """
    Get next state. Player action is a number between 0 and 8 so divide by row_count for row position and
    take remainder after dividing by column count for column position
    """
    def get_next_state(self, state, action, player):
        row = action // self.row_count
        column = action % self.column_count
        state[row, column] = player
        return state

    # Check for valid moves left in the board
    def get_valid_moves(self, state):
        return (state.reshape(-1) == 0).astype(np.uint8)

    """
    Check if a player has won. If a player has won, they will occupy either one entire row, one entire column or 
    one entire diagonal. For the top right to bottom left diagonal, we flip the board before using np.diag
    """
    def check_win(self, state, action):
        # Return false if no action is taken
        if action is None:
            return False

        row = action // self.row_count
        column = action % self.column_count
        player = state[row, column]

        return(
            np.sum(state[row, :]) == player * self.column_count
            or np.sum(state[:, column]) == player * self.row_count
            or np.sum(np.diag(state)) == player * self.row_count
            or np.sum(np.diag(np.flip(state, axis=0))) == player * self.column_count
        )

    # Check if the game has terminated. Return True if so. If game is a draw, return 0, else return 1.
    def check_win_and_termination(self, state, action):
        if self.check_win(state, action):
            return 1, True
        elif np.sum(self.get_valid_moves(state)) == 0:
            return 0, True
        else:
            return 0, False

    # Get the next player
    def get_opponent(self, player):
        return -player

    # Get opponent value
    def get_opponent_value(self, value):
        return -value

    # Flip board state
    def change_perspective(self, state, player):
        return state * player

    # Encode the state to send it to the ResNet model
    def get_encoded_state(self, state):
        encoded_state = np.stack(
            (state == -1, state == 0, state == 1)
        ).astype(np.float32)
        return encoded_state


# Defining the Connect Four Board
class ConnectFour:
    def __init__(self):
        self.row_count = 6
        self.column_count = 7
        self.action_size = self.column_count
        self.in_a_row = 4

    def __repr__(self):
        return "ConnectFour"

    # The board is initialized with zeros
    def get_initial_state(self):
        return np.zeros((self.row_count, self.column_count))

    """
    Get next state. Player action the deepest empty field in a column and 
    fill it
    """
    def get_next_state(self, state, action, player):
        row = np.max(np.where(state[:, action] == 0))
        state[row, action] = player
        return state

    # Check for valid moves left in the board
    def get_valid_moves(self, state):
        return (state[0] == 0).astype(np.uint8)

    """
    Check if a player has won. If a player has won, they will have 4 tokens vertically,
    horizontally or diagonally
    """
    def check_win(self, state, action):
        # Return false if no action is taken
        if action is None:
            return False

        row = np.min(np.where(state[:, action] != 0))
        column = action
        player = state[row][column]

        # Define a count function for checking for wins
        def count(offset_row, offset_column):
            for i in range(1, self.in_a_row):

                r = row + offset_row * i
                c = action + offset_column* i
                if (
                    r < 0
                    or r >= self.row_count
                    or c < 0
                    or c >= self.column_count
                    or state[r][c] != player
                ):
                    return i - 1

            return self.in_a_row - 1

        return (
            count(1, 0) >= self.in_a_row - 1
            or (count(0, 1) + count(0, -1)) >= self.in_a_row - 1
            or (count(1, 1) + count(-1, -1)) >= self.in_a_row - 1
            or (count(1, -1) + count(-1, 1)) >= self.in_a_row - 1
        )

    # Check if the game has terminated. Return True if so. If game is a draw, return 0, else return 1.
    def check_win_and_termination(self, state, action):
        if self.check_win(state, action):
            return 1, True
        elif np.sum(self.get_valid_moves(state)) == 0:
            return 0, True
        else:
            return 0, False

    # Get the next player
    def get_opponent(self, player):
        return -player

    # Get opponent value
    def get_opponent_value(self, value):
        return -value

    # Flip board state
    def change_perspective(self, state, player):
        return state * player

    # Encode the state to send it to the ResNet model
    def get_encoded_state(self, state):
        encoded_state = np.stack(
            (state == -1, state == 0, state == 1)
        ).astype(np.float32)
        return encoded_state
