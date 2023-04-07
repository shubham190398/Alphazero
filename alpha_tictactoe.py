"""
Machine trained to play TicTacToe with Alpha Monte Carlo Tree Search
"""

# Importing dependencies
import numpy as np
from montecarlo import MCTS
import torch
import matplotlib.pyplot as plt
from model import ResNet

# Setting manual seed for consistency
torch.manual_seed(0)


# Defining the Tic Tac Toe board
class TicTacToe:
    def __init__(self):
        self.row_count = 3
        self.column_count = 3
        self.action_size = self.row_count * self.column_count

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


# Function to play tictactoe with human input
def play_tictactoe():
    tictactoe = TicTacToe()
    player = 1
    state = tictactoe.get_initial_state()

    while True:
        print(state)
        valid_moves = tictactoe.get_valid_moves(state)
        print("valid moves", [i+1 for i in range(tictactoe.action_size) if valid_moves[i] == 1])
        action = int(input(f"{player}:")) - 1

        if valid_moves[action] == 0:
            print("Invalid move, Try again")
            continue

        state = tictactoe.get_next_state(state, action, player)
        value, is_terminated = tictactoe.check_win_and_termination(state, action)

        if is_terminated:
            print(state)
            if value == 1:
                print(player, "Won")
            else:
                print("Draw")
            break

        player = tictactoe.get_opponent(player)


# Play Tictactoe with the MCTS algorithm
def mcts_tictactoe():
    tictactoe = TicTacToe()
    player = 1
    args = {
        'C': 1.41,
        'num_searches': 1000
    }
    mcts = MCTS(tictactoe, args)
    state = tictactoe.get_initial_state()

    while True:
        print(state)

        if player == 1:
            valid_moves = tictactoe.get_valid_moves(state)
            print("valid moves", [i+1 for i in range(tictactoe.action_size) if valid_moves[i] == 1])
            action = int(input(f"{player}:")) - 1

            if valid_moves[action] == 0:
                print("Invalid move, Try again")
                continue

        # Have the action done by MCTS when it's the second player's turn
        else:
            neutral_state = tictactoe.change_perspective(state, player)
            mcts_probs = mcts.search(neutral_state)
            action = np.argmax(mcts_probs)

        state = tictactoe.get_next_state(state, action, player)
        value, is_terminated = tictactoe.check_win_and_termination(state, action)

        if is_terminated:
            print(state)
            if value == 1:
                print(player, "Won")
            else:
                print("Draw")
            break

        player = tictactoe.get_opponent(player)


# Function to visualize output from the model
def model_visualize():
    tictactoe = TicTacToe()

    state = tictactoe.get_initial_state()
    state = tictactoe.get_next_state(state, 2, 1)
    state = tictactoe.get_next_state(state, 6, -1)

    print("State is currently", state)

    tensor_state = torch.tensor(tictactoe.get_encoded_state(state)).unsqueeze(0)
    model = ResNet(tictactoe, 4, 64)

    policy, value = model(tensor_state)
    policy = torch.softmax(policy, axis=1).squeeze(0).detach().cpu().numpy()

    plt.bar(range(tictactoe.action_size), policy)
    plt.show()


model_visualize()
