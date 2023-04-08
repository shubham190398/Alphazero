"""
Machine trained to play TicTacToe with Alpha Monte Carlo Tree Search
"""

# Importing dependencies
import numpy as np
from montecarlo import MCTS
import torch
import matplotlib.pyplot as plt
from model import ResNet
from alphamontecarlo import AlphaMCTS
from alphazero import AlphaZero
from games import TicTacToe

# Setting manual seed for consistency
torch.manual_seed(0)


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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    state = tictactoe.get_initial_state()
    state = tictactoe.get_next_state(state, 2, 1)
    state = tictactoe.get_next_state(state, 6, -1)
    state = tictactoe.get_next_state(state, 8, 1)
    state = tictactoe.get_next_state(state, 4, -1)

    print("State is currently", state)

    tensor_state = torch.tensor(tictactoe.get_encoded_state(state), device=device).unsqueeze(0)
    model = ResNet(tictactoe, 4, 64, device=device)
    model.load_state_dict(torch.load('models/TicTacToe/model_3.pt', map_location=device))
    model.eval()

    policy, value = model(tensor_state)
    policy = torch.softmax(policy, axis=1).squeeze(0).detach().cpu().numpy()

    plt.bar(range(tictactoe.action_size), policy)
    plt.show()


# Play Tictactoe with the Alpha MCTS algorithm output
def alpha_mcts_tictactoe():
    tictactoe = TicTacToe()
    player = 1
    args = {
        'C': 2,
        'num_searches': 1000
    }
    model = ResNet(tictactoe, 4, 64)
    model.eval()

    mcts = AlphaMCTS(tictactoe, args, model)
    state = tictactoe.get_initial_state()

    while True:
        print(state)

        if player == 1:
            valid_moves = tictactoe.get_valid_moves(state)
            print("valid moves", [i for i in range(tictactoe.action_size) if valid_moves[i] == 1])
            action = int(input(f"{player}:"))

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


# Function to train the AlphaZero model
def alphaTrain():
    tictactoe = TicTacToe()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet(tictactoe, 4, 64, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    args = {
        'C': 2,
        'num_searches': 60,
        'num_iterations': 4,
        'num_selfPlay_iterations': 500,
        'num_epochs': 4,
        'batch_size': 64,
        'temperature': 1.25,
        'dirichlet_epsilon': 0.25,
        'dirichlet_alpha': 0.3
    }

    alphaZero = AlphaZero(model, optimizer, tictactoe, args)
    alphaZero.learn()


alphaTrain()
