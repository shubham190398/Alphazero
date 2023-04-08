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
from games import TicTacToe, ConnectFour
from parallelalphazero import AlphaZeroParallel
import kaggle_environments
from kaggle import KaggleAgent

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

    print("State is currently", state)

    tensor_state = torch.tensor(tictactoe.get_encoded_state(state), device=device).unsqueeze(0)
    model = ResNet(tictactoe, 4, 64, device=device)
    model.load_state_dict(torch.load('models/TicTacToe/model_colab_2.pt', map_location=device))
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


# Function to train the AlphaZero model for TicTacToe
def alphaTrainTicTacToe():
    game = TicTacToe()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet(game, 4, 64, device)
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

    alphaZero = AlphaZero(model, optimizer, game, args)
    alphaZero.learn()


# Function to train the AlphaZero model for ConnectFour
def alphaTrainConnectFour():
    game = ConnectFour()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet(game, 9, 128, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    args = {
        'C': 2,
        'num_searches': 600,
        'num_iterations': 8,
        'num_selfPlay_iterations': 500,
        'num_epochs': 4,
        'batch_size': 128,
        'temperature': 1.25,
        'dirichlet_epsilon': 0.25,
        'dirichlet_alpha': 0.3
    }

    alphaZero = AlphaZero(model, optimizer, game, args)
    alphaZero.learn()


# Parallel ConnectFour Trainer
def alphaTrainConnectFourParallel():
    game = ConnectFour()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet(game, 9, 128, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    args = {
        'C': 2,
        'num_searches': 600,
        'num_iterations': 8,
        'num_selfPlay_iterations': 600,
        'num_epochs': 4,
        'num_parallel_games': 200,
        'batch_size': 128,
        'temperature': 1.25,
        'dirichlet_epsilon': 0.25,
        'dirichlet_alpha': 0.3
    }

    alphaZero = AlphaZeroParallel(model, optimizer, game, args)
    alphaZero.learn()


# Parallel ConnectFour Trainer
def alphaTrainTicTacToeParallel():
    game = TicTacToe()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet(game, 4, 64, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    args = {
        'C': 2,
        'num_searches': 60,
        'num_iterations': 4,
        'num_selfPlay_iterations': 500,
        'num_parallel_games': 100,
        'num_epochs': 4,
        'batch_size': 64,
        'temperature': 1.25,
        'dirichlet_epsilon': 0.25,
        'dirichlet_alpha': 0.3
    }

    alphaZero = AlphaZeroParallel(model, optimizer, game, args)
    alphaZero.learn()


# Dynamic Visualization of ConnectFour
def ConnectFourVisualizer():
    game = ConnectFour()
    args = {
        'C': 2,
        'search': True,
        'num_searches': 600,
        'dirichlet_epsilon': 0.1,
        'dirichlet_alpha': 0.3,
        'temperature': 0,
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ResNet(game, 9, 128, device)
    model.load_state_dict(torch.load("models/ConnectFour/model_1.pt", map_location=device))
    model.eval()

    env = kaggle_environments.make("connectx")

    player1 = KaggleAgent(model, game, args)
    player2 = KaggleAgent(model, game, args)

    players = [player1.run, player2.run]
    env.run(players)
    out = env.render(mode='html')
    file = open("output.html", "w")
    file.write(out)
    file.close()


# Main function
def main():
    ConnectFourVisualizer()


if __name__ == '__main__':
    main()
