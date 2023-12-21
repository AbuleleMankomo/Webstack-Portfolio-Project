# Importing necessary libraries
import random
import math
# Class representing the Tic Tac Toe game state
class TicTacToe:
    def __init__(self):
        """
        Initializes the Tic Tac Toe game state.
        """
        self.board = [" " for _ in range(9)]
        self.current_player = "X"
        self.winner = None

    def make_move(self, position):
        """
        Makes a move in the game at the specified position.
        """
        if self.board[position] == " ":
            self.board[position] = self.current_player
            self.current_player = "X" if self.current_player == "O" else "O"
            self.check_winner()

    def make_move_to_check_and_block_winning_move(self, position):
        """
        Makes a move for checking and potentially blocking the opponent's winning move.
        """
        if self.board[position] == " ":
            self.board[position] = 'X'

    def check_winner(self):
        """
        Checks if there is a winner in the current game state.
        """
        winning_combinations = [(0, 1, 2), (3, 4, 5), (6, 7, 8),
                                (0, 3, 6), (1, 4, 7), (2, 5, 8),
                                (0, 4, 8), (2, 4, 6)]
        for combo in winning_combinations:
            a, b, c = combo
            if self.board[a] == self.board[b] == self.board[c] != " ":
                self.winner = self.board[a]
                return

    def is_draw(self):
        """
        Checks if the game is a draw.
        """
        return " " not in self.board

    def is_terminal(self):
        """
        Checks if the game has reached a terminal state (draw or a player wins).
        """
        return self.is_draw() or self.winner is not None

    def get_legal_moves(self):
        """
        Returns a list of legal moves in the current game state.
        """
        return [i for i in range(9) if self.board[i] == " "]

    def copy(self):
        """
        Creates a copy of the current game state.
        """
        copied_state = TicTacToe()
        copied_state.board = self.board.copy()
        copied_state.current_player = self.current_player
        copied_state.winner = self.winner
        return copied_state
# RAVE MCTS Node Class
class MCTSNodeRAVE:
    def __init__(self, game_state, parent=None, move=None):
        """
        Class representing a node in the MCTS tree with RAVE.
        """
        self.game_state = game_state
        self.parent = parent
        self.move = move
        self.children = []
        self.wins = 0
        self.visits = 0
        self.rave_wins = {m: 0 for m in self.game_state.get_legal_moves()}
        self.rave_visits = {m: 0 for m in self.game_state.get_legal_moves()}

    def is_fully_expanded(self):
        """
        Checks if the node is fully expanded (all legal moves have corresponding children).
        """
        return all(move in [child.move for child in self.children] for move in self.game_state.get_legal_moves())

    def select_child(self):
        """
        Selects a child node based on the UCT formula.
        """
        c = 2.0  # Exploration parameter
        best_child = max(self.children, key=lambda child:
                         (child.wins / child.visits if child.visits > 0 else 0) +
                         c * math.sqrt(2 * math.log(self.visits) / child.visits if child.visits > 0 else 0))
        return best_child
class MCTSNodeRAVE:
    """
    Class representing a node in the MCTS tree with RAVE.
    """
    def __init__(self, game_state, parent=None, move=None):
        """
        Initializes the MCTS node with RAVE.
        """
        self.game_state = game_state
        self.parent = parent
        self.move = move
        self.children = []
        self.wins = 0
        self.visits = 0
        self.rave_wins = {m: 0 for m in self.game_state.get_legal_moves()}
        self.rave_visits = {m: 0 for m in self.game_state.get_legal_moves()}
 def is_fully_expanded(self):
        """
        Checks if the node is fully expanded (all legal moves have corresponding children).
        """
        return all(move in [child.move for child in self.children] for move in self.game_state.get_legal_moves())

    def select_child(self):
        """
        Selects a child node based on the UCT formula.
        """
        c = 2.0  # Exploration parameter
        best_child = max(self.children, key=lambda child:
                         (child.wins / child.visits if child.visits > 0 else 0) +
                         c * math.sqrt(2 * math.log(self.visits) / child.visits if child.visits > 0 else 0))
        return best_child

    def expand(self):
        """
        Expands the tree by adding a child node for an unexplored move.
        """
        legal_moves = self.game_state.get_legal_moves()
        for move in legal_moves:
            if move not in [child.move for child in self.children]:
                new_state = self.game_state.copy()
                new_state.make_move(move)
                new_child = MCTSNodeRAVE(new_state, parent=self, move=move)
                self.children.append(new_child)
                return new_child
 def simulate(self):
        """
        Simulates a game from the current state until a terminal state is reached.
        """
        current_state = self.game_state.copy()
        while not current_state.is_terminal():
            possible_moves = current_state.get_legal_moves()
            move = random.choice(possible_moves)
            current_state.make_move(move)
        return current_state

    def backpropagate(self, result):
        """
        Backpropagates the result of a simulation up the tree.
        """
        node = self
        while node is not None:
            node.visits += 1
            node.wins += result
            if node.move is not None and node.parent:
                node.parent.rave_visits[node.move] += 1
                if result == 1:
                    node.parent.rave_wins[node.move] += 1
            result = -result
            node = node.parent
 def best_move(self):
        """
        Returns the move with the highest number of visits.
        """
        best_move = max(self.children, key=lambda child: child.visits).move
        return best_move
class MCTSRAVE:
    """
    Class representing the MCTS algorithm with RAVE.
    """
    def __init__(self, iterations):
        """
        Initializes the MCTSRAVE algorithm with a specified number of iterations.
        """
        self.iterations = iterations

    def get_best_move(self, current_game_state):
        """
        Gets the best move using the MCTSRAVE algorithm.
        """
        root = MCTSNodeRAVE(current_game_state)
        for _ in range(self.iterations):
            node = root
            while node and not node.game_state.is_terminal() and node.is_fully_expanded():
                node = node.select_child()
            if node and not node.game_state.is_terminal():
                node = node.expand()
            if node:
                simulation_result = node.simulate()
                node.backpropagate(1 if simulation_result.winner == 'X' else -1)
        return root.best_move()
 def select_best_move(self, node):
        """
        Selects the best move based on the MCTSRAVE algorithm.
        """
        legal_moves = node.state.get_legal_moves()
        current_player = node.state.current_player
        opponent = "O" if current_player == "X" else "X"
        
        # Check for winning move
        for move in legal_moves:
            new_state = node.state.copy()
            new_state.make_move(move)
            new_state.check_winner()
            if new_state.winner == current_player:
                position = move
                node.state.make_move(position)
                return position
        
        # Check for blocking opponent's winning move
        for move in legal_moves:
            new_state = node.state.copy()
            new_state.make_move_to_check_and_block_w
