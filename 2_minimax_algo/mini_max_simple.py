class AlphaBetaPruning:
    def __init__(self, depth, game_state, player):
        """
        Initialize the depth, current game state, and player (maximizer or minimizer).
        """
        self.depth = depth
        self.game_state = game_state
        self.player = player  # 'X' for maximizer, 'O' for minimizer

    def is_terminal(self, state):
        """
        Check if the game has reached a terminal state (win, lose, draw).
        """
        # Check rows, columns, and diagonals for a win
        for i in range(3):
            if state[i][0] == state[i][1] == state[i][2] and state[i][0] != ' ':
                return True
            if state[0][i] == state[1][i] == state[2][i] and state[0][i] != ' ':
                return True
        if state[0][0] == state[1][1] == state[2][2] and state[0][0] != ' ':
            return True
        if state[0][2] == state[1][1] == state[2][0] and state[0][2] != ' ':
            return True

        # Check if the board is full (draw)
        for row in state:
            if ' ' in row:
                return False
        return True

    def utility(self, state):
        """
        Return the utility value of the terminal state.
        +1 for a win for 'X', -1 for a win for 'O', 0 for a draw.
        """
        for i in range(3):
            if state[i][0] == state[i][1] == state[i][2]:
                if state[i][0] == 'X':
                    return 1
                elif state[i][0] == 'O':
                    return -1
            if state[0][i] == state[1][i] == state[2][i]:
                if state[0][i] == 'X':
                    return 1
                elif state[0][i] == 'O':
                    return -1
        if state[0][0] == state[1][1] == state[2][2]:
            if state[0][0] == 'X':
                return 1
            elif state[0][0] == 'O':
                return -1
        if state[0][2] == state[1][1] == state[2][0]:
            if state[0][2] == 'X':
                return 1
            elif state[0][2] == 'O':
                return -1
        return 0

    def alphabeta(self, state, depth, alpha, beta, maximizing_player):
        """
        Implement Alpha-Beta pruning.
        """
        if depth == 0 or self.is_terminal(state):
            return self.utility(state)

        if maximizing_player:
            max_eval = float('-inf')
            for move in self.get_possible_moves(state):
                new_state = self.make_move(state, move, 'X')
                eval = self.alphabeta(new_state, depth - 1, alpha, beta, False)
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break  # Beta cutoff
            return max_eval
        else:
            min_eval = float('inf')
            for move in self.get_possible_moves(state):
                new_state = self.make_move(state, move, 'O')
                eval = self.alphabeta(new_state, depth - 1, alpha, beta, True)
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break  # Alpha cutoff
            return min_eval

    def best_move(self, state):
        """
        Determine the best move using Alpha-Beta pruning.
        """
        best_value = float('-inf') if self.player == 'X' else float('inf')
        best_move = None
        for move in self.get_possible_moves(state):
            new_state = self.make_move(state, move, self.player)
            value = self.alphabeta(new_state, self.depth - 1, float('-inf'), float('inf'), self.player == 'O')
            if self.player == 'X' and value > best_value:
                best_value = value
                best_move = move
            elif self.player == 'O' and value < best_value:
                best_value = value
                best_move = move
        return best_move

    def get_possible_moves(self, state):
        """
        Get all possible moves in the current state.
        """
        moves = []
        for i in range(3):
            for j in range(3):
                if state[i][j] == ' ':
                    moves.append((i, j))
        return moves

    def make_move(self, state, move, player):
        """
        Apply a move to the game state and return the new state.
        """
        new_state = [row[:] for row in state]
        new_state[move[0]][move[1]] = player
        return new_state


# Example Usage
if __name__ == "__main__":
    initial_state = [[' ', ' ', ' '], [' ', ' ', ' '], [' ', ' ', ' ']]
    game = AlphaBetaPruning(depth=9, game_state=initial_state, player='X')
    print("Best Move:", game.best_move(initial_state))
