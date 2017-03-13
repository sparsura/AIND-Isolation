"""This file contains all the classes you must complete for this project.

You can use the test cases in agent_test.py to help during development, and
augment the test suite with your own test cases to further test your code.

You must test your agent's strength against a set of agents with known
relative strength using tournament.py and include the results in your report.
"""
import random

class Timeout(Exception):
    """Subclass base exception for code clarity."""
    pass

def weighted_score(game, player):
    """The cells on the board can be classified based on the number of moves that would be possible from that cell (given a blank board).
    For instance from each of the 4 corners, only 2 moves are possible.
    From the center cell of the board (assuming at least a 5x5 board), 8 moves are possible.

    Categorize the legal move cells into eight-move, six-move, four-move, three-move and two-move cells.
    Calculate a weighted score based on this break down.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
        The weighted score as defined above.
    """

    gwidth = game.width
    gheight = game.height
    eightmoves = sixmoves = fourmoves = threemoves = twomoves = 0
    for row, col in game.get_legal_moves(player):
        # Cells that are in the middle of the board i.e. inside a border that is 2 cells wide/high
        if 1 < row < gheight - 2 and 1 < col < gwidth - 2:
             eightmoves = eightmoves + 1
        # Cells that are in the middle of the board and on the border just inside of the outermost border
        elif (row == 1 or row == gheight - 2) and 1 < col < gwidth - 2:
             sixmoves = sixmoves + 1
        elif (col == 1 or col == gwidth - 2) and 1 < row < gheight - 2:
             sixmoves = sixmoves + 1
        # Cells that are in the middle of the board and on the outermost border
        elif (row == 0 or row == gheight - 1) and 1 < col < gwidth - 2:
             fourmoves = fourmoves + 1
        elif (col == 0 or col == gwidth - 1) and 1 < row < gheight - 2:
             fourmoves = fourmoves + 1
        # Cells that are in the same row/column as the corners and adjacent to them
        elif (row, col) in [(0, 1), (0, gwidth - 2), (1, 0), (1, gwidth - 1)]:
             threemoves = threemoves + 1
        elif (row, col) in [(gheight - 2, 0), (gheight - 2, gwidth - 1), (gheight - 1, 1), (gheight - 1, gwidth - 2)]:
             threemoves = threemoves + 1
        # Corner cells
        elif (row, col) in [(0, 0), (0, gwidth - 1), (gheight - 1, 0), (gheight - 1, gwidth - 1)]:
             twomoves = twomoves + 1
    wscore = 16 * eightmoves + 8 * sixmoves + 4 * fourmoves + 2 * threemoves + 1 * twomoves
    return(wscore)

def custom1_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """

    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    # player's weighted score - opponent's weighted score
    own_move_score = weighted_score(game, player)
    opp_move_score = weighted_score(game, game.get_opponent(player))
    return float(own_move_score - opp_move_score)

def custom2_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """

    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    # 2 * player's weighted score - opponent's weighted score
    own_move_score = weighted_score(game, player)
    opp_move_score = weighted_score(game, game.get_opponent(player))
    return float(2 * own_move_score - opp_move_score)

def custom3_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """

    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    # player's weighted score - 2 * opponent's weighted score
    own_move_score = weighted_score(game, player)
    opp_move_score = weighted_score(game, game.get_opponent(player))
    return float(own_move_score - 2 * opp_move_score)

custom_score = custom3_score

class CustomPlayer:
    """Game-playing agent that chooses a move using your evaluation function
    and a depth-limited minimax algorithm with alpha-beta pruning. You must
    finish and test this player to make sure it properly uses minimax and
    alpha-beta to return a good move before the search time limit expires.

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    iterative : boolean (optional)
        Flag indicating whether to perform fixed-depth search (False) or
        iterative deepening search (True).

    method : {'minimax', 'alphabeta'} (optional)
        The name of the search method to use in get_move().

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """

    def __init__(self, search_depth=3, score_fn=custom_score,
                 iterative=True, method='minimax', timeout=10.):
        self.search_depth = search_depth
        self.iterative = iterative
        self.score = score_fn
        self.method = method
        self.time_left = None
        self.TIMER_THRESHOLD = timeout

    def get_move(self, game, legal_moves, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        This function must perform iterative deepening if self.iterative=True,
        and it must use the search method (minimax or alphabeta) corresponding
        to the self.method value.

        **********************************************************************
        NOTE: If time_left < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        legal_moves : list<(int, int)>
            A list containing legal moves. Moves are encoded as tuples of pairs
            of ints defining the next (row, col) for the agent to occupy.

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """

        self.time_left = time_left


        # Perform any required initializations, including selecting an initial
        # move from the game board (i.e., an opening book), or returning
        # immediately if there are no legal moves

        if not legal_moves:
            return((-1, -1))
        else:
            move = legal_moves[random.randrange(len(legal_moves))]

        try:
            # The search method call (alpha beta or minimax) should happen in
            # here in order to avoid timeout. The try/except block will
            # automatically catch the exception raised by the search method
            # when the timer gets close to expiring

            # If iterative deepening is enabled, repeatedly invoke minimax/alphabeta with increasing depths
            if self.iterative:
                depth = 1
                if self.method == "minimax":
                    while True:
                        score, move = self.minimax(game, depth)
                        depth = depth + 1
                elif self.method == "alphabeta":
                    while True:
                        score, move = self.alphabeta(game, depth)
                        depth = depth + 1
            # If iterative deepening is not enabled, invoke minimax/alphabeta for fixed depth
            else:
                if self.method == "minimax":
                    score, move = self.minimax(game, self.search_depth)
                elif self.method == "alphabeta":
                    score, move = self.minimax(game, self.search_depth)

        except Timeout:
            # Handle any actions required at timeout, if necessary
            pass

        # Return the best move from the last completed search iteration
        return(move)

    def minimax(self, game, depth, maximizing_player=True):
        """Implement the minimax search algorithm as described in the lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        # If its a leaf node, return the player's current score & position
        if depth == 0:
            score = self.score(game, self)
            move = game.get_player_location(self)
            return(score, move)
        # If its a maximizing layer, recursively call the minimizing layer for each legal move.
        # Return the score & move corresponding to the maximum of those minimum scores.
        if maximizing_player:
            max_score = float("-inf")
            max_move = (-1, -1)
            for move in game.get_legal_moves():
                new_game = game.forecast_move(move)
                new_score, new_move = self.minimax(new_game, depth - 1, False)
                if new_score > max_score:
                    max_score = new_score
                    max_move = move
            return(max_score, max_move)
        # If its a minimizing layer, recursively call the maximizing layer for each legal move.
        # Return the score & move corresponding to the minimum of those maximum scores.
        else:
            min_score = float("inf")
            min_move = (-1, -1)
            for move in game.get_legal_moves():
                new_game = game.forecast_move(move)
                new_score, new_move = self.minimax(new_game, depth - 1, True)
                if new_score < min_score:
                    min_score = new_score
                    min_move = move
            return(min_score, min_move)

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf"), maximizing_player=True):
        """Implement minimax search with alpha-beta pruning as described in the
        lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        # If its a leaf node, return the player's current score & position
        if depth == 0:
            score = self.score(game, self)
            move = game.get_player_location(self)
            return(score, move)
        # If its a maximizing layer, recursively call the minimizing layer for each legal move, pruning as needed.
        # Return the score & move corresponding to the maximum of those minimum scores.
        if maximizing_player:
            max_score = float("-inf")
            max_move = (-1, -1)
            for move in game.get_legal_moves():
                new_game = game.forecast_move(move)
                new_score, new_move = self.alphabeta(new_game, depth - 1, alpha, beta, False)
                if new_score > max_score:
                    max_score = new_score
                    max_move = move
                if max_score >= beta:
                    return(max_score, max_move)
                if max_score > alpha:
                    alpha = max_score
            return(max_score, max_move)
        # If its a minimizing layer, recursively call the maximizing layer for each legal move, pruning as needed.
        # Return the score & move corresponding to the minimum of those maximum scores.
        else:
            min_score = float("inf")
            min_move = (-1, -1)
            for move in game.get_legal_moves():
                new_game = game.forecast_move(move)
                new_score, new_move = self.alphabeta(new_game, depth - 1, alpha, beta, True)
                if new_score < min_score:
                    min_score = new_score
                    min_move = move
                if min_score <= alpha:
                    return(min_score, min_move)
                if min_score < beta:
                    beta = min_score
            return(min_score, min_move)
