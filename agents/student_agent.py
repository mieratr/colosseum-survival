# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import sys
from copy import deepcopy
import numpy as np
from time import time

start_time = None

# pre_mature = False

moves = ((-1, 0), (0, 1), (1, 0), (0, -1))

opposites = {0: 2, 1: 3, 2: 0, 3: 1}

neg_inf = -float('inf')

pos_inf = float('inf')

time_out_value = 1.5

def generate_valid_moves(chess_board, my_pos, adv_pos, max_step):
    start_pos = my_pos
    state_queue = [(start_pos, 0)]
    visited = {tuple(start_pos)}

    valid_moves = []

    while state_queue:
        cur_pos, cur_step = state_queue.pop(0)

        r, c = cur_pos
        if cur_step == max_step:
            break
        for dir, move in enumerate(moves):
            if chess_board[r, c, dir]:
                continue

            next_pos = np.array(cur_pos) + move


            if np.array_equal(next_pos, adv_pos) or tuple(next_pos) in visited:
                continue
            
            visited.add(tuple(next_pos))
            state_queue.append((next_pos, cur_step + 1))

            valid_moves.append((tuple(cur_pos), dir))

    return valid_moves



def get_score(chess_board, my_pos, adv_pos, max_step):
    start_pos = my_pos
    state_queue = [(start_pos, 0)]
    visited = {tuple(start_pos)}

    # valid_moves = []

    score = 0

    while state_queue:
        cur_pos, cur_step = state_queue.pop(0)

        r, c = cur_pos
        if cur_step == max_step:
            break

        score += max_step - cur_step
        for dir, move in enumerate(moves):
            if chess_board[r, c, dir]:
                continue

            next_pos = np.array(cur_pos) + move



            if np.array_equal(next_pos, adv_pos) or tuple(next_pos) in visited:
                continue
            
            visited.add(tuple(next_pos))
            state_queue.append((next_pos, cur_step + 1))

            # valid_moves.append((tuple(cur_pos), dir))

    return score



def check_endgame(chess_board, my_pos, adv_pos):
    """
    Check if the game ends and compute the current score of the agents.

    Returns
    -------
    is_endgame : bool
        Whether the game ends.
    player_1_score : int
        The score of player 1.
    player_2_score : int
        The score of player 2.
    """
    # Union-Find

    board_size = len(chess_board)
    
    father = dict()
    for r in range(board_size):
        for c in range(board_size):
            father[(r, c)] = (r, c)

    def find(pos):
        if father[pos] != pos:
            father[pos] = find(father[pos])
        return father[pos]

    def union(pos1, pos2):
        father[pos1] = pos2

    for r in range(board_size):
        for c in range(board_size):
            for dir, move in enumerate(
                moves[1:3]
            ):  # Only check down and right
                if chess_board[r, c, dir + 1]:
                    continue
                pos_a = find((r, c))
                pos_b = find((r + move[0], c + move[1]))
                if pos_a != pos_b:
                    union(pos_a, pos_b)

    for r in range(board_size):
        for c in range(board_size):
            find((r, c))

    p0_pos = my_pos
    p1_pos = adv_pos

    p0_r = find(tuple(p0_pos))
    p1_r = find(tuple(p1_pos))
    p0_score = list(father.values()).count(p0_r)
    p1_score = list(father.values()).count(p1_r)
    if p0_r == p1_r:
        return False, p0_score, p1_score
    player_win = None
    win_blocks = -1
    if p0_score > p1_score:
        player_win = 0
        win_blocks = p0_score
    elif p0_score < p1_score:
        player_win = 1
        win_blocks = p1_score
    else:
        player_win = -1  # Tie

    return True, p0_score, p1_score


def get_score_game_over(p0_score, p1_score):

    return 10000 * (p0_score - p1_score)


def valuate_game(chess_board, my_pos, adv_pos, max_step):

    return get_score(chess_board, my_pos, adv_pos, max_step) - get_score(chess_board, adv_pos, my_pos, max_step)


def set_barrier(chess_board, pos, dir):
    r, c = pos
    # Set the barrier to True
    chess_board[r, c, dir] = True
    # Set the opposite barrier to True
    move = moves[dir]
    chess_board[r + move[0], c + move[1], opposites[dir]] = True


def negmax(chess_board, my_pos, adv_pos, max_step, depth, alpha, beta):

    game_over, p0_score, p1_score = check_endgame(chess_board, my_pos, adv_pos)


    if game_over:

        score = get_score_game_over(p0_score, p1_score)

        
        return score, None

    if depth == 0:

        score = valuate_game(chess_board, my_pos, adv_pos, max_step)

        return score, None

    if time() - start_time > time_out_value:
        raise Exception("time out")

    moves = generate_valid_moves(chess_board, my_pos, adv_pos, max_step)

    moves = sorted(moves, key=lambda x: (x[0][0] - adv_pos[0])**2 + (x[0][1] - adv_pos[1])**2)


    value = neg_inf

    best_move = None

    for pos, dir in moves:

        new_board = deepcopy(chess_board)
        set_barrier(new_board, pos, dir)

        subscore, _ = negmax(new_board, adv_pos, pos,  max_step, depth - 1, - beta, - alpha)


        if value < - subscore:

            best_move = (pos, dir)

            value = - subscore

        alpha = max(alpha, value)

        if alpha >= beta:

            break

    # if best_move is None:
    #     print("best_move is none")
    #     print(moves)


    # print("res", value, best_move)

    return value, best_move







@register_agent("student_agent")
class StudentAgent(Agent):
    """
    A dummy class for your implementation. Feel free to use this class to
    add any helper functionalities needed for your agent.
    """

    def __init__(self):
        super(StudentAgent, self).__init__()
        self.name = "StudentAgent"
        self.dir_map = {
            "u": 0,
            "r": 1,
            "d": 2,
            "l": 3,
        }
        self.autoplay = True
        self.first_move = True

    def step(self, chess_board, my_pos, adv_pos, max_step):
        """
        Implement the step function of your agent here.
        You can use the following variables to access the chess board:
        - chess_board: a numpy array of shape (x_max, y_max, 4)
        - my_pos: a tuple of (x, y)
        - adv_pos: a tuple of (x, y)
        - max_step: an integer

        You should return a tuple of ((x, y), dir),
        where (x, y) is the next position of your agent and dir is the direction of the wall
        you want to put on.

        Please check the sample implementation in agents/random_agent.py or agents/human_agent.py for more details.
        """
        # dummy return
        # return my_pos, self.dir_map["u"]

        # moves = generate_valid_moves(chess_board, my_pos, adv_pos, max_step)

        # print("moves", moves)

        # print(chess_board, my_pos, adv_pos, max_step)

        # exit(1)

        global start_time

        global time_out_value

        start_time = time()

        if self.first_move:
            time_out_value = 25
            self.first_move = False

        else:
            time_out_value = 1.5

        # pre_move = None

        res_move = None

        for  max_depth in range(1, 10):

            # max_depth = 2

            try:
                _, move = negmax(chess_board, my_pos, adv_pos, max_step, max_depth, neg_inf, pos_inf)
                # print(max_depth, time() - start_time)
                res_move = move
            except Exception as e:
                # print(e)
                break

        # if res_move is None:
        #     print("invalid")
        #     print(move)

            # print("my choosen move")
            # print(move)

        # timetaken = time() - start_time
        # print(max_depth, timetaken)

        # print(timetaken)

        # if timetaken > 2:
            
        #     return move

        return move

