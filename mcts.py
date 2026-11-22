import math
import random

MAX_SIM_MOVES = 80 # so nuoc 1 lan mo phong
HEURISTIC_RADIUS = 2 # heuristic xet cac o nam gan 2 nuoc di gan nhat
DIRECTIONS = ((1,0), (0,1), (1,1), (1,-1))
LAST_STATS = {} # thong ke lan goi get_best_move_mcts gan nhat (cho tv4 thu thap du lieu)
class MCTSNode:
    def __init__(self, state, parent=None, move=None, player=1):
        self.state = state # Board tai nut nay
        self.parent = parent # nut cha
        self.move = move # nuoc di (row,col) dan toi nut nay
        self.player = player # nguoi vua di nuoc nay (1 hoac -1)
        self.children = [] # danh sach nut con
        self.N = 0 # so luot ghe tham
        self.W = 0.0 # tong diem thang

        #cac nuoc chua mo rong, nut ket thuc thi ko con nuoc
        self.untried_moves = [] if state.is_terminal() else state.get_candidate_moves()
        random.shuffle(self.untried_moves)
        self.ordered = False # da sap xep nuoc di theo Heuristic chua

    # 
    def is_fully_expanded(self):
        return len(self.untried_moves) == 0

# tinh toan uct
def uct_value(node, c):
    if node.N == 0:
        return float("inf") # nut chua tham co uct = vo cuc
    return (node.W / node.N) + (c * math.sqrt(math.log(node.parent.N) / node.N))

# chon con co UCT lon nhat den khi gap nut con chua thu
def select(node, c):
    while node.is_fully_expanded() and node.children:
        node = max(node.children, key=lambda ch: uct_value(ch, c))
    return node

def expand(node, mode="Random"):
    if node.state.is_terminal() or not node.untried_moves:
        return node
    mover = node.state.current_player
    if mode == "Heuristic" and not node.ordered:
        st = node.state
        node.untried_moves.sort(
            key=lambda m: _score_cell(st.grid, st.size, m[0], m[1], mover, st.win_length)
        )
        node.ordered = True
    row, col = node.untried_moves.pop()
    new_state = node.state.clone()
    new_state.make_move(row, col, mover)
    child = MCTSNode(new_state, parent=node, move=(row, col), player=mover)
    node.children.append(child)
    return child

def _line_info(grid, size, r, c, player):
    best_len, best_open = 1, 0
    for dr, dc in DIRECTIONS:
        length, open_ends = 1, 0
        for sign in (1, -1):
            nr = r + sign * dr
            nc = c + sign * dc
            while 0 <= nr < size and 0 <= nc < size and grid[nr][nc] == player:
                nr += sign * dr
                nc += sign * dc
                length += 1
            if 0 <= nr < size and 0 <= nc < size and grid[nr][nc] == 0:
                open_ends += 1
        if length > best_len or (length == best_len and open_ends > best_open):
            best_len = length
            best_open = open_ends
    return best_len, best_open

def _score_cell(grid, size, r, c, me, win_len):
    """Diem uu tien cua o (r, c): thang > chan thua > chuoi 4 > chan 4 > chuoi 3 > chan 3."""
    my_len, my_open = _line_info(grid, size, r, c, me)
    op_len, op_open = _line_info(grid, size, r, c, -me)

    #thang
    if my_len >= win_len:
        return 100000

    #chan thua
    if op_len >= win_len:
        return 50000

    score = 0
    #chuoi 4
    if my_len == win_len - 1:
        score += 5000 if my_open else 500

    #chan 4
    if op_len == win_len - 1:
        score += 4000 if op_open else 400

    #chuoi 3
    if my_len == win_len - 2:
        score += 200 if my_open == 2 else 40

    #chan 3
    if op_len == win_len - 2:
        score += 150 if op_open == 2 else 30

    return score


def heuristic_move(state, candidates, player):
    recent = [(r,c) for r, c, _ in state.move_history[-2:]]
    near = [m for m in candidates
            if any(abs(m[0]-r) <= HEURISTIC_RADIUS and abs(m[1]-c) <= HEURISTIC_RADIUS for r, c in recent)]
    pool = near if near else candidates
    best_score = -1
    best_moves = []
    for r, c in pool:
        s = _score_cell(state.grid, state.size, r, c, player, state.win_length)
        if s > best_score:
            best_score = s
            best_moves = [(r,c)]
        elif s == best_score:
            best_moves.append((r,c))
    if best_score <= 0:
        return random.choice(candidates)
    return random.choice(best_moves)

def simulate(node, mode = "Random"):
    state = node.state.clone()
    if state.winner != 0:
        return state.winner

    candidates = set(state.get_candidate_moves())
    size = state.size
    player = state.current_player

    for _ in range(MAX_SIM_MOVES):
        if not candidates:
            return 0
        cand_list = list(candidates)
        if mode ==  "Heuristic":
            move = heuristic_move(state, cand_list, player)
        else:
            move = random.choice(cand_list)
        state.make_move(move[0], move[1], player)
        if state.winner != 0:
            return state.winner

        candidates.discard(move)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr = move[0] + dr
                nc = move[1] + dc
                if 0 <= nr < size and 0 <= nc < size and state.grid[nr][nc] == 0:
                    candidates.add((nr,nc))
        player = -player
    return 0 #het gioi han nuoc di -> hoa


def backpropagate(node, result):
    """Cap nhat N, W tu nut la nguoc ve goc. W tinh theo goc nhin cua nguoi da tao ra nut do."""
    while node is not None:
        node.N += 1
        if result == 0:
            node.W += 0.5
        elif result == node.player:
            node.W += 1.0
        node = node.parent

def get_best_move_mcts(board, iterations=1000, c=1.4, mode="Random", player=None):
    """Nhan ban co hien tai, tra ve (nuoc di toi uu, so nut da duyet).

    iterations : so luot lap MCTS (100 -> 5000 khi lam thuc nghiem)
    c          : he so kham pha cua UCT (0.5, 1.0, 1.4, 2.0)
    mode       : "Random" hoac "Heuristic" (cho buoc simulate)
    player     : ben se di (mac dinh = board.current_player)
    """
    if board.is_terminal():
        return None, 0

    state = board.clone()
    if player is not None:
        state.current_player = player

    to_move = state.current_player

    root = MCTSNode(state, player = -to_move)
    node_count = 1

    #Heuristic : di ngay neu co nuoc de win hoac co chuoi 4 cua dich
    if mode == "Heuristic":
        scored = [(_score_cell(state.grid, state.size, r, c, to_move, state.win_length), (r,c))
                  for r, c in root.untried_moves]
        top = max(scored)[0] if scored else 0
        if top >= 50000:
            move = random.choice([m for sc, m in scored if sc == top])
            LAST_STATS.clear()
            LAST_STATS.update({"iterations": 0, "c": c, "mode": mode, "nodes": 1, "root_N": 0,
                               "best_N": 0, "best_W": 0.0, "best_winrate": 1.0 if top >= 100000 else 0.0})
            return move, node_count

    for _ in range(iterations):
        leaf = select(root, c)
        leaf = expand(leaf, mode)
        if leaf.parent is not None and leaf.N == 0:
            node_count += 1
        result = simulate(leaf, mode)
        backpropagate(leaf, result)

    best = max(root.children, key=lambda ch: ch.N) #chon nut dc tham nhieu nhat

    LAST_STATS.clear()
    LAST_STATS.update({
        "iterations": iterations,
        "c": c,
        "mode": mode,
        "nodes": node_count,
        "root_N": root.N,
        "best_N": best.N,
        "best_W": best.W,
        "best_winrate": best.W / best.N if best.N else 0.0,
    })
    return best.move, node_count

if __name__ == "__main__":
    from board import Board
    b = Board()
    b.make_move(10,10,1)
    mv, nodes = get_best_move_mcts(b, iterations=500, c=1.4, mode="Heuristic")
    print("Nuoc di:", mv, "| So nut:", nodes, "|", LAST_STATS)