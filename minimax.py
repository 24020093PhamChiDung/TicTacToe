"""Thuat toan Minimax & Alpha-Beta cho Tic-Tac-Toe / Gomoku 20x20.

Thanh vien 2 phu trach file nay. Cac ham chinh:
    - evaluate_board(board, player)          : ham heuristic danh gia the co
    - minimax(...)                           : Minimax thuan, dem node
    - alpha_beta(...)                        : Minimax + cat tia Alpha-Beta, dem node
    - get_best_move_minimax(...)             : tra ve (nuoc di toi uu, so node da duyet)

Quy uoc diem: diem > 0 co loi cho `player` (quan dang duoc danh gia), < 0 bat loi.
`node_counter` la list 1 phan tu, vd [0]; moi node ghe tham se tang node_counter[0] them 1.
"""

from typing import List, Optional, Tuple

from board import Board, EMPTY, X, O, DIRECTIONS

# --------------------------------------------------------------------------- #
# HANG SO
# --------------------------------------------------------------------------- #
INF = float("inf")
WIN_SCORE = 10 ** 9          # "vo cung" cho chuoi 5 (huu han de cong/tru depth duoc)

# Bang diem cho chuoi: SCORES[do_dai] = (diem khi ho 1 dau, diem khi ho 2 dau)
# Chuoi bi chan ca 2 dau => 0 diem (khong the thanh 5).
SCORES = {
    4: (1_000, 10_000),
    3: (100, 1_000),
    2: (1, 10),          # khong co trong de bai, them de AI biet phat trien the co
}

# So nuoc di toi da xet o moi node (sau khi sap xep). Giu nhanh voi ban 20x20.
DEFAULT_MAX_BRANCH = 12


# --------------------------------------------------------------------------- #
# 1. HAM LUONG GIA (HEURISTIC)
# --------------------------------------------------------------------------- #
def evaluate_board(board: Board, player: int = X) -> int:
    """Danh gia the co tu goc nhin cua `player`.

    Quet moi chuoi quan lien tiep theo 4 huong (ngang, doc, 2 cheo).
    Moi chuoi chi duoc tinh 1 lan (tai quan dau chuoi). Diem cua `player`
    tru diem cua doi thu.
    """
    grid = board.grid
    size = board.size
    win_len = board.win_length
    totals = {X: 0, O: 0}

    for r in range(size):
        row = grid[r]
        for c in range(size):
            p = row[c]
            if p == EMPTY:
                continue
            for dr, dc in DIRECTIONS:
                pr, pc = r - dr, c - dc
                # Neu o truoc cung la quan cua p => chuoi nay da duoc tinh roi
                if 0 <= pr < size and 0 <= pc < size and grid[pr][pc] == p:
                    continue

                # Dem do dai chuoi
                length = 1
                nr, nc = r + dr, c + dc
                while 0 <= nr < size and 0 <= nc < size and grid[nr][nc] == p:
                    length += 1
                    nr += dr
                    nc += dc

                if length >= win_len:                 # Chuoi 5: +vo cung
                    totals[p] += WIN_SCORE
                    continue
                if length < 2 or length not in SCORES:
                    continue

                # Kiem tra 2 dau co ho (o trong) khong
                open_ends = 0
                if 0 <= pr < size and 0 <= pc < size and grid[pr][pc] == EMPTY:
                    open_ends += 1
                if 0 <= nr < size and 0 <= nc < size and grid[nr][nc] == EMPTY:
                    open_ends += 1
                if open_ends:
                    totals[p] += SCORES[length][open_ends - 1]

    return totals[player] - totals[-player]


# --------------------------------------------------------------------------- #
# 2. SAP XEP NUOC DI (giup Alpha-Beta cat tia nhieu hon + gioi han nhanh)
# --------------------------------------------------------------------------- #
_RUN_VALUE = {1: 1, 2: 10, 3: 100, 4: 1_000}


def _local_score(board: Board, row: int, col: int, player: int) -> int:
    """Diem nhanh cho viec dat `player` vao (row, col): noi chuoi cua minh + chan doi thu."""
    grid = board.grid
    size = board.size
    total = 0
    for dr, dc in DIRECTIONS:
        for who, weight in ((player, 10), (-player, 9)):
            n = 1
            for sign in (1, -1):
                r, c = row + sign * dr, col + sign * dc
                while 0 <= r < size and 0 <= c < size and grid[r][c] == who:
                    n += 1
                    r += sign * dr
                    c += sign * dc
            total += weight * (100_000 if n >= 5 else _RUN_VALUE.get(n, 0))
    return total


def _ordered_moves(board: Board, player: int, max_branch: int) -> List[Tuple[int, int]]:
    """Lay cac nuoc di ung vien (ban kinh 1), sap xep tot -> xau, cat con max_branch."""
    moves = board.get_candidate_moves(radius=1)
    if len(moves) <= 1:
        return moves
    mid = board.size // 2
    moves.sort(
        key=lambda m: (-_local_score(board, m[0], m[1], player),
                       abs(m[0] - mid) + abs(m[1] - mid))
    )
    return moves[:max_branch]


# --------------------------------------------------------------------------- #
# 3. MINIMAX THUAN
# --------------------------------------------------------------------------- #
def minimax(board: Board, depth: int, is_maximizing: bool, node_counter: List[int],
            ai_player: int = X, max_branch: int = DEFAULT_MAX_BRANCH) -> float:
    """Minimax duyet cay den do sau `depth`. Tang node_counter[0] moi node ghe tham.

    ai_player    : ben dang MAX (diem duoc tinh theo goc nhin cua ben nay).
    is_maximizing: True neu den luot ai_player, False neu den luot doi thu.
    """
    node_counter[0] += 1

    if board.winner != EMPTY:
        # +depth de uu tien thang som / thua muon
        return (WIN_SCORE + depth) if board.winner == ai_player else -(WIN_SCORE + depth)
    if board.is_full():
        return 0
    if depth == 0:
        return evaluate_board(board, ai_player)

    mover = ai_player if is_maximizing else -ai_player
    moves = _ordered_moves(board, mover, max_branch)

    if is_maximizing:
        best = -INF
        for r, c in moves:
            board.make_move(r, c, mover)
            best = max(best, minimax(board, depth - 1, False, node_counter, ai_player, max_branch))
            board.undo_move()
        return best

    best = INF
    for r, c in moves:
        board.make_move(r, c, mover)
        best = min(best, minimax(board, depth - 1, True, node_counter, ai_player, max_branch))
        board.undo_move()
    return best


# --------------------------------------------------------------------------- #
# 4. ALPHA-BETA
# --------------------------------------------------------------------------- #
def alpha_beta(board: Board, depth: int, alpha: float, beta: float,
               is_maximizing: bool, node_counter: List[int],
               ai_player: int = X, max_branch: int = DEFAULT_MAX_BRANCH) -> float:
    """Minimax co cat tia Alpha-Beta. Cho ket qua giong minimax nhung it node hon."""
    node_counter[0] += 1

    if board.winner != EMPTY:
        return (WIN_SCORE + depth) if board.winner == ai_player else -(WIN_SCORE + depth)
    if board.is_full():
        return 0
    if depth == 0:
        return evaluate_board(board, ai_player)

    mover = ai_player if is_maximizing else -ai_player
    moves = _ordered_moves(board, mover, max_branch)

    if is_maximizing:
        value = -INF
        for r, c in moves:
            board.make_move(r, c, mover)
            value = max(value, alpha_beta(board, depth - 1, alpha, beta, False,
                                          node_counter, ai_player, max_branch))
            board.undo_move()
            alpha = max(alpha, value)
            if alpha >= beta:          # cat tia beta
                break
        return value

    value = INF
    for r, c in moves:
        board.make_move(r, c, mover)
        value = min(value, alpha_beta(board, depth - 1, alpha, beta, True,
                                      node_counter, ai_player, max_branch))
        board.undo_move()
        beta = min(beta, value)
        if alpha >= beta:              # cat tia alpha
            break
    return value


# --------------------------------------------------------------------------- #
# 5. HAM GOI CHINH: TIM NUOC DI TOT NHAT
# --------------------------------------------------------------------------- #
def get_best_move_minimax(board: Board, depth: int = 2, player: Optional[int] = None,
                          use_alpha_beta: bool = True,
                          max_branch: int = DEFAULT_MAX_BRANCH
                          ) -> Tuple[Optional[Tuple[int, int]], int]:
    """Nhan ban co hien tai, tra ve (nuoc di toi uu (row, col), so node da duyet).

    player         : quan AI dang choi (X/O). Mac dinh la board.current_player.
    use_alpha_beta : True -> Alpha-Beta, False -> Minimax thuan.
    Ham khong lam thay doi ban co goc (lam viec tren ban sao).
    Neu khong con nuoc di -> (None, so node).
    """
    if player is None:
        player = board.current_player

    work = board.clone()
    counter = [1]                      # tinh node goc
    if work.is_terminal():
        return None, counter[0]

    moves = _ordered_moves(work, player, max_branch)
    if not moves:
        return None, counter[0]

    best_move = moves[0]
    best_value = -INF
    alpha, beta = -INF, INF

    for r, c in moves:
        work.make_move(r, c, player)
        if use_alpha_beta:
            value = alpha_beta(work, depth - 1, alpha, beta, False, counter, player, max_branch)
        else:
            value = minimax(work, depth - 1, False, counter, player, max_branch)
        work.undo_move()

        if value > best_value:         # strict '>' : giu nuoc co thu tu uu tien cao hon khi bang diem
            best_value = value
            best_move = (r, c)
        if use_alpha_beta:
            alpha = max(alpha, best_value)

    return best_move, counter[0]


# --------------------------------------------------------------------------- #
# TEST NHANH
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import time

    # Test 1: ban co rong -> di giua
    b = Board()
    print("Ban rong:", get_best_move_minimax(b, depth=2))

    # Test 2: X co 4 quan lien tiep, den luot X -> phai thang
    b = Board()
    for i in range(4):
        b.make_move(5, 5 + i, X)
        b.make_move(8, 5 + i, O)
    print("X thang:", get_best_move_minimax(b, depth=2, player=X)[0], "(ky vong (5,4) hoac (5,9))")

    # Test 3: O co 4 quan, den luot X -> phai chan
    b = Board()
    b.make_move(10, 10, X)
    for i in range(4):
        b.make_move(3, 3 + i, O)
        if i < 3:
            b.make_move(12 + i, 12, X)
    b.current_player = X
    print("X chan O:", get_best_move_minimax(b, depth=2, player=X)[0], "(ky vong (3,2) hoac (3,7))")

    # Test 4: so sanh node Minimax vs Alpha-Beta tren the co giua van
    b = Board()
    seq = [(10, 10), (10, 11), (11, 11), (9, 10), (12, 12), (13, 13), (11, 10), (11, 9)]
    for r, c in seq:
        b.make_move(r, c)
    for d in (1, 2, 3):
        t0 = time.time()
        m1, n1 = get_best_move_minimax(b, depth=d, use_alpha_beta=False)
        t1 = time.time()
        m2, n2 = get_best_move_minimax(b, depth=d, use_alpha_beta=True)
        t2 = time.time()
        print(f"depth={d}: Minimax {m1} nodes={n1:,} ({t1 - t0:.2f}s) | "
              f"AlphaBeta {m2} nodes={n2:,} ({t2 - t1:.2f}s) | cung nuoc di: {m1 == m2}")