"""Minimax va Alpha-Beta cho Gomoku 20x20, tuong thich voi board.py cua nhom.

Board API duoc su dung:
    board.grid
    board.size, board.win_length
    board.clone()
    board.make_move(row, col, player)
    board.check_winner()
    board.is_full()
    board.get_candidate_moves(radius=1)

Quy uoc: X = 1 (MAX), O = -1 (MIN), o trong = 0.
"""

from math import inf
from typing import Optional, Tuple, Any

EMPTY = 0
X_PLAYER = 1
O_PLAYER = -1
WIN_SCORE = 10**12

DIRECTIONS = ((0, 1), (1, 0), (1, 1), (1, -1))


def _grid(board):
    """Lay ma tran tu doi tuong Board hoac tu ma tran 2 chieu."""
    return board.grid if hasattr(board, "grid") else board


def _win_length(board):
    return getattr(board, "win_length", 5)


def _count_nodes(counter):
    if counter is None:
        return
    if isinstance(counter, dict):
        counter["count"] = counter.get("count", 0) + 1
    elif isinstance(counter, list):
        if counter:
            counter[0] += 1
        else:
            counter.append(1)
    else:
        raise TypeError("node_counter phai la dict, list hoac None")


def _nodes_value(counter):
    if counter is None:
        return 0
    if isinstance(counter, dict):
        return counter.get("count", 0)
    return counter[0] if counter else 0


def _winner_from_grid(grid, length):
    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    for r in range(rows):
        for c in range(cols):
            player = grid[r][c]
            if player == EMPTY:
                continue
            for dr, dc in DIRECTIONS:
                end_r = r + (length - 1) * dr
                end_c = c + (length - 1) * dc
                if not (0 <= end_r < rows and 0 <= end_c < cols):
                    continue
                if all(grid[r + k * dr][c + k * dc] == player for k in range(length)):
                    return player
    return EMPTY


def _winner(board):
    # Dung check_winner() cua board.py; fallback cho ma tran truc tiep.
    if hasattr(board, "check_winner"):
        return board.check_winner()
    return _winner_from_grid(_grid(board), _win_length(board))


def _is_full(board):
    if hasattr(board, "is_full"):
        return board.is_full()
    return all(cell != EMPTY for row in _grid(board) for cell in row)


def _pattern_score(length, open_ends):
    """Diem theo dung bang heuristic trong phan cong."""
    if length >= 5:
        return WIN_SCORE
    scores = {
        (4, 2): 10_000,
        (4, 1): 1_000,
        (3, 2): 1_000,
        (3, 1): 100,
    }
    return scores.get((length, open_ends), 0)


def _evaluate_one_player(grid, player):
    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    total = 0

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != player:
                continue
            for dr, dc in DIRECTIONS:
                # Chi danh gia tai dau chuoi de tranh dem lap.
                pr, pc = r - dr, c - dc
                if 0 <= pr < rows and 0 <= pc < cols and grid[pr][pc] == player:
                    continue

                length = 0
                nr, nc = r, c
                while 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == player:
                    length += 1
                    nr += dr
                    nc += dc

                before_r, before_c = r - dr, c - dc
                open_ends = 0
                if 0 <= before_r < rows and 0 <= before_c < cols and grid[before_r][before_c] == EMPTY:
                    open_ends += 1
                if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == EMPTY:
                    open_ends += 1
                total += _pattern_score(length, open_ends)
    return total


def evaluate_board(board):
    """Danh gia tu goc nhin X: diem duong tot cho X, diem am tot cho O."""
    grid = _grid(board)
    winner = _winner(board)
    if winner == X_PLAYER:
        return float(WIN_SCORE)
    if winner == O_PLAYER:
        return float(-WIN_SCORE)
    return float(
        _evaluate_one_player(grid, X_PLAYER)
        - _evaluate_one_player(grid, O_PLAYER)
    )

def _get_moves(board, radius=1):
    if hasattr(board, "get_candidate_moves"):
        return board.get_candidate_moves(radius=radius)
    grid = _grid(board)
    rows, cols = len(grid), len(grid[0])
    occupied = [(r, c) for r in range(rows) for c in range(cols) if grid[r][c] != EMPTY]
    if not occupied:
        return [(rows // 2, cols // 2)]
    moves = set()
    for r, c in occupied:
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == EMPTY:
                    moves.add((nr, nc))
    return sorted(moves)


def _make_child(board, move, player):
    r, c = move
    if hasattr(board, "clone") and hasattr(board, "make_move"):
        child = board.clone()
        if not child.make_move(r, c, player):
            raise ValueError(f"Nuoc di khong hop le: {move}")
        return child
    child = [row[:] for row in _grid(board)]
    if child[r][c] != EMPTY:
        raise ValueError(f"O {move} da co quan")
    child[r][c] = player
    return child


def _ordered_moves(board, moves, player, candidate_limit):
    """Uu tien nuoc di tao the co tot; candidate_limit=None de xet tat ca."""
    ranked = []
    for move in moves:
        child = _make_child(board, move, player)
        if _winner(child) == player:
            score = WIN_SCORE
        else:
            score = evaluate_board(child)
        # X muon diem cao, O muon diem thap.
        ranked.append((score if player == X_PLAYER else -score, move))
    ranked.sort(key=lambda item: item[0], reverse=True)
    ordered = [move for _, move in ranked]
    if candidate_limit is not None:
        ordered = ordered[:max(1, candidate_limit)]
    return ordered


def minimax(board, depth, is_maximizing, node_counter=None, candidate_limit=12):
    """Minimax khong cat tia. is_maximizing=True tuong ung X, False tuong ung O."""
    _count_nodes(node_counter)
    winner = _winner(board)
    if winner == X_PLAYER:
        return float(WIN_SCORE + depth)
    if winner == O_PLAYER:
        return float(-WIN_SCORE - depth)
    if depth <= 0 or _is_full(board):
        return evaluate_board(board)

    player = X_PLAYER if is_maximizing else O_PLAYER
    moves = _get_moves(board)
    if not moves:
        return evaluate_board(board)
    moves = _ordered_moves(board, moves, player, candidate_limit)

    if is_maximizing:
        best = -inf
        for move in moves:
            best = max(best, minimax(_make_child(board, move, player), depth - 1,
                                     False, node_counter, candidate_limit))
        return best
    best = inf
    for move in moves:
        best = min(best, minimax(_make_child(board, move, player), depth - 1,
                                 True, node_counter, candidate_limit))
    return best


def alpha_beta(board, depth, alpha=-inf, beta=inf, is_maximizing=True,
               node_counter=None, candidate_limit=12):
    """Minimax co cat tia Alpha-Beta."""
    _count_nodes(node_counter)
    winner = _winner(board)
    if winner == X_PLAYER:
        return float(WIN_SCORE + depth)
    if winner == O_PLAYER:
        return float(-WIN_SCORE - depth)
    if depth <= 0 or _is_full(board):
        return evaluate_board(board)

    player = X_PLAYER if is_maximizing else O_PLAYER
    moves = _get_moves(board)
    if not moves:
        return evaluate_board(board)
    moves = _ordered_moves(board, moves, player, candidate_limit)

    if is_maximizing:
        value = -inf
        for move in moves:
            value = max(value, alpha_beta(_make_child(board, move, player), depth - 1,
                                          alpha, beta, False, node_counter, candidate_limit))
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return value

    value = inf
    for move in moves:
        value = min(value, alpha_beta(_make_child(board, move, player), depth - 1,
                                      alpha, beta, True, node_counter, candidate_limit))
        beta = min(beta, value)
        if alpha >= beta:
            break
    return value


def get_best_move_minimax(board, depth=2, player=X_PLAYER, use_alpha_beta=True,
                          candidate_limit=12):
    """Tra ve (best_move=(row,col), so_node_da_duyet)."""
    if player not in (X_PLAYER, O_PLAYER):
        raise ValueError("player phai la 1 (X) hoac -1 (O)")
    if _winner(board) != EMPTY or _is_full(board):
        return None, 0

    moves = _get_moves(board)
    if not moves:
        return None, 0
    moves = _ordered_moves(board, moves, player, candidate_limit)
    counter = {"count": 0}
    best_move = None
    best_score = -inf if player == X_PLAYER else inf

    for move in moves:
        child = _make_child(board, move, player)
        next_is_maximizing = (player == O_PLAYER)
        if use_alpha_beta:
            score = alpha_beta(child, depth - 1, -inf, inf,
                               next_is_maximizing, counter, candidate_limit)
        else:
            score = minimax(child, depth - 1, next_is_maximizing,
                            counter, candidate_limit)
        if (player == X_PLAYER and score > best_score) or (player == O_PLAYER and score < best_score):
            best_score, best_move = score, move
    return best_move, _nodes_value(counter)


def get_best_move_alpha_beta(board, depth=2, player=X_PLAYER, candidate_limit=12):
    return get_best_move_minimax(board, depth, player, True, candidate_limit)


if __name__ == "__main__":
    from board import Board
    b = Board()
    # Tao tinh huong X co bon quan lien tiep, O dang co luot di.
    for col in range(4):
        b.make_move(10, col + 8, X_PLAYER)
        if col < 3:
            b.make_move(11, col + 8, O_PLAYER)
    print("Winner:", b.check_winner())
    print("Heuristic:", evaluate_board(b))
    print("Minimax:", get_best_move_minimax(b, depth=1, player=X_PLAYER, use_alpha_beta=False))
    print("Alpha-Beta:", get_best_move_minimax(b, depth=1, player=X_PLAYER, use_alpha_beta=True))
