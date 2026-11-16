"""
Conventions:
    0  : empty cell
    1  : X (whoever first move)
   -1  : O
"""

from typing import List, Tuple

EMPTY = 0
X = 1
O = -1

DIRECTIONS = ((0, 1), (1, 0), (1, 1), (1, -1))


class Board:
    def __init__(self, size: int = 20, win_length: int = 5):
        self.size = size
        self.win_length = win_length
        self.grid: List[List[int]] = [[EMPTY] * size for _ in range(size)]
        self.current_player = X
        self.move_history: List[Tuple[int, int, int]] = []
        self.last_move = None
        self.winner = EMPTY
        self.winning_line: List[Tuple[int, int]] = []

    #check for board boundary and position
    def in_bounds(self, row: int, col: int) -> bool:
        return 0 <= row < self.size and 0 <= col < self.size

    def is_empty(self, row: int, col: int) -> bool:
        return self.in_bounds(row, col) and self.grid[row][col] == EMPTY

    def get(self, row: int, col: int) -> int:
        return self.grid[row][col]

    def make_move(self, row: int, col: int, player: int = None) -> bool:
        #put player in row,col; false if unavailable
        if player is None:
            player = self.current_player
        if player not in (X, O):
            return False
        if self.winner != EMPTY or not self.is_empty(row, col):
            return False

        self.grid[row][col] = player
        self.move_history.append((row, col, player))
        self.last_move = (row, col)
        self.current_player = -player

        line = self._find_line_through(row, col, player)
        if line:
            self.winner = player
            self.winning_line = line
        return True

    def undo_move(self) -> bool:
        #undo the last move
        if not self.move_history:
            return False
        row, col, player = self.move_history.pop()
        self.grid[row][col] = EMPTY
        self.current_player = player
        self.last_move = (self.move_history[-1][0], self.move_history[-1][1]) \
            if self.move_history else None
        self.winner = EMPTY
        self.winning_line = []
        return True

    #winning condition check
    def _find_line_through(self, row: int, col: int, player: int) -> List[Tuple[int, int]]:
        #search for win length at row,col; return the result board
        n = self.win_length
        for dr, dc in DIRECTIONS:
            cells = [(row, col)]
            #move in 2 directions
            for sign in (1, -1):
                r, c = row + sign * dr, col + sign * dc
                while self.in_bounds(r, c) and self.grid[r][c] == player:
                    cells.append((r, c))
                    r += sign * dr
                    c += sign * dc
            if len(cells) >= n:
                cells.sort()
                return cells
        return []

    def check_winner(self) -> int:
        # 1 -> X won, -1 -> O won and 0 for draw
        # 5 consecutive X or O -> winning condition
        return self.winner

    def is_full(self) -> bool:
        return len(self.move_history) >= self.size * self.size

    def is_terminal(self) -> bool:
        return self.winner != EMPTY or self.is_full()

    # generate candidate moves
    def get_candidate_moves(self, radius: int = 1) -> List[Tuple[int, int]]:
        # empty cells in radius of used cell
        # if board empty return middle cell
        if not self.move_history:
            mid = self.size // 2
            return [(mid, mid)]

        candidates = set()
        for r, c, _ in self.move_history:
            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    nr, nc = r + dr, c + dc
                    if self.in_bounds(nr, nc) and self.grid[nr][nc] == EMPTY:
                        candidates.add((nr, nc))
        return sorted(candidates)

    def get_empty_cells(self) -> List[Tuple[int, int]]:
        #used for random mode of mtcs
        return [(r, c) for r in range(self.size) for c in range(self.size)
                if self.grid[r][c] == EMPTY]

    #clone board func
    def clone(self) -> "Board":
        new = Board.__new__(Board)
        new.size = self.size
        new.win_length = self.win_length
        new.grid = [row[:] for row in self.grid]
        new.current_player = self.current_player
        new.move_history = self.move_history[:]
        new.last_move = self.last_move
        new.winner = self.winner
        new.winning_line = self.winning_line[:]
        return new

    #debug
    def __str__(self) -> str:
        symbols = {EMPTY: ".", X: "X", O: "O"}
        header = "   " + " ".join(f"{c % 10}" for c in range(self.size))
        rows = [f"{r:2d} " + " ".join(symbols[v] for v in row)
                for r, row in enumerate(self.grid)]
        return "\n".join([header] + rows)


if __name__ == "__main__":
    #test 
    b = Board()
    for i in range(5):
        b.make_move(5, i, X)
        if i < 4:
            b.make_move(6, i, O)
    print(b)
    print("Winner:", b.check_winner(), b.winning_line)