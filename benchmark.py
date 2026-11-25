"""Run MCTS against minimax and save per-game benchmark results to CSV."""

import argparse
import random
import time
from pathlib import Path
from typing import Dict, List, Sequence

import pandas as pd

from board import Board
from minimax import get_best_move_minimax
import mcts


DEFAULT_ITERATIONS = (100, 500, 1000, 2000, 3000, 4000, 5000)
DEFAULT_C_VALUES = (0.5, 1.0, 1.4, 2.0)
CSV_COLUMNS = (
    "iteration",
    "c",
    "game",
    "first_player",
    "winner",
    "mcts_result",
    "elapsed_seconds",
    "minimax_nodes",
    "mcts_N",
    "mcts_W",
)


def play_game(iterations: int, c: float, game_number: int, board_size: int,
              win_length: int, minimax_depth: int) -> Dict[str, object]:
    """Play one game; even-numbered games give MCTS the first move."""
    first_player = "MCTS" if game_number % 2 == 1 else "Minimax"
    mcts_player = 1 if first_player == "MCTS" else -1
    board = Board(size=board_size, win_length=win_length)
    minimax_nodes = 0
    mcts_N = 0
    mcts_W = 0.0
    started_at = time.time()

    while not board.is_terminal():
        if board.current_player == mcts_player:
            move, _ = mcts.get_best_move_mcts(
                board, iterations=iterations, c=c, mode="Heuristic"
            )
            stats = mcts.LAST_STATS
            mcts_N += stats.get("root_N", 0)
            mcts_W += stats.get("best_W", 0.0)
        else:
            move, nodes = get_best_move_minimax(
                board, depth=minimax_depth, player=board.current_player
            )
            minimax_nodes += nodes

        if move is None or not board.make_move(move[0], move[1], board.current_player):
            raise RuntimeError(
                "An AI returned an invalid move before the game ended "
                "(iteration={}, c={}, game={}).".format(iterations, c, game_number)
            )

    elapsed_seconds = time.time() - started_at
    if board.winner == 0:
        winner = "Draw"
        mcts_result = "draw"
    elif board.winner == mcts_player:
        winner = "MCTS"
        mcts_result = "win"
    else:
        winner = "Minimax"
        mcts_result = "loss"

    return {
        "iteration": iterations,
        "c": c,
        "game": game_number,
        "first_player": first_player,
        "winner": winner,
        "mcts_result": mcts_result,
        "elapsed_seconds": elapsed_seconds,
        "minimax_nodes": minimax_nodes,
        "mcts_N": mcts_N,
        "mcts_W": mcts_W,
    }


def run_benchmark(games: int, iterations_values: Sequence[int],
                  c_values: Sequence[float], board_size: int, win_length: int,
                  minimax_depth: int, output: Path, seed: int = None) -> pd.DataFrame:
    # if games < 50:
    #     raise ValueError("--games must be at least 50.")
    if board_size < win_length or win_length < 1:
        raise ValueError("--board-size must be >= --win-length >= 1.")
    if minimax_depth < 1:
        raise ValueError("--minimax-depth must be at least 1.")
    if not iterations_values or any(value < 1 for value in iterations_values):
        raise ValueError("Provide at least one positive MCTS iteration count.")
    if not c_values or any(value < 0 for value in c_values):
        raise ValueError("Provide at least one non-negative UCT c value.")
    if seed is not None:
        random.seed(seed)

    rows: List[Dict[str, object]] = []
    for iterations in iterations_values:
        for c in c_values:
            print("Running iteration={}, c={} ({} games)".format(iterations, c, games))
            for game_number in range(1, games + 1):
                rows.append(
                    play_game(
                        iterations, c, game_number, board_size, win_length,
                        minimax_depth
                    )
                )

    results = pd.DataFrame(rows, columns=CSV_COLUMNS)
    output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output, index=False)
    print("Saved {} games to {}".format(len(results), output))
    return results


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark MCTS against minimax, alternating the first player."
    )
    parser.add_argument("--games", type=_positive_int, default=50,
                        help="games per (iteration, c) pair; minimum 50 (default: 50)")
    parser.add_argument("--iterations", type=_positive_int, nargs="+",
                        default=DEFAULT_ITERATIONS,
                        help="MCTS iterations to test (default: 100 500 1000 2000 3000 4000 5000)")
    parser.add_argument("--c-values", type=float, nargs="+", default=DEFAULT_C_VALUES,
                        help="UCT exploration values (default: 0.5 1.0 1.4 2.0)")
    parser.add_argument("--board-size", type=_positive_int, default=20)
    parser.add_argument("--win-length", type=_positive_int, default=5)
    parser.add_argument("--minimax-depth", type=_positive_int, default=2)
    parser.add_argument("--output", type=Path, default=Path("result.csv"),
                        help="output CSV path (default: result.csv)")
    parser.add_argument("--seed", type=int, default=None,
                        help="optional random seed for repeatable runs")
    args = parser.parse_args()

    run_benchmark(
        games=args.games,
        iterations_values=args.iterations,
        c_values=args.c_values,
        board_size=args.board_size,
        win_length=args.win_length,
        minimax_depth=args.minimax_depth,
        output=args.output,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
