"""Plot MCTS win rate by iteration count and UCT exploration constant."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


C_VALUES = (0.5, 1.0, 1.4, 2.0)
REQUIRED_COLUMNS = {"iteration", "c", "mcts_result"}


def _resolve_input(path: Path) -> Path:
    if path.is_file():
        return path
    if path == Path("results.csv") and Path("result.csv").is_file():
        print("results.csv not found; reading result.csv instead.")
        return Path("result.csv")
    raise FileNotFoundError("Results CSV not found: {}".format(path))


def plot_results(input_path: Path, output_path: Path) -> None:
    csv_path = _resolve_input(input_path)
    results = pd.read_csv(csv_path)
    missing = REQUIRED_COLUMNS.difference(results.columns)
    if missing:
        raise ValueError(
            "CSV is missing required columns: {}".format(", ".join(sorted(missing)))
        )
    if results.empty:
        raise ValueError("CSV contains no benchmark games: {}".format(csv_path))

    results["mcts_result"] = results["mcts_result"].astype(str).str.lower()
    invalid_results = set(results["mcts_result"]) - {"win", "loss", "draw"}
    if invalid_results:
        raise ValueError(
            "Unexpected mcts_result values: {}".format(
                ", ".join(sorted(invalid_results))
            )
        )

    summary = (
        results.assign(mcts_win=results["mcts_result"].eq("win"))
        .groupby(["c", "iteration"], as_index=False)
        .agg(win_rate=("mcts_win", "mean"))
    )
    summary["win_rate"] *= 100

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for c in C_VALUES:
        curve = summary[summary["c"].round(6) == round(c, 6)].sort_values("iteration")
        ax.plot(curve["iteration"], curve["win_rate"], marker="o", label="c = {}".format(c))

    ax.set(
        title="MCTS win rate by iteration and UCT c",
        xlabel="Iteration",
        ylabel="Win rate (%)",
        xlim=(100, 5000),
        ylim=(0, 100),
    )
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(title="UCT exploration")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print("Saved chart to {}".format(output_path))


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot MCTS benchmark win rates.")
    parser.add_argument("--input", type=Path, default=Path("results.csv"),
                        help="input CSV (default: results.csv; falls back to result.csv)")
    parser.add_argument("--output", type=Path, default=Path("chart_uct.png"),
                        help="chart image path (default: chart_uct.png)")
    args = parser.parse_args()
    plot_results(args.input, args.output)


if __name__ == "__main__":
    main()
