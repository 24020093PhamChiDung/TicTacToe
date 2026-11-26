import os
import pandas as pd
import matplotlib.pyplot as plt


def plot_mcts_uct():
    if not os.path.exists("results_mcts_uct.csv"):
        print("Khong tim thay file results_mcts_uct.csv. Hay chay benchmark.py truoc!")
        return

    df = pd.read_csv("results_mcts_uct.csv")
    plt.figure(figsize=(10, 6))

    c_values = sorted(df["c"].unique())
    markers = ["o", "s", "^", "D"]

    for idx, c in enumerate(c_values):
        sub = df[df["c"] == c].sort_values("iterations")
        plt.plot(
            sub["iterations"],
            sub["win_rate"],
            marker=markers[idx % len(markers)],
            linewidth=2,
            label=f"c = {c}"
        )

    plt.title("Ty le thang cua MCTS theo so luot lap (Iteration) va he so c", fontsize=13, fontweight="bold")
    plt.xlabel("So luot lap (Iterations)", fontsize=11)
    plt.ylabel("Ty le thang (%)", fontsize=11)
    plt.ylim(0, 105)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(title="He so kham pha (c)", fontsize=10)
    plt.tight_layout()
    plt.savefig("chart_uct.png", dpi=300)
    print("Da luu bieu do UCT thanh cong tai: chart_uct.png")
    plt.close()


def plot_minimax_comparison():
    if not os.path.exists("results_minimax_ab.csv"):
        return

    df = pd.read_csv("results_minimax_ab.csv")
    summary = df.groupby("depth")[["minimax_nodes", "alphabeta_nodes"]].mean().reset_index()

    x = range(len(summary))
    width = 0.35

    plt.figure(figsize=(8, 5))
    plt.bar([i - width/2 for i in x], summary["minimax_nodes"], width, label="Minimax thuan", color="#e07a5f")
    plt.bar([i + width/2 for i in x], summary["alphabeta_nodes"], width, label="Minimax + Alpha-Beta", color="#3d405b")

    plt.xticks(list(x), [f"Depth = {d}" for d in summary["depth"]])
    plt.ylabel("So nut trung binh da duyet (Nodes)", fontsize=11)
    plt.title("So sanh so nut duyet: Minimax vs Alpha-Beta Pruning", fontsize=12, fontweight="bold")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("chart_minimax_nodes.png", dpi=300)
    print("Da luu bieu do so sanh Minimax tai: chart_minimax_nodes.png")
    plt.close()


def plot_rollout_comparison():
    if not os.path.exists("results_rollout_compare.csv"):
        return

    df = pd.read_csv("results_rollout_compare.csv")
    plt.figure(figsize=(7, 5))
    bars = plt.bar(df["mode"], df["win_rate"], color=["#2a9d8f", "#e9c46a"], width=0.5)

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 1.5, f"{yval:.1f}%", ha="center", fontweight="bold")

    plt.ylim(0, 105)
    plt.ylabel("Ty le thang (%)", fontsize=11)
    plt.title("So sanh hieu nang: Random Rollout vs Heuristic Rollout", fontsize=12, fontweight="bold")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("chart_rollout.png", dpi=300)
    print("Da luu bieu do so sanh Rollout tai: chart_rollout.png")
    plt.close()


if __name__ == "__main__":
    plot_mcts_uct()
    plot_minimax_comparison()
    plot_rollout_comparison()