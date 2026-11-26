import time
import random
import pandas as pd
from board import Board, X, O
from minimax import get_best_move_minimax
from mcts import get_best_move_mcts, LAST_STATS


def random_agent_move(board: Board):
    cands = board.get_candidate_moves(radius=1)
    return random.choice(cands) if cands else None


def play_single_game(board_size, agent_x_fn, agent_o_fn):
    """Chay 1 van dau va thu thap chi so."""
    board = Board(size=board_size, win_length=5)
    nodes_x, nodes_o = [], []
    time_x, time_o = [], []
    root_stats_samples = []

    while not board.is_terminal():
        current = board.current_player
        t0 = time.time()
        if current == X:
            move, nodes, stats = agent_x_fn(board, X)
            dt = time.time() - t0
            nodes_x.append(nodes)
            time_x.append(dt)
            if stats:
                root_stats_samples.append(stats)
        else:
            move, nodes, stats = agent_o_fn(board, O)
            dt = time.time() - t0
            nodes_o.append(nodes)
            time_o.append(dt)

        if move is None:
            break
        board.make_move(move[0], move[1], current)

    return {
        "winner": board.check_winner(),
        "total_moves": len(board.move_history),
        "avg_nodes_x": sum(nodes_x) / len(nodes_x) if nodes_x else 0,
        "avg_nodes_o": sum(nodes_o) / len(nodes_o) if nodes_o else 0,
        "avg_time_x": sum(time_x) / len(time_x) if time_x else 0,
        "avg_time_o": sum(time_o) / len(time_o) if time_o else 0,
        "sample_stats": root_stats_samples[len(root_stats_samples) // 2] if root_stats_samples else {}
    }


def exp1_minimax_vs_alphabeta(board_size=15, num_games=10):
    """Thuc nghiem 1: So sanh so node duyet giua Minimax thuan va Alpha-Beta."""
    print("\n--- [EXP 1] Minimax thuan vs Alpha-Beta ---")
    records = []
    for depth in [1, 2]:
        for g in range(num_games):
            b = Board(size=board_size, win_length=5)
            # Tao 4 nuoc ngau nhien ban dau de cac van khac nhau
            for _ in range(4):
                cands = b.get_candidate_moves(radius=1)
                mv = random.choice(cands)
                b.make_move(mv[0], mv[1], b.current_player)

            t0 = time.time()
            _, nodes_mm = get_best_move_minimax(b, depth=depth, player=X, use_alpha_beta=False, max_branch=8)
            t_mm = time.time() - t0

            t0 = time.time()
            _, nodes_ab = get_best_move_minimax(b, depth=depth, player=X, use_alpha_beta=True, max_branch=8)
            t_ab = time.time() - t0

            reduction = (1 - nodes_ab / nodes_mm) * 100 if nodes_mm > 0 else 0
            records.append({
                "depth": depth,
                "game_id": g + 1,
                "minimax_nodes": nodes_mm,
                "alphabeta_nodes": nodes_ab,
                "reduction_pct": round(reduction, 2),
                "minimax_time": round(t_mm, 4),
                "alphabeta_time": round(t_ab, 4)
            })
            print(f"Depth {depth} | Game {g+1}: Minimax={nodes_mm} nodes, AlphaBeta={nodes_ab} nodes (Giam {reduction:.1f}%)")

    df = pd.DataFrame(records)
    df.to_csv("results_minimax_ab.csv", index=False)
    print("Da luu ket qua Exp 1 vao results_minimax_ab.csv")


def exp2_mcts_c_and_iterations(board_size=15, games_per_setting=50):
    """Thuc nghiem 2: Khao sat MCTS voi c in {0.5, 1.0, 1.4, 2.0} va Iteration in [100..5000].
    Doi thu chuan: Baseline Heuristic / Random agent de do ty le thang khach quan.
    """
    print(f"\n--- [EXP 2] MCTS UCT (c & iterations) - {games_per_setting} van/cau hinh ---")
    c_values = [0.5, 1.0, 1.4, 2.0]
    iter_values = [100, 500, 1000, 2500, 5000]
    records = []
    root_table = []

    def baseline_fn(board, player):
        # Doi thu co dinh: Minimax depth 1 (nhanh va co chien thuat co ban)
        mv, nodes = get_best_move_minimax(board, depth=1, player=player, use_alpha_beta=True, max_branch=8)
        return mv, nodes, None

    for c in c_values:
        for iters in iter_values:
            wins, draws, losses = 0, 0, 0
            total_time = 0.0
            total_nodes = 0.0

            def mcts_fn(board, player):
                mv, nodes = get_best_move_mcts(board, iterations=iters, c=c, mode="Random", player=player)
                return mv, nodes, dict(LAST_STATS)

            for g in range(games_per_setting):
                # Chia deu 50% di truoc (X), 50% di sau (O)
                mcts_is_x = (g % 2 == 0)
                if mcts_is_x:
                    res = play_single_game(board_size, mcts_fn, baseline_fn)
                    w = res["winner"]
                    total_time += res["avg_time_x"]
                    total_nodes += res["avg_nodes_x"]
                    if w == X:
                        wins += 1
                    elif w == O:
                        losses += 1
                    else:
                        draws += 1
                else:
                    res = play_single_game(board_size, baseline_fn, mcts_fn)
                    w = res["winner"]
                    total_time += res["avg_time_o"]
                    total_nodes += res["avg_nodes_o"]
                    if w == O:
                        wins += 1
                    elif w == X:
                        losses += 1
                    else:
                        draws += 1

                if g == 0 and res["sample_stats"]:
                    st = res["sample_stats"]
                    root_table.append({
                        "c": c,
                        "iterations": iters,
                        "root_N": st.get("root_N", 0),
                        "best_N": st.get("best_N", 0),
                        "best_W": round(st.get("best_W", 0), 2),
                        "W_per_N": round(st.get("best_winrate", 0), 4)
                    })

            win_rate = (wins / games_per_setting) * 100
            records.append({
                "c": c,
                "iterations": iters,
                "games": games_per_setting,
                "wins": wins,
                "draws": draws,
                "losses": losses,
                "win_rate": round(win_rate, 2),
                "avg_nodes": round(total_nodes / games_per_setting, 1),
                "avg_time_per_move": round(total_time / games_per_setting, 4)
            })
            print(f"c={c} | iter={iters:4d} -> Win: {wins}/{games_per_setting} ({win_rate:.1f}%), Draw: {draws}, Loss: {losses}")

    pd.DataFrame(records).to_csv("results_mcts_uct.csv", index=False)
    pd.DataFrame(root_table).to_csv("results_mcts_root_stats.csv", index=False)
    print("Da luu ket qua Exp 2 vao results_mcts_uct.csv va results_mcts_root_stats.csv")


def exp3_random_vs_heuristic_rollout(board_size=15, num_games=50):
    """Thuc nghiem 3: So sanh MCTS Random Rollout vs MCTS Heuristic Rollout."""
    print(f"\n--- [EXP 3] Random Rollout vs Heuristic Rollout ({num_games} van) ---")
    records = []

    def mcts_random_fn(board, player):
        mv, nodes = get_best_move_mcts(board, iterations=800, c=1.4, mode="Random", player=player)
        return mv, nodes, None

    def mcts_heur_fn(board, player):
        mv, nodes = get_best_move_mcts(board, iterations=800, c=1.4, mode="Heuristic", player=player)
        return mv, nodes, None

    heur_wins, rand_wins, draws = 0, 0, 0
    t_heur_list, t_rand_list = [], []

    for g in range(num_games):
        heur_is_x = (g % 2 == 0)
        if heur_is_x:
            res = play_single_game(board_size, mcts_heur_fn, mcts_random_fn)
            t_heur_list.append(res["avg_time_x"])
            t_rand_list.append(res["avg_time_o"])
            if res["winner"] == X:
                heur_wins += 1
            elif res["winner"] == O:
                rand_wins += 1
            else:
                draws += 1
        else:
            res = play_single_game(board_size, mcts_random_fn, mcts_heur_fn)
            t_rand_list.append(res["avg_time_x"])
            t_heur_list.append(res["avg_time_o"])
            if res["winner"] == O:
                heur_wins += 1
            elif res["winner"] == X:
                rand_wins += 1
            else:
                draws += 1
        print(f"[Exp 3] Van {g+1}/{num_games} xong ({time.time() - t_game:.1f}s) | "
              f"Ti so Heuristic - Random - Hoa: {heur_wins} - {rand_wins} - {draws}", flush=True)

    records.append({
        "mode": "MCTS-Heuristic",
        "wins": heur_wins,
        "draws": draws,
        "losses": rand_wins,
        "win_rate": round(heur_wins / num_games * 100, 2),
        "avg_time_per_move": round(sum(t_heur_list) / len(t_heur_list), 4)
    })
    records.append({
        "mode": "MCTS-Random",
        "wins": rand_wins,
        "draws": draws,
        "losses": heur_wins,
        "win_rate": round(rand_wins / num_games * 100, 2),
        "avg_time_per_move": round(sum(t_rand_list) / len(t_rand_list), 4)
    })

    df = pd.DataFrame(records)
    df.to_csv("results_rollout_compare.csv", index=False)
    print(df)
    print("Da luu ket qua Exp 3 vao results_rollout_compare.csv")


if __name__ == "__main__":
    print("=== BAT DAU CHAY THUC NGHIEM TONG HOP ===")
    # Luu y: Co the giam so van (vd: games_per_setting=10) khi test nhanh,
    # sau do tang len 50 de lay so lieu chinh thuc cho bao cao.
    exp1_minimax_vs_alphabeta(board_size=15, num_games=10)
    exp2_mcts_c_and_iterations(board_size=15, games_per_setting=10)
    exp3_random_vs_heuristic_rollout(board_size=15, num_games=50)
    print("\n=== HOAN THANH TAT CA THUC NGHIEM! ===")