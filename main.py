import time
from board import Board, X, O
from minimax import get_best_move_minimax
from mcts import get_best_move_mcts, LAST_STATS


def print_board(board: Board):
    print("\n" + str(board))
    if board.last_move:
        print(f"Nuoc di gan nhat: {board.last_move}")


def get_human_move(board: Board):
    while True:
        try:
            raw = input("Nhap toa do nuoc di cua ban (hang cot, vi du: 7 7): ").strip()
            parts = raw.split()
            if len(parts) != 2:
                print("Vui long nhap dung 2 so nguyen cach nhau boi dau cach!")
                continue
            r, c = int(parts[0]), int(parts[1])
            if board.is_empty(r, c):
                return r, c
            else:
                print("O nay khong hop le hoac da co quan. Thu lai!")
        except ValueError:
            print("Toa do phai la so nguyen!")


def choose_ai_agent(label: str):
    print(f"\n--- Chon thuat toan cho {label} ---")
    print("1. Minimax + Alpha-Beta")
    print("2. MCTS (Random Rollout)")
    print("3. MCTS (Heuristic Rollout)")
    choice = input("Lua chon (1/2/3): ").strip()
    if choice == "1":
        depth = int(input("Nhap do sau depth (khuyen nghi 2): ").strip() or "2")
        return {"type": "minimax", "depth": depth, "name": f"Minimax-AB(d={depth})"}
    elif choice == "2":
        iters = int(input("Nhap so iterations (khuyen nghi 1000): ").strip() or "1000")
        c = float(input("Nhap he so c (khuyen nghi 1.4): ").strip() or "1.4")
        return {"type": "mcts", "mode": "Random", "iterations": iters, "c": c, "name": f"MCTS-Random(it={iters},c={c})"}
    else:
        iters = int(input("Nhap so iterations (khuyen nghi 1000): ").strip() or "1000")
        c = float(input("Nhap he so c (khuyen nghi 1.4): ").strip() or "1.4")
        return {"type": "mcts", "mode": "Heuristic", "iterations": iters, "c": c, "name": f"MCTS-Heuristic(it={iters},c={c})"}


def get_ai_move(board: Board, agent: dict, player: int):
    t0 = time.time()
    if agent["type"] == "minimax":
        move, nodes = get_best_move_minimax(
            board, depth=agent["depth"], player=player, use_alpha_beta=True
        )
        dt = time.time() - t0
        print(f"[{agent['name']}] Chon {move} | Nodes: {nodes} | Time: {dt:.3f}s")
        return move
    else:
        move, nodes = get_best_move_mcts(
            board, iterations=agent["iterations"], c=agent["c"], mode=agent["mode"], player=player
        )
        dt = time.time() - t0
        print(f"[{agent['name']}] Chon {move} | Nodes: {nodes} | WinRate: {LAST_STATS.get('best_winrate', 0):.2f} | Time: {dt:.3f}s")
        return move


def play_human_vs_ai(board_size: int = 15):
    board = Board(size=board_size, win_length=5)
    ai_agent = choose_ai_agent("AI")
    side = input("Ban muon cam quan nao? (X di truoc / O di sau) [X/O]: ").strip().upper()
    human_player = X if side != "O" else O
    ai_player = -human_player

    print(f"\nBat dau tran dau! Ban: {'X' if human_player == X else 'O'} | AI: {ai_agent['name']}")
    print_board(board)

    while not board.is_terminal():
        current = board.current_player
        if current == human_player:
            r, c = get_human_move(board)
            board.make_move(r, c, human_player)
        else:
            print("\nAI dang suy nghi...")
            move = get_ai_move(board, ai_agent, ai_player)
            if move:
                board.make_move(move[0], move[1], ai_player)
        print_board(board)

    winner = board.check_winner()
    if winner == human_player:
        print("\n>>> CHUC MUNG! BAN DA THANG AI! <<<")
    elif winner == ai_player:
        print(f"\n>>> AI ({ai_agent['name']}) DA THANG! <<<")
    else:
        print("\n>>> VAN DAU HOA! <<<")


def play_ai_vs_ai(board_size: int = 15):
    board = Board(size=board_size, win_length=5)
    agent_x = choose_ai_agent("AI 1 (Cam quan X - di truoc)")
    agent_o = choose_ai_agent("AI 2 (Cam quan O - di sau)")

    print(f"\nBat dau tran dau: {agent_x['name']} (X) vs {agent_o['name']} (O)")
    print_board(board)

    while not board.is_terminal():
        current = board.current_player
        if current == X:
            move = get_ai_move(board, agent_x, X)
        else:
            move = get_ai_move(board, agent_o, O)
        if move:
            board.make_move(move[0], move[1], current)
        print_board(board)

    winner = board.check_winner()
    if winner == X:
        print(f"\n>>> {agent_x['name']} (X) THANG! <<<")
    elif winner == O:
        print(f"\n>>> {agent_o['name']} (O) THANG! <<<")
    else:
        print("\n>>> VAN DAU HOA! <<<")


def main():
    print("==========================================")
    print("   TIC-TAC-TOE / GOMOKU 5-IN-A-ROW AI")
    print("==========================================")
    size_raw = input("Nhap kich thuoc ban co (mac dinh 15): ").strip()
    board_size = int(size_raw) if size_raw.isdigit() else 15

    while True:
        print("\nCHON CHE DO CHOI:")
        print("1. Nguoi vs AI")
        print("2. AI vs AI")
        print("0. Thoat")
        mode = input("Lua chon cua ban: ").strip()

        if mode == "1":
            play_human_vs_ai(board_size)
        elif mode == "2":
            play_ai_vs_ai(board_size)
        elif mode == "0":
            print("Tam biet!")
            break
        else:
            print("Lua chon khong hop le!")


if __name__ == "__main__":
    main()