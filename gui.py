import time
import random
import pygame

from board import Board, X, O

#gọi AI - > mỗi AI là hàm  f(board) -> (nước đi(row, col), số node đã duyệt)

MINIMAX_DEPTH = 2
MCTS_ITERATIONS = 360
MCTS_C = 1.4


def _random_ai(board):
    return random.choice(board.get_candidate_moves()), 0


def _make_minimax_ai(use_alpha_beta):
    def ai(board):
        from minimax import get_best_move_minimax
        return get_best_move_minimax(
            board, depth=MINIMAX_DEPTH, player=board.current_player,
            use_alpha_beta=use_alpha_beta)
    return ai


def _make_mcts_ai(mode):
    def ai(board):
        from mcts import get_best_move_mcts
        return get_best_move_mcts(
            board, iterations=MCTS_ITERATIONS, c=MCTS_C, mode=mode,
            player=board.current_player)
    return ai


AI_REGISTRY = {
    "Random": _random_ai,
    "Minimax": _make_minimax_ai(False),
    "Alpha-Beta": _make_minimax_ai(True),
    "MCTS Random": _make_mcts_ai("Random"),
    "MCTS Heuristic": _make_mcts_ai("Heuristic"),
}
AI_NAMES = list(AI_REGISTRY.keys())

BOARD_N = 20
CELL = 34
MARGIN = 30
BOARD_PX = CELL * (BOARD_N - 1)
PANEL_W = 270
WIN_W = BOARD_PX + 2 * MARGIN + PANEL_W
WIN_H = BOARD_PX + 2 * MARGIN
FPS = 30
AI_VS_AI_DELAY_MS = 400

BG = (236, 206, 150)
LINE = (90, 65, 35)
PANEL_BG = (38, 42, 52)
TEXT = (235, 235, 235)
MUTED = (160, 165, 175)
X_COLOR = (200, 40, 40)
O_COLOR = (30, 90, 200)
WIN_COLOR = (40, 170, 70)
LAST_COLOR = (250, 190, 30)
BTN = (70, 78, 96)
BTN_HOVER = (95, 105, 130)


def _font(size, bold=False):
    return pygame.font.SysFont("tahoma,verdana,arial", size, bold=bold)


class Button:
    def __init__(self, rect, label):
        self.rect = pygame.Rect(rect)
        self.label = label

    def draw(self, screen, font, mouse):
        color = BTN_HOVER if self.rect.collidepoint(mouse) else BTN
        pygame.draw.rect(screen, color, self.rect, border_radius=8)
        surf = font.render(self.label, True, TEXT)
        screen.blit(surf, surf.get_rect(center=self.rect.center))

    def hit(self, pos):
        return self.rect.collidepoint(pos)

class GameApp:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Tic-Tac-Toe AI (Caro 20x20 - 5 quân)")
        self.screen = pygame.display.set_mode((WIN_W, WIN_H))
        self.clock = pygame.time.Clock()
        self.f_title = _font(34, True)
        self.f_big = _font(22, True)
        self.f_norm = _font(18, True)
        self.f_small = _font(15, True)

        self.state = "menu"         #menu
        self.mode = "HvA"           #người vs AI / AI vs AI
        self.human_first = True     #người đi trước
        self.ai_x = "Alpha-Beta"
        self.ai_o = "MCTS Heuristic"

        #trạng thái
        self.board = Board(BOARD_N)
        self.players = {X: "human", O: "human"}
        self.thinking = False
        self.paused = False
        self.last_ai_time = 0.0
        self.last_ai_nodes = None
        self.last_ai_label = ""
        self.ai_ready_at = 0
        self.message = ""

    @staticmethod
    def cell_to_px(r, c):
        return MARGIN + c * CELL, MARGIN + r * CELL

    @staticmethod
    def px_to_cell(pos):
        x, y = pos
        c = round((x - MARGIN) / CELL)
        r = round((y - MARGIN) / CELL)
        if 0 <= r < BOARD_N and 0 <= c < BOARD_N:
            return r, c
        return None

    #bắt đầu ván
    def start_game(self):
        self.board = Board(BOARD_N)
        if self.mode == "HvA":
            if self.human_first:
                self.players = {X: "human", O: self.ai_o}
            else:
                self.players = {X: self.ai_x, O: "human"}
        else:
            self.players = {X: self.ai_x, O: self.ai_o}
        self.thinking = False
        self.paused = False
        self.last_ai_time = 0.0
        self.last_ai_nodes = None
        self.last_ai_label = ""
        self.message = ""
        self.ai_ready_at = pygame.time.get_ticks() + AI_VS_AI_DELAY_MS
        self.state = "game"

    def is_human_turn(self):
        return self.players[self.board.current_player] == "human"

    #vòng lặp chính
    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif self.state == "game":
                        if event.key == pygame.K_r:
                            self.start_game()
                        elif event.key == pygame.K_m:
                            self.state = "menu"
                        elif event.key == pygame.K_SPACE:
                            self.paused = not self.paused
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if self.state == "menu":
                        self.on_menu_click(event.pos)
                    else:
                        self.on_game_click(event.pos)

            if self.state == "menu":
                self.draw_menu()
            else:
                self.update_ai()
                self.draw_game()

            pygame.display.flip()
            self.clock.tick(FPS)
        pygame.quit()

    #logic
    def on_game_click(self, pos):
        if self.board.is_terminal():
            return
        if not self.is_human_turn() or self.thinking:
            return
        cell = self.px_to_cell(pos)
        if cell is None:
            return
        if not self.board.make_move(cell[0], cell[1]):
            self.message = "Ô này đã có quân!"
        else:
            self.message = ""

    def update_ai(self):
        b = self.board
        if b.is_terminal() or self.is_human_turn() or self.paused:
            return
        if pygame.time.get_ticks() < self.ai_ready_at:
            return

        #frame 1: suy nghĩ - frame 2: nước đi
        if not self.thinking:
            self.thinking = True
            return

        name = self.players[b.current_player]
        t0 = time.time()
        try:
            move, nodes = AI_REGISTRY[name](b.clone())
        except Exception as exc: #catch lỗi
            self.message = f"Lỗi AI '{name}': {exc}"
            self.paused = True
            self.thinking = False
            return
        self.last_ai_time = time.time() - t0
        self.last_ai_nodes = nodes
        self.last_ai_label = name

        if move is None or not b.make_move(move[0], move[1]):
            self.message = f"AI '{name}' trả về nước đi không hợp lệ: {move}"
            self.paused = True
        self.thinking = False
        self.ai_ready_at = pygame.time.get_ticks() + (
            AI_VS_AI_DELAY_MS if self.mode == "AvA" else 0)

    #menu
    def menu_buttons(self):
        x, w, h = WIN_W // 2 - 190, 380, 48
        y0 = 190
        buttons = {
            "mode": Button((x, y0, w, h),
                           "Chế độ: " + ("Người vs AI" if self.mode == "HvA" else "AI vs AI")),
        }
        y = y0 + 65
        if self.mode == "HvA":
            buttons["first"] = Button(
                (x, y, w, h),
                "Người cầm: " + ("X (đi trước)" if self.human_first else "O (đi sau)"))
            y += 65
            if self.human_first:
                buttons["ai_o"] = Button((x, y, w, h), f"AI (O): {self.ai_o}")
            else:
                buttons["ai_x"] = Button((x, y, w, h), f"AI (X): {self.ai_x}")
            y += 65
        else:
            buttons["ai_x"] = Button((x, y, w, h), f"AI 1 (X, đi trước): {self.ai_x}")
            y += 65
            buttons["ai_o"] = Button((x, y, w, h), f"AI 2 (O): {self.ai_o}")
            y += 65
        buttons["start"] = Button((x, y + 15, w, 56), "BẮT ĐẦU")
        return buttons

    def on_menu_click(self, pos):
        for key, btn in self.menu_buttons().items():
            if not btn.hit(pos):
                continue
            if key == "mode":
                self.mode = "AvA" if self.mode == "HvA" else "HvA"
            elif key == "first":
                self.human_first = not self.human_first
            elif key == "ai_x":
                self.ai_x = AI_NAMES[(AI_NAMES.index(self.ai_x) + 1) % len(AI_NAMES)]
            elif key == "ai_o":
                self.ai_o = AI_NAMES[(AI_NAMES.index(self.ai_o) + 1) % len(AI_NAMES)]
            elif key == "start":
                self.start_game()
            break

    def draw_menu(self):
        self.screen.fill(PANEL_BG)
        title = self.f_title.render("TIC-TAC-TOE AI", True, TEXT)
        self.screen.blit(title, title.get_rect(center=(WIN_W // 2, 90)))
        sub = self.f_norm.render("Caro 20x20 - thắng khi có 5 quân liên tiếp", True, MUTED)
        self.screen.blit(sub, sub.get_rect(center=(WIN_W // 2, 135)))

        mouse = pygame.mouse.get_pos()
        for btn in self.menu_buttons().values():
            btn.draw(self.screen, self.f_norm, mouse)

        hint = self.f_small.render("Bấm vào nút để đổi tuỳ chọn  |  ESC: thoát", True, MUTED)
        self.screen.blit(hint, hint.get_rect(center=(WIN_W // 2, WIN_H - 30)))

    #vẽ bàn cờ
    def draw_piece(self, r, c, player):
        cx, cy = self.cell_to_px(r, c)
        rad = CELL // 2 - 4
        if player == X:
            pygame.draw.line(self.screen, X_COLOR, (cx - rad, cy - rad), (cx + rad, cy + rad), 4)
            pygame.draw.line(self.screen, X_COLOR, (cx - rad, cy + rad), (cx + rad, cy - rad), 4)
        else:
            pygame.draw.circle(self.screen, O_COLOR, (cx, cy), rad, 4)

    def draw_board(self):
        b = self.board
        pygame.draw.rect(self.screen, BG, (0, 0, BOARD_PX + 2 * MARGIN, WIN_H))

        #lưới
        for i in range(BOARD_N):
            p = MARGIN + i * CELL
            pygame.draw.line(self.screen, LINE, (MARGIN, p), (MARGIN + BOARD_PX, p), 1)
            pygame.draw.line(self.screen, LINE, (p, MARGIN), (p, MARGIN + BOARD_PX), 1)

        #index hàng/cột
        for i in range(BOARD_N):
            p = MARGIN + i * CELL
            s = self.f_small.render(str(i), True, LINE)
            self.screen.blit(s, s.get_rect(center=(p, 12)))
            self.screen.blit(s, s.get_rect(center=(12, p)))

        #lượt của người
        if (not b.is_terminal() and self.is_human_turn() and self.state == "game"):
            cell = self.px_to_cell(pygame.mouse.get_pos())
            if cell and b.is_empty(*cell):
                cx, cy = self.cell_to_px(*cell)
                pygame.draw.circle(self.screen, (200, 170, 110), (cx, cy), CELL // 2 - 5, 2)

        #nước đi gần nhất
        if b.last_move:
            cx, cy = self.cell_to_px(*b.last_move)
            pygame.draw.rect(self.screen, LAST_COLOR,
                             (cx - CELL // 2 + 2, cy - CELL // 2 + 2, CELL - 4, CELL - 4), 3)

        #quân cờ
        for r, c, player in b.move_history:
            self.draw_piece(r, c, player)

        #chuỗi thắng
        for r, c in b.winning_line:
            cx, cy = self.cell_to_px(r, c)
            pygame.draw.rect(self.screen, WIN_COLOR,
                             (cx - CELL // 2 + 1, cy - CELL // 2 + 1, CELL - 2, CELL - 2), 4)

    def draw_panel(self):
        b = self.board
        x0 = BOARD_PX + 2 * MARGIN
        pygame.draw.rect(self.screen, PANEL_BG, (x0, 0, PANEL_W, WIN_H))
        x = x0 + 18
        y = 22

        def line(text, font, color=TEXT, dy=30):
            nonlocal y
            self.screen.blit(font.render(text, True, color), (x, y))
            y += dy

        line("TIC-TAC-TOE AI", self.f_big, dy=40)

        def pname(p):
            return "Người" if self.players[p] == "human" else self.players[p]

        line(f"X: {pname(X)}", self.f_norm, X_COLOR, 26)
        line(f"O: {pname(O)}", self.f_norm, (90, 150, 255), 40)

        # trạng thái
        if b.winner:
            sym = "X" if b.winner == X else "O"
            line(f"{sym} THẮNG!", self.f_big, WIN_COLOR, 36)
        elif b.is_full():
            line("HÒA!", self.f_big, LAST_COLOR, 36)
        else:
            sym = "X" if b.current_player == X else "O"
            line(f"Lượt: {sym}", self.f_big, TEXT, 32)
            if self.paused:
                line("(Tạm dừng)", self.f_norm, LAST_COLOR, 28)
            elif self.thinking:
                line("AI đang suy nghĩ...", self.f_norm, LAST_COLOR, 28)
            elif self.is_human_turn():
                line("Mời bạn đi", self.f_norm, MUTED, 28)
            else:
                y += 28

        y += 10
        line(f"Số nước đã đi: {len(b.move_history)}", self.f_norm, MUTED, 26)
        if self.last_ai_label:
            line(f"AI vừa đi: {self.last_ai_label}", self.f_small, MUTED, 22)
            line(f"Thời gian: {self.last_ai_time:.3f}s", self.f_small, MUTED, 22)
            nodes = "-" if self.last_ai_nodes is None else self.last_ai_nodes
            line(f"Số node duyệt: {nodes}", self.f_small, MUTED, 22)

        if self.message:
            y += 8
            cur = ""
            for ch in self.message:
                if self.f_small.size(cur + ch)[0] > PANEL_W - 36:
                    line(cur, self.f_small, (255, 120, 120), 20)
                    cur = ch
                else:
                    cur += ch
            line(cur, self.f_small, (255, 120, 120), 20)

        #phím tắt
        hints = ["R: ván mới", "M: về menu", "SPACE: tạm dừng", "ESC: thoát"]
        yy = WIN_H - 22 * len(hints) - 14
        for h in hints:
            self.screen.blit(self.f_small.render(h, True, MUTED), (x, yy))
            yy += 22

    def draw_game(self):
        self.draw_board()
        self.draw_panel()


def run_gui():
    GameApp().run()


if __name__ == "__main__":
    run_gui()