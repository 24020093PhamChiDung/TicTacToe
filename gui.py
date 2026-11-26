import time
import random
import pygame
from pygame import gfxdraw

from board import Board, X, O

MINIMAX_DEPTH = 2
MCTS_ITERATIONS = 360
MCTS_C = 1.4


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
    "Minimax": _make_minimax_ai(False),
    "Alpha-Beta": _make_minimax_ai(True),
    "MCTS Random": _make_mcts_ai("Random"),
    "MCTS Heuristic": _make_mcts_ai("Heuristic"),
}
AI_NAMES = list(AI_REGISTRY.keys())

BOARD_N = 20
CELL = 34
MARGIN = 36
BOARD_PX = CELL * BOARD_N
PANEL_W = 280
WIN_W = BOARD_PX + 2 * MARGIN + PANEL_W
WIN_H = BOARD_PX + 2 * MARGIN
FPS = 30
AI_VS_AI_DELAY_MS = 400

# --- BẢNG MÀU MIDNIGHT SLATE & LUXURY NEON (DỊU MẮT, SANG TRỌNG) ---
BG_OUTER = (15, 23, 42)         # Nền xanh đen sâu (Slate 900)
BOARD_BG = (30, 41, 59)         # Mặt bàn cờ xám xanh trầm (Slate 800)
BOARD_SHADOW = (8, 13, 26)      # Bóng đổ sâu dưới bàn cờ
GRID_DARK = (19, 27, 42)        # Rãnh khắc tối
GRID_LIGHT = (51, 65, 85)       # Viền sáng nhẹ tạo độ nổi 3D (Slate 700)
PANEL_BG = (20, 29, 49)         # Nền thanh điều khiển bên phải
CARD_BG = (35, 48, 70)          # Nền thẻ thông tin

TEXT = (248, 250, 252)          # Trắng ngà sắc nét
MUTED = (148, 163, 184)         # Xám bạc dịu mắt

# Quân X: Xanh Ngọc Bích Sáng (Cyan / Sky Glow)
X_MAIN = (56, 189, 248)
X_DARK = (3, 105, 161)
X_LIGHT = (186, 230, 253)

# Quân O: Đỏ San Hô Rực (Coral Rose)
O_MAIN = (251, 113, 133)
O_DARK = (159, 18, 57)
O_LIGHT = (254, 205, 211)

LAST_COLOR = (250, 204, 21)     # Vàng hổ phách đánh dấu nước vừa đi
BTN = (45, 59, 85)
BTN_HOVER = (65, 84, 118)
BTN_ACCENT = (14, 165, 233)
BTN_ACCENT_HOVER = (56, 189, 248)


def _font(size, bold=False):
    return pygame.font.SysFont("segoeui,tahoma,verdana,arial", size, bold=bold)


class Button:
    def __init__(self, rect, label, accent=False):
        self.rect = pygame.Rect(rect)
        self.label = label
        self.accent = accent

    def draw(self, screen, font, mouse):
        hover = self.rect.collidepoint(mouse)
        if self.accent:
            color = BTN_ACCENT_HOVER if hover else BTN_ACCENT
            shadow_col = (7, 89, 133)
        else:
            color = BTN_HOVER if hover else BTN
            shadow_col = (12, 19, 34)

        shadow_rect = self.rect.move(0, 4)
        pygame.draw.rect(screen, shadow_col, shadow_rect, border_radius=12)
        pygame.draw.rect(screen, color, self.rect, border_radius=12)
        pygame.draw.rect(screen, (255, 255, 255, 35), self.rect, width=1, border_radius=12)

        surf = font.render(self.label, True, TEXT)
        screen.blit(surf, surf.get_rect(center=self.rect.center))

    def hit(self, pos):
        return self.rect.collidepoint(pos)


class GameApp:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Tic-Tac-Toe AI (Caro 20x20)")
        self.screen = pygame.display.set_mode((WIN_W, WIN_H))
        self.clock = pygame.time.Clock()
        self.f_title = _font(36, True)
        self.f_big = _font(22, True)
        self.f_norm = _font(17, True)
        self.f_small = _font(14, True)

        self.state = "menu"
        self.mode = "HvA"
        self.human_first = True
        self.ai_x = "Alpha-Beta"
        self.ai_o = "MCTS Heuristic"

        self.board = Board(BOARD_N)
        self.players = {X: "human", O: "human"}
        self.thinking = False
        self.paused = False
        self.last_ai_time = 0.0
        self.last_ai_nodes = None
        self.last_ai_label = ""
        self.ai_ready_at = 0
        self.message = ""

        rng = random.Random(42)
        self.stars = [
            (rng.randint(10, WIN_W - 10), rng.randint(10, WIN_H - 10), rng.choice([1, 2]))
            for _ in range(45)
        ]

    @staticmethod
    def cell_to_px(r, c):
        return MARGIN + c * CELL + CELL // 2, MARGIN + r * CELL + CELL // 2

    @staticmethod
    def px_to_cell(pos):
        x, y = pos
        dx = x - MARGIN
        dy = y - MARGIN
        if 0 <= dx < BOARD_PX and 0 <= dy < BOARD_PX:
            return int(dy // CELL), int(dx // CELL)
        return None

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

    def on_game_click(self, pos):
        if self.board.is_terminal() or not self.is_human_turn() or self.thinking:
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
        if not self.thinking:
            self.thinking = True
            return

        name = self.players[b.current_player]
        t0 = time.time()
        try:
            move, nodes = AI_REGISTRY[name](b.clone())
        except Exception as exc:
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

    def draw_stars(self):
        for sx, sy, sr in self.stars:
            pygame.draw.circle(self.screen, (71, 85, 105), (sx, sy), sr)

    def _draw_x_shape(self, cx, cy, rad, color, thick):
        p1 = (cx - rad, cy - rad)
        p2 = (cx + rad, cy + rad)
        p3 = (cx - rad, cy + rad)
        p4 = (cx + rad, cy - rad)
        pygame.draw.line(self.screen, color, p1, p2, thick)
        pygame.draw.line(self.screen, color, p3, p4, thick)
        r_cap = thick // 2
        for pt in (p1, p2, p3, p4):
            pygame.draw.circle(self.screen, color, pt, r_cap)

    def draw_piece(self, r, c, player, is_win=False):
        cx, cy = self.cell_to_px(r, c)
        rad = CELL // 2 - 6

        if player == X:
            if is_win:
                glow_surf = pygame.Surface((CELL * 2, CELL * 2), pygame.SRCALPHA)
                pygame.draw.circle(glow_surf, (56, 189, 248, 80), (CELL, CELL), CELL - 4)
                self.screen.blit(glow_surf, (cx - CELL, cy - CELL))

            self._draw_x_shape(cx, cy + 3, rad, X_DARK, 7)
            self._draw_x_shape(cx, cy, rad, X_MAIN, 6)
            self._draw_x_shape(cx, cy - 1, rad - 1, X_LIGHT, 2)
        else:
            if is_win:
                glow_surf = pygame.Surface((CELL * 2, CELL * 2), pygame.SRCALPHA)
                pygame.draw.circle(glow_surf, (251, 113, 133, 80), (CELL, CELL), CELL - 4)
                self.screen.blit(glow_surf, (cx - CELL, cy - CELL))

            gfxdraw.aacircle(self.screen, cx, cy + 3, rad, O_DARK)
            pygame.draw.circle(self.screen, O_DARK, (cx, cy + 3), rad, 6)
            pygame.draw.circle(self.screen, O_MAIN, (cx, cy), rad, 5)
            pygame.draw.arc(
                self.screen, O_LIGHT,
                (cx - rad + 1, cy - rad + 1, (rad - 1) * 2, (rad - 1) * 2),
                0.4, 2.7, 2
            )

    def draw_board(self):
        b = self.board
        self.screen.fill(BG_OUTER)
        self.draw_stars()

        board_rect = pygame.Rect(MARGIN - 6, MARGIN - 6, BOARD_PX + 12, BOARD_PX + 12)
        pygame.draw.rect(self.screen, BOARD_SHADOW, board_rect.move(0, 8), border_radius=16)
        pygame.draw.rect(self.screen, BOARD_BG, board_rect, border_radius=16)
        pygame.draw.rect(self.screen, GRID_LIGHT, board_rect, width=2, border_radius=16)

        for i in range(1, BOARD_N):
            p = MARGIN + i * CELL
            pygame.draw.line(self.screen, GRID_DARK, (MARGIN + 2, p), (MARGIN + BOARD_PX - 2, p), 2)
            pygame.draw.line(self.screen, GRID_LIGHT, (MARGIN + 2, p + 1), (MARGIN + BOARD_PX - 2, p + 1), 1)

            pygame.draw.line(self.screen, GRID_DARK, (p, MARGIN + 2), (p, MARGIN + BOARD_PX - 2), 2)
            pygame.draw.line(self.screen, GRID_LIGHT, (p + 1, MARGIN + 2), (p + 1, MARGIN + BOARD_PX - 2), 1)

        for i in range(BOARD_N):
            p = MARGIN + i * CELL + CELL // 2
            s = self.f_small.render(str(i), True, MUTED)
            self.screen.blit(s, s.get_rect(center=(p, 16)))
            self.screen.blit(s, s.get_rect(center=(16, p)))

        if not b.is_terminal() and self.is_human_turn() and self.state == "game":
            cell = self.px_to_cell(pygame.mouse.get_pos())
            if cell and b.is_empty(*cell):
                cx, cy = self.cell_to_px(*cell)
                ghost = pygame.Surface((CELL, CELL), pygame.SRCALPHA)
                if b.current_player == X:
                    rad = CELL // 2 - 7
                    pygame.draw.line(ghost, (56, 189, 248, 120),
                                     (CELL // 2 - rad, CELL // 2 - rad),
                                     (CELL // 2 + rad, CELL // 2 + rad), 4)
                    pygame.draw.line(ghost, (56, 189, 248, 120),
                                     (CELL // 2 - rad, CELL // 2 + rad),
                                     (CELL // 2 + rad, CELL // 2 - rad), 4)
                else:
                    pygame.draw.circle(ghost, (251, 113, 133, 120),
                                       (CELL // 2, CELL // 2), CELL // 2 - 6, 4)
                self.screen.blit(ghost, (cx - CELL // 2, cy - CELL // 2))

        if b.last_move:
            cx, cy = self.cell_to_px(*b.last_move)
            rect = pygame.Rect(cx - CELL // 2 + 3, cy - CELL // 2 + 3, CELL - 6, CELL - 6)
            pygame.draw.rect(self.screen, LAST_COLOR, rect, width=2, border_radius=6)

        win_set = set(b.winning_line)
        for r, c, player in b.move_history:
            self.draw_piece(r, c, player, is_win=((r, c) in win_set))

        if len(b.winning_line) >= 2:
            p_start = self.cell_to_px(*b.winning_line[0])
            p_end = self.cell_to_px(*b.winning_line[-1])
            line_col = X_MAIN if b.winner == X else O_MAIN
            pygame.draw.line(self.screen, line_col, p_start, p_end, 6)
            pygame.draw.line(self.screen, (255, 255, 255), p_start, p_end, 2)

    def menu_buttons(self):
        x, w, h = WIN_W // 2 - 200, 400, 50
        y0 = 210
        buttons = {
            "mode": Button((x, y0, w, h),
                           "Chế độ: " + ("Người vs AI" if self.mode == "HvA" else "AI vs AI")),
        }
        y = y0 + 66
        if self.mode == "HvA":
            buttons["first"] = Button(
                (x, y, w, h),
                "Người cầm: " + ("X (đi trước)" if self.human_first else "O (đi sau)"))
            y += 66
            if self.human_first:
                buttons["ai_o"] = Button((x, y, w, h), f"AI (O): {self.ai_o}")
            else:
                buttons["ai_x"] = Button((x, y, w, h), f"AI (X): {self.ai_x}")
            y += 66
        else:
            buttons["ai_x"] = Button((x, y, w, h), f"AI 1 (X, đi trước): {self.ai_x}")
            y += 66
            buttons["ai_o"] = Button((x, y, w, h), f"AI 2 (O): {self.ai_o}")
            y += 66
        buttons["start"] = Button((x, y + 16, w, 56), "BẮT ĐẦU CHƠI", accent=True)
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
        self.screen.fill(BG_OUTER)
        self.draw_stars()

        self._draw_x_shape(130, WIN_H - 130, 55, X_DARK, 22)
        self._draw_x_shape(130, WIN_H - 138, 55, X_MAIN, 20)
        pygame.draw.circle(self.screen, O_DARK, (WIN_W - 130, WIN_H - 125), 62, 20)
        pygame.draw.circle(self.screen, O_MAIN, (WIN_W - 130, WIN_H - 133), 62, 18)

        title = self.f_title.render("TIC-TAC-TOE AI", True, X_MAIN)
        self.screen.blit(title, title.get_rect(center=(WIN_W // 2, 105)))
        sub = self.f_norm.render("Caro 20x20  •  Thắng khi đạt 5 quân liên tiếp", True, MUTED)
        self.screen.blit(sub, sub.get_rect(center=(WIN_W // 2, 150)))

        mouse = pygame.mouse.get_pos()
        for btn in self.menu_buttons().values():
            btn.draw(self.screen, self.f_norm, mouse)

        hint = self.f_small.render("Bấm vào nút để đổi tuỳ chọn  |  ESC: thoát", True, MUTED)
        self.screen.blit(hint, hint.get_rect(center=(WIN_W // 2, WIN_H - 32)))

    def draw_panel(self):
        b = self.board
        x0 = BOARD_PX + 2 * MARGIN
        pygame.draw.rect(self.screen, PANEL_BG, (x0, 0, PANEL_W, WIN_H))
        pygame.draw.line(self.screen, GRID_LIGHT, (x0, 0), (x0, WIN_H), 2)

        x = x0 + 20
        y = 26

        def line(text, font, color=TEXT, dy=30):
            nonlocal y
            self.screen.blit(font.render(text, True, color), (x, y))
            y += dy

        line("TIC-TAC-TOE AI", self.f_big, X_MAIN, dy=42)

        card_rect = pygame.Rect(x, y, PANEL_W - 40, 74)
        pygame.draw.rect(self.screen, CARD_BG, card_rect, border_radius=12)
        pygame.draw.rect(self.screen, GRID_LIGHT, card_rect, width=1, border_radius=12)

        def pname(p):
            return "Người chơi" if self.players[p] == "human" else self.players[p]

        self.screen.blit(self.f_norm.render(f"X:  {pname(X)}", True, X_MAIN), (x + 14, y + 12))
        self.screen.blit(self.f_norm.render(f"O:  {pname(O)}", True, O_MAIN), (x + 14, y + 40))
        y += 94

        if b.winner:
            sym = "X" if b.winner == X else "O"
            col = X_MAIN if b.winner == X else O_MAIN
            line(f"★  {sym} CHIẾN THẮNG!", self.f_big, col, 38)
        elif b.is_full():
            line("HÒA CỜ!", self.f_big, LAST_COLOR, 38)
        else:
            sym = "X" if b.current_player == X else "O"
            col = X_MAIN if b.current_player == X else O_MAIN
            line(f"Lượt đi:  {sym}", self.f_big, col, 34)
            if self.paused:
                line("⏸ Đang tạm dừng", self.f_norm, LAST_COLOR, 28)
            elif self.thinking:
                line("⚡ AI đang tính toán...", self.f_norm, LAST_COLOR, 28)
            elif self.is_human_turn():
                line("Click vào ô để đánh", self.f_norm, MUTED, 28)
            else:
                y += 28

        y += 14
        line(f"Số nước đã đi: {len(b.move_history)}", self.f_norm, TEXT, 28)
        if self.last_ai_label:
            line(f"AI vừa đi: {self.last_ai_label}", self.f_small, MUTED, 22)
            line(f"Thời gian: {self.last_ai_time:.3f}s", self.f_small, MUTED, 22)
            nodes = "-" if self.last_ai_nodes is None else f"{self.last_ai_nodes:,}"
            line(f"Node đã duyệt: {nodes}", self.f_small, MUTED, 22)

        if self.message:
            y += 12
            cur = ""
            for ch in self.message:
                if self.f_small.size(cur + ch)[0] > PANEL_W - 40:
                    line(cur, self.f_small, O_MAIN, 20)
                    cur = ch
                else:
                    cur += ch
            line(cur, self.f_small, O_MAIN, 20)

        hints = [
            ("R", "Ván mới"),
            ("M", "Về menu"),
            ("SPACE", "Tạm dừng"),
            ("ESC", "Thoát"),
        ]
        yy = WIN_H - 28 * len(hints) - 16
        for key_txt, desc in hints:
            badge = pygame.Rect(x, yy, 64, 22)
            pygame.draw.rect(self.screen, CARD_BG, badge, border_radius=6)
            pygame.draw.rect(self.screen, GRID_LIGHT, badge, width=1, border_radius=6)
            ks = self.f_small.render(key_txt, True, X_MAIN)
            self.screen.blit(ks, ks.get_rect(center=badge.center))
            self.screen.blit(self.f_small.render(desc, True, MUTED), (x + 74, yy + 2))
            yy += 28

    def draw_game(self):
        self.draw_board()
        self.draw_panel()


def run_gui():
    GameApp().run()


if __name__ == "__main__":
    run_gui()