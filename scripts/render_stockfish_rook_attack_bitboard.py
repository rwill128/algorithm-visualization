#!/usr/bin/env python3
"""Render a Shorts-ready Stockfish rook attack bitboard walkthrough."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw

import render_bubble_sort_short as bubble


ROOT = Path(__file__).resolve().parents[1]
SHORTS_DIR = ROOT / "artifacts" / "shorts"
DEFAULT_OUTPUT = SHORTS_DIR / "stockfish_rook_attack_bitboard.mp4"
DEFAULT_THUMBNAIL = SHORTS_DIR / "stockfish_rook_attack_bitboard_thumbnail.png"

FILES = "abcdefgh"
ROOK_SQUARE = "d4"
FRIENDLY_BLOCKERS = ("d6", "f4", "g2")
ENEMY_BLOCKERS = ("b4", "d2", "h4", "a7")

LIGHT = (220, 207, 178)
DARK = (92, 116, 92)
BOARD_EDGE = (52, 60, 72)
BLUE = (86, 176, 255)
GREEN = (126, 220, 135)
YELLOW = (255, 202, 77)
RED = (255, 104, 91)
PURPLE = (176, 134, 255)
TEAL = (45, 208, 184)
CARD = (24, 29, 41)
PANEL = (15, 19, 29)
LINE = (45, 55, 74)
FRIENDLY = (245, 242, 232)
ENEMY = (37, 42, 52)

COMPARISON_ROWS: tuple[tuple[str, int, str, int, str, tuple[int, int, int]], ...] = (
    ("Try every square", 64, "64 tests", 1, "64 destination tests, plus path checks", RED),
    ("Scan rays now", 14, "14 probes", 1, "walk north/south/east/west until blocked", YELLOW),
    ("Bitboard lookup", 3, "3 steps", 800, "mask + index + table load", GREEN),
)


@dataclass(frozen=True)
class FrameState:
    phase: str
    progress: float
    audio_event: str | None = None


def square_index(square: str) -> int:
    file_index = FILES.index(square[0])
    rank_index = int(square[1]) - 1
    return rank_index * 8 + file_index


def bit(square: str) -> int:
    return 1 << square_index(square)


def bitboard(squares: tuple[str, ...]) -> int:
    value = 0
    for square in squares:
        value |= bit(square)
    return value


def square_to_rc(index: int) -> tuple[int, int]:
    rank = index // 8
    file_index = index % 8
    return 7 - rank, file_index


def rc_to_square(row: int, col: int) -> str:
    return f"{FILES[col]}{8 - row}"


def rook_rays(square: str) -> int:
    idx = square_index(square)
    row, col = square_to_rc(idx)
    mask = 0
    for dr, dc in ((-1, 0), (1, 0), (0, 1), (0, -1)):
        rr, cc = row + dr, col + dc
        while 0 <= rr < 8 and 0 <= cc < 8:
            mask |= bit(rc_to_square(rr, cc))
            rr += dr
            cc += dc
    return mask


def rook_attacks(square: str, occupied: int) -> int:
    idx = square_index(square)
    row, col = square_to_rc(idx)
    attacks = 0
    for dr, dc in ((-1, 0), (1, 0), (0, 1), (0, -1)):
        rr, cc = row + dr, col + dc
        while 0 <= rr < 8 and 0 <= cc < 8:
            sq = rc_to_square(rr, cc)
            sq_bit = bit(sq)
            attacks |= sq_bit
            if occupied & sq_bit:
                break
            rr += dr
            cc += dc
    return attacks


ROOK_BB = bit(ROOK_SQUARE)
FRIENDLY_BB = ROOK_BB | bitboard(FRIENDLY_BLOCKERS)
ENEMY_BB = bitboard(ENEMY_BLOCKERS)
OCCUPIED_BB = FRIENDLY_BB | ENEMY_BB
EMPTY_RAYS_BB = rook_rays(ROOK_SQUARE)
RELEVANT_OCCUPANCY_BB = OCCUPIED_BB & EMPTY_RAYS_BB
RAW_ATTACKS_BB = rook_attacks(ROOK_SQUARE, OCCUPIED_BB)
QUIET_TARGETS_BB = RAW_ATTACKS_BB & ~OCCUPIED_BB
CAPTURE_TARGETS_BB = RAW_ATTACKS_BB & ENEMY_BB
LEGAL_TARGETS_BB = RAW_ATTACKS_BB & ~FRIENDLY_BB

PHASES: tuple[tuple[str, float, str | None], ...] = (
    ("intro", 2.3, "lock"),
    ("empty", 4.4, "swap"),
    ("occupied", 4.8, "swap"),
    ("stop", 5.2, "lock"),
    ("lookup", 5.0, "swap"),
    ("attacks", 5.2, "lock"),
    ("legal", 5.2, "swap"),
    ("stockfish", 5.5, "sorted"),
    ("comparison", 6.4, "sorted"),
)


def make_timeline(fps: int) -> list[FrameState]:
    frames: list[FrameState] = []
    for phase, seconds, event in PHASES:
        count = int(seconds * fps)
        for index in range(count):
            frames.append(FrameState(phase, index / max(1, count - 1), event if index == 0 else None))
    return frames


def active_overlay(state: FrameState) -> tuple[int, tuple[int, int, int], str]:
    if state.phase == "empty":
        return EMPTY_RAYS_BB, BLUE, "empty-board rook rays"
    if state.phase == "occupied":
        return RELEVANT_OCCUPANCY_BB, YELLOW, "relevant occupied squares"
    if state.phase == "stop":
        return RAW_ATTACKS_BB, YELLOW, "rays stop at first blocker"
    if state.phase == "lookup":
        return RELEVANT_OCCUPANCY_BB, PURPLE, "occupied & rookMask"
    if state.phase == "attacks":
        return RAW_ATTACKS_BB, GREEN, "attack mask"
    if state.phase in {"legal", "stockfish"}:
        return LEGAL_TARGETS_BB, GREEN, "legal targets"
    return 0, BLUE, ""


def board_cell(left: int, top: int, size: int, square: str) -> tuple[int, int, int, int]:
    row, col = square_to_rc(square_index(square))
    x0 = left + col * size
    y0 = top + row * size
    return x0, y0, x0 + size, y0 + size


def draw_rook(draw: ImageDraw.ImageDraw, cx: float, cy: float, size: int, fill: tuple[int, int, int], outline: tuple[int, int, int]) -> None:
    unit = size / 80
    def box(x0: float, y0: float, x1: float, y1: float) -> tuple[float, float, float, float]:
        return cx + x0 * unit, cy + y0 * unit, cx + x1 * unit, cy + y1 * unit

    shadow = (0, 0, 0)
    for b in [(-27, 18, 27, 33), (-21, -10, 21, 22), (-28, -28, 28, -10)]:
        draw.rounded_rectangle(box(*b), radius=int(5 * unit), fill=shadow)
    for b in [(-27, 18, 27, 33), (-21, -10, 21, 22), (-28, -28, 28, -10)]:
        draw.rounded_rectangle(box(*b), radius=int(5 * unit), fill=fill, outline=outline, width=max(2, int(3 * unit)))
    for x in (-22, -4, 14):
        draw.rectangle(box(x, -39, x + 12, -22), fill=fill, outline=outline, width=max(2, int(2 * unit)))


def draw_disc_piece(draw: ImageDraw.ImageDraw, cx: float, cy: float, size: int, fill: tuple[int, int, int], outline: tuple[int, int, int], label: str) -> None:
    radius = size * 0.28
    draw.ellipse((cx - radius + 3, cy - radius + 3, cx + radius + 3, cy + radius + 3), fill=(0, 0, 0))
    draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=fill, outline=outline, width=3)
    font = bubble.font(int(size * 0.22), bold=True)
    draw.text((cx - bubble.text_width(draw, label, font) / 2, cy - size * 0.13), label, font=font, fill=outline)


def draw_board(draw: ImageDraw.ImageDraw, left: int, top: int, size: int, state: FrameState) -> None:
    board_size = size * 8
    bubble.rounded_rect(draw, (left - 14, top - 14, left + board_size + 14, top + board_size + 14), 18, (14, 18, 28), BOARD_EDGE, 2)

    mask, overlay_color, _ = active_overlay(state)
    for row in range(8):
        for col in range(8):
            square = rc_to_square(row, col)
            x0 = left + col * size
            y0 = top + row * size
            fill = LIGHT if (row + col) % 2 == 0 else DARK
            draw.rectangle((x0, y0, x0 + size, y0 + size), fill=fill)
            sq_bit = bit(square)
            if mask & sq_bit:
                inset = 8
                draw.rounded_rectangle(
                    (x0 + inset, y0 + inset, x0 + size - inset, y0 + size - inset),
                    radius=12,
                    fill=overlay_color,
                    outline=(255, 255, 255),
                    width=2,
                )
            if state.phase in {"legal", "stockfish"} and CAPTURE_TARGETS_BB & sq_bit:
                draw.rounded_rectangle((x0 + 6, y0 + 6, x0 + size - 6, y0 + size - 6), radius=12, outline=RED, width=5)
            if row == 7:
                draw.text((x0 + 7, y0 + size - 24), FILES[col], font=bubble.font(16, bold=True), fill=(35, 42, 48))
            if col == 0:
                draw.text((x0 + 7, y0 + 6), str(8 - row), font=bubble.font(16, bold=True), fill=(35, 42, 48))

    for square in FRIENDLY_BLOCKERS:
        x0, y0, x1, y1 = board_cell(left, top, size, square)
        draw_disc_piece(draw, (x0 + x1) / 2, (y0 + y1) / 2, size, FRIENDLY, BLUE, "own")
    for square in ENEMY_BLOCKERS:
        x0, y0, x1, y1 = board_cell(left, top, size, square)
        draw_disc_piece(draw, (x0 + x1) / 2, (y0 + y1) / 2, size, ENEMY, RED, "enemy")

    x0, y0, x1, y1 = board_cell(left, top, size, ROOK_SQUARE)
    draw_rook(draw, (x0 + x1) / 2, (y0 + y1) / 2, size, FRIENDLY, BLUE)


def draw_bit_grid(draw: ImageDraw.ImageDraw, x: int, y: int, size: int, mask: int, color: tuple[int, int, int]) -> None:
    bubble.rounded_rect(draw, (x - 10, y - 10, x + size * 8 + 10, y + size * 8 + 10), 14, (14, 18, 28), LINE, 2)
    bit_font = bubble.font(int(size * 0.38), bold=True)
    for row in range(8):
        for col in range(8):
            square = rc_to_square(row, col)
            sq_bit = bit(square)
            x0 = x + col * size
            y0 = y + row * size
            on = bool(mask & sq_bit)
            fill = (36, 48, 55) if on else (25, 30, 42)
            if on:
                fill = tuple(int((fill[i] + color[i]) / 2) for i in range(3))
            draw.rounded_rectangle((x0 + 3, y0 + 3, x0 + size - 3, y0 + size - 3), radius=6, fill=fill, outline=(53, 64, 84), width=1)
            text = "1" if on else "0"
            draw.text((x0 + (size - bubble.text_width(draw, text, bit_font)) / 2, y0 + size * 0.22), text, font=bit_font, fill=bubble.TEXT if on else (84, 94, 114))


def draw_title(draw: ImageDraw.ImageDraw, width: int, state: FrameState) -> None:
    title = "STOCKFISH ROOK ATTACKS"
    title_font = bubble.fit_font(draw, title, 68, int(width * 0.9), bold=True, min_size=42)
    bubble.draw_centered_text(draw, 66, title, title_font, bubble.TEXT, width)
    subtitles = {
        "intro": "How a chess engine finds sliding moves fast.",
        "empty": "A rook can slide along ranks and files on an empty board.",
        "occupied": "Real positions add blockers: one occupied bitboard.",
        "stop": "Sliding attacks include the first blocker, then stop.",
        "lookup": "Stockfish turns the blocker pattern into an attack lookup.",
        "attacks": "The result is one attack bitboard for this rook.",
        "legal": "Then friendly pieces are removed from the target mask.",
        "stockfish": "Move generation becomes bit masks plus table lookups.",
        "comparison": "The speed comes from trading memory for less repeated work.",
    }
    subtitle = subtitles[state.phase]
    subtitle_font = bubble.fit_font(draw, subtitle, 31, int(width * 0.9), min_size=24)
    bubble.draw_centered_text(draw, 150, subtitle, subtitle_font, bubble.MUTED, width)


def draw_equation(draw: ImageDraw.ImageDraw, width: int, y: int, state: FrameState) -> None:
    margin = int(width * 0.065)
    bubble.rounded_rect(draw, (margin, y, width - margin, y + 260), 18, PANEL, bubble.GRID, 2)
    title_font = bubble.font(39, bold=True)
    body_font = bubble.font(28)
    code_font = bubble.font(29, bold=True)
    x = margin + 26

    if state.phase == "intro":
        title = "One 64-bit number can represent the whole board."
        lines = ["The next trick: sliding pieces depend on blockers.", "Rooks, bishops, and queens need occupancy-aware attacks."]
        code = "Bitboard occupied = pos.pieces()"
        color = BLUE
    elif state.phase == "empty":
        title = "rookMask = rank ray OR file ray"
        lines = ["Empty-board rays are easy, but they are too generous.", "They keep going through pieces that should block the rook."]
        code = f"from = {ROOK_SQUARE}"
        color = BLUE
    elif state.phase == "occupied":
        title = "occupied = all pieces on the board"
        lines = ["Only blockers on the rook's rank/file matter here.", f"Relevant blocker bits: {RELEVANT_OCCUPANCY_BB.bit_count()}"]
        code = "relevant = occupied & rookMask"
        color = YELLOW
    elif state.phase == "stop":
        title = "Attack rays stop at the first occupied square."
        lines = ["Enemy blockers are capture targets.", "Friendly blockers stop the ray but are not legal destinations."]
        code = "attacks = sliding_attack(ROOK, d4, occupied)"
        color = YELLOW
    elif state.phase == "lookup":
        title = "Magic bitboards avoid scanning every ray."
        lines = ["The blocker pattern indexes a precomputed attack mask.", "Same question, faster answer."]
        code = "attacks = table[index(occupied & mask)]"
        color = PURPLE
    elif state.phase == "attacks":
        title = "One bitboard answers: where can this rook attack?"
        lines = [f"Raw attack bits: {RAW_ATTACKS_BB.bit_count()}", "This mask still includes friendly blocker squares."]
        code = "attackMask = attacks_bb<ROOK>(from, occupied)"
        color = GREEN
    elif state.phase == "legal":
        title = "Legal targets remove friendly pieces."
        lines = [f"Quiet moves: {QUIET_TARGETS_BB.bit_count()}    Captures: {CAPTURE_TARGETS_BB.bit_count()}", f"Final target bits: {LEGAL_TARGETS_BB.bit_count()}"]
        code = "legal = attackMask & ~ownPieces"
        color = GREEN
    else:
        title = "This is the movegen hot path."
        lines = ["Stockfish loops over rooks, gets an attack bitboard,", "ANDs it with the target mask, then writes those moves."]
        code = "attacks_bb<ROOK>(from, pos.pieces()) & target"
        color = TEAL

    draw.text((x, y + 24), title, font=title_font, fill=color)
    for index, line in enumerate(lines):
        draw.text((x, y + 86 + index * 38), line, font=body_font, fill=bubble.TEXT if index == 0 else bubble.MUTED)
    bubble.rounded_rect(draw, (x, y + 177, width - margin - 26, y + 232), 13, (25, 31, 43), color, 2)
    draw.text((x + 18, y + 190), code, font=bubble.fit_font(draw, code, 29, width - 2 * margin - 88, bold=True, min_size=20), fill=color)


def draw_masks_panel(draw: ImageDraw.ImageDraw, width: int, y: int, state: FrameState) -> None:
    margin = int(width * 0.065)
    mask, color, label = active_overlay(state)
    card_w = (width - 2 * margin - 28) / 3
    cards = [
        ("occupied", OCCUPIED_BB, YELLOW),
        ("raw attacks", RAW_ATTACKS_BB, GREEN),
        ("legal targets", LEGAL_TARGETS_BB, TEAL),
    ]
    if state.phase in {"empty", "occupied", "lookup"}:
        cards = [
            ("rook rays", EMPTY_RAYS_BB, BLUE),
            ("relevant occupancy", RELEVANT_OCCUPANCY_BB, YELLOW),
            ("lookup result", RAW_ATTACKS_BB, GREEN),
        ]
    font = bubble.font(23, bold=True)
    small = bubble.font(20)
    for index, (name, value, card_color) in enumerate(cards):
        x0 = margin + index * (card_w + 14)
        active = value == mask or name in label
        outline = color if active else card_color
        bubble.rounded_rect(draw, (x0, y, x0 + card_w, y + 124), 16, CARD, outline, 2)
        draw.text((x0 + 16, y + 16), name, font=font, fill=outline)
        draw.text((x0 + 16, y + 56), f"{value.bit_count()} bits set", font=small, fill=bubble.TEXT)
        draw.text((x0 + 16, y + 86), f"0x{value:016X}"[:18], font=small, fill=bubble.MUTED)


def draw_source_note(draw: ImageDraw.ImageDraw, width: int, y: int) -> None:
    margin = int(width * 0.065)
    bubble.rounded_rect(draw, (margin, y, width - margin, y + 172), 18, PANEL, bubble.GRID, 2)
    title_font = bubble.font(27, bold=True)
    code_font = bubble.font(26, bold=True)
    draw.text((margin + 22, y + 20), "Stockfish shape:", font=title_font, fill=bubble.MUTED)
    draw.text((margin + 22, y + 66), "Bitboard b = Attacks::attacks_bb<Pt>(from, pos.pieces()) & target;", font=bubble.fit_font(draw, "Bitboard b = Attacks::attacks_bb<Pt>(from, pos.pieces()) & target;", 26, width - 2 * margin - 44, bold=True, min_size=17), fill=TEAL)
    draw.text((margin + 22, y + 116), "For rooks, Pt = ROOK. The output is a mask of destination squares.", font=code_font, fill=bubble.TEXT)


def draw_comparison(draw: ImageDraw.ImageDraw, width: int, height: int, state: FrameState) -> None:
    margin = int(width * 0.065)
    top = 245
    panel_bottom = height - 140
    bubble.rounded_rect(draw, (margin, top, width - margin, panel_bottom), 22, PANEL, bubble.GRID, 2)

    x = margin + 34
    y = top + 34
    title = "Three ways to answer the same question"
    title_font = bubble.fit_font(draw, title, 43, width - 2 * margin - 68, bold=True, min_size=32)
    body_font = bubble.font(27)
    small_font = bubble.font(22)
    label_font = bubble.font(25, bold=True)
    value_font = bubble.font(25, bold=True)
    draw.text((x, y), title, font=title_font, fill=TEAL)
    draw.text((x, y + 62), "Where can this rook move, given the occupied squares?", font=body_font, fill=bubble.TEXT)
    draw.text((x, y + 104), "Relative work per rook attack query. Lower CPU bars are better.", font=small_font, fill=bubble.MUTED)

    cpu_x = x
    mem_x = x + 520
    graph_y = y + 170
    row_h = 148
    cpu_bar_w = 350
    mem_bar_w = 250
    max_cpu = max(row[1] for row in COMPARISON_ROWS)
    max_mem = max(row[3] for row in COMPARISON_ROWS)

    draw.text((cpu_x, graph_y - 46), "CPU work", font=label_font, fill=bubble.TEXT)
    draw.text((mem_x, graph_y - 46), "Memory", font=label_font, fill=bubble.TEXT)
    for index, (name, cpu_units, cpu_label, memory_kb, note, color) in enumerate(COMPARISON_ROWS):
        row_y = graph_y + index * row_h
        visible = state.progress >= 0.18 + index * 0.18
        alpha = min(1.0, max(0.0, (state.progress - (0.18 + index * 0.18)) / 0.18))
        bar_scale = alpha if visible else 0.0

        draw.text((cpu_x, row_y), name, font=label_font, fill=color if visible else bubble.MUTED)
        draw.text((cpu_x, row_y + 34), note, font=small_font, fill=bubble.MUTED)

        cpu_y = row_y + 74
        bubble.rounded_rect(draw, (cpu_x, cpu_y, cpu_x + cpu_bar_w, cpu_y + 32), 12, (23, 28, 39), LINE, 1)
        cpu_fill = int(cpu_bar_w * (cpu_units / max_cpu) * bar_scale)
        if cpu_fill > 0:
            bubble.rounded_rect(draw, (cpu_x, cpu_y, cpu_x + cpu_fill, cpu_y + 32), 12, color, color, 1)
        draw.text((cpu_x + cpu_bar_w + 14, cpu_y + 1), cpu_label, font=value_font, fill=bubble.TEXT if visible else bubble.MUTED)

        mem_y = row_y + 74
        bubble.rounded_rect(draw, (mem_x, mem_y, mem_x + mem_bar_w, mem_y + 32), 12, (23, 28, 39), LINE, 1)
        mem_fill = int(mem_bar_w * (memory_kb / max_mem) * bar_scale)
        if mem_fill > 0:
            bubble.rounded_rect(draw, (mem_x, mem_y, mem_x + mem_fill, mem_y + 32), 12, color, color, 1)
        mem_label = "~800 KB" if memory_kb >= 800 else "tiny"
        draw.text((mem_x + mem_bar_w + 14, mem_y + 1), mem_label, font=value_font, fill=bubble.TEXT if visible else bubble.MUTED)

    callout_y = graph_y + len(COMPARISON_ROWS) * row_h + 22
    bubble.rounded_rect(draw, (x, callout_y, width - margin - 34, callout_y + 178), 18, (23, 28, 39), TEAL, 2)
    draw.text((x + 22, callout_y + 22), "The engine pays for a table once.", font=bubble.font(34, bold=True), fill=TEAL)
    draw.text((x + 22, callout_y + 74), "Then each sliding-piece query becomes a few bit operations", font=body_font, fill=bubble.TEXT)
    draw.text((x + 22, callout_y + 112), "plus one attack-mask lookup instead of repeated board walking.", font=body_font, fill=bubble.MUTED)


def draw_frame(width: int, height: int, state: FrameState, frame_number: int, total_frames: int) -> Image.Image:
    image = bubble.gradient_background(width, height).copy()
    draw = ImageDraw.Draw(image)
    draw_title(draw, width, state)

    if state.phase == "comparison":
        draw_comparison(draw, width, height, state)
        footer_font = bubble.font(22)
        footer = f"frame {frame_number + 1}/{total_frames} | occupancy-aware sliding attacks"
        draw.text((int(width * 0.065), int(height * 0.965)), footer, font=footer_font, fill=bubble.MUTED)
        return image

    board_size = 82
    board_left = int((width - board_size * 8) / 2)
    board_top = 245
    draw_board(draw, board_left, board_top, board_size, state)

    if state.phase in {"lookup", "attacks", "legal"}:
        mask, color, label = active_overlay(state)
        bit_x = int(width * 0.065)
        bit_y = 900
        bit_size = 31
        label_x = bit_x + bit_size * 8 + 34
        draw_bit_grid(draw, bit_x, bit_y, bit_size, mask, color)
        label_font = bubble.font(27, bold=True)
        draw.text((label_x, bit_y + 42), label, font=label_font, fill=color)
        draw.text((label_x, bit_y + 87), f"as 64-bit mask: 0x{mask:016X}", font=bubble.font(25, bold=True), fill=bubble.TEXT)
        draw.text((label_x, bit_y + 131), f"bits set: {mask.bit_count()}", font=bubble.font(25), fill=bubble.MUTED)
        equation_y = 1180
    else:
        equation_y = 930

    draw_equation(draw, width, equation_y, state)
    draw_masks_panel(draw, width, equation_y + 290, state)
    if state.phase == "stockfish":
        draw_source_note(draw, width, equation_y + 444)

    footer_font = bubble.font(22)
    footer = f"frame {frame_number + 1}/{total_frames} | occupancy-aware sliding attacks"
    draw.text((int(width * 0.065), int(height * 0.965)), footer, font=footer_font, fill=bubble.MUTED)
    return image


def render_video(args: argparse.Namespace) -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required but was not found on PATH")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    frames = make_timeline(args.fps)
    audio_enabled = not args.no_audio
    temp_context = tempfile.TemporaryDirectory() if audio_enabled else nullcontext(None)

    with temp_context as temp_dir:
        video_output = Path(temp_dir) / "silent.mp4" if audio_enabled else args.output
        command = [
            "ffmpeg",
            "-y",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{args.width}x{args.height}",
            "-r",
            str(args.fps),
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-crf",
            "18",
            str(video_output),
        ]
        process = subprocess.Popen(command, stdin=subprocess.PIPE)
        assert process.stdin is not None
        thumbnail_saved = False
        try:
            for frame_number, frame_state in enumerate(frames):
                image = draw_frame(args.width, args.height, frame_state, frame_number, len(frames))
                if not thumbnail_saved and frame_state.phase == "legal" and frame_state.progress > 0.25:
                    args.thumbnail.parent.mkdir(parents=True, exist_ok=True)
                    image.save(args.thumbnail)
                    thumbnail_saved = True
                process.stdin.write(image.tobytes())
        finally:
            process.stdin.close()

        return_code = process.wait()
        if return_code != 0:
            raise RuntimeError(f"ffmpeg exited with status {return_code}")

        if not thumbnail_saved:
            draw_frame(args.width, args.height, frames[-1], len(frames) - 1, len(frames)).save(args.thumbnail)

        if audio_enabled:
            audio_output = Path(temp_dir) / "operations.wav"
            bubble.generate_audio_track(frames, args.fps, audio_output)
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    str(video_output),
                    "-i",
                    str(audio_output),
                    "-c:v",
                    "copy",
                    "-c:a",
                    "aac",
                    "-b:a",
                    "192k",
                    "-shortest",
                    "-movflags",
                    "+faststart",
                    str(args.output),
                ],
                check=True,
            )

    print(f"Rendered {args.output}")
    print(f"Rendered {args.thumbnail}")
    print(f"Frames: {len(frames)} at {args.fps} fps ({len(frames) / args.fps:.1f}s)")
    print(f"Audio: {'on' if audio_enabled else 'off'}")
    print(f"Rook square: {ROOK_SQUARE}")
    print(f"Relevant blockers: {RELEVANT_OCCUPANCY_BB.bit_count()}, 0x{RELEVANT_OCCUPANCY_BB:016X}")
    print(f"Raw attacks: {RAW_ATTACKS_BB.bit_count()}, 0x{RAW_ATTACKS_BB:016X}")
    print(f"Legal targets: {LEGAL_TARGETS_BB.bit_count()}, 0x{LEGAL_TARGETS_BB:016X}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--thumbnail", type=Path, default=DEFAULT_THUMBNAIL)
    parser.add_argument("--width", type=int, default=1080)
    parser.add_argument("--height", type=int, default=1920)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--no-audio", action="store_true")
    args = parser.parse_args()
    if args.width <= 0 or args.height <= 0 or args.fps <= 0:
        parser.error("--width, --height, and --fps must be positive")
    return args


if __name__ == "__main__":
    render_video(parse_args())
