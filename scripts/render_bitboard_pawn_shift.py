#!/usr/bin/env python3
"""Render a Shorts-ready bitboard pawn-shift walkthrough."""

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
DEFAULT_OUTPUT = SHORTS_DIR / "bitboard_pawn_shift.mp4"
DEFAULT_THUMBNAIL = SHORTS_DIR / "bitboard_pawn_shift_thumbnail.png"

BOARD_FILES = "abcdefgh"
WHITE_PAWNS = ("a2", "b2", "c4", "d2", "e2", "f3", "g2", "h2")
BLACK_PIECES = ("a3", "b5", "e3", "g3", "g4")

LIGHT = (220, 207, 178)
DARK = (92, 116, 92)
BOARD_EDGE = (52, 60, 72)
WHITE_PIECE = (245, 242, 232)
BLACK_PIECE = (37, 42, 52)
PAWN_BLUE = (86, 176, 255)
TARGET_GREEN = (126, 220, 135)
CAPTURE_RED = (255, 104, 91)
SHIFT_YELLOW = (255, 202, 77)
MASK_TEAL = (45, 208, 184)


@dataclass(frozen=True)
class FrameState:
    phase: str
    progress: float
    audio_event: str | None = None


def square_index(name: str) -> int:
    file_index = BOARD_FILES.index(name[0])
    rank_index = int(name[1]) - 1
    return rank_index * 8 + file_index


def bitboard(squares: tuple[str, ...]) -> int:
    value = 0
    for square in squares:
        value |= 1 << square_index(square)
    return value


def shift_north(bb: int) -> int:
    return (bb << 8) & ((1 << 64) - 1)


def shift_north_east(bb: int) -> int:
    file_h = 0x8080808080808080
    return ((bb & ~file_h) << 9) & ((1 << 64) - 1)


def shift_north_west(bb: int) -> int:
    file_a = 0x0101010101010101
    return ((bb & ~file_a) << 7) & ((1 << 64) - 1)


WHITE_BB = bitboard(WHITE_PAWNS)
BLACK_BB = bitboard(BLACK_PIECES)
OCCUPIED_BB = WHITE_BB | BLACK_BB
EMPTY_BB = (~OCCUPIED_BB) & ((1 << 64) - 1)
PUSHED_BB = shift_north(WHITE_BB)
LEGAL_PUSH_BB = PUSHED_BB & EMPTY_BB
ATTACK_BB = shift_north_east(WHITE_BB) | shift_north_west(WHITE_BB)
CAPTURE_BB = ATTACK_BB & BLACK_BB


PHASES: tuple[tuple[str, float, str | None], ...] = (
    ("intro", 2.0, "lock"),
    ("board", 4.5, None),
    ("bitboard", 5.0, "swap"),
    ("shift", 6.0, "swap"),
    ("empty", 5.5, "lock"),
    ("and", 5.5, "swap"),
    ("result", 5.0, "lock"),
    ("captures", 6.0, "swap"),
    ("stockfish", 5.0, "sorted"),
)


def make_timeline(fps: int) -> list[FrameState]:
    frames: list[FrameState] = []
    for phase, seconds, event in PHASES:
        count = int(seconds * fps)
        for index in range(count):
            progress = index / max(1, count - 1)
            frames.append(FrameState(phase, progress, event if index == 0 else None))
    return frames


def active_mask_for_phase(phase: str) -> int:
    if phase in {"intro", "board", "bitboard"}:
        return WHITE_BB
    if phase == "shift":
        return PUSHED_BB
    if phase == "empty":
        return EMPTY_BB
    if phase in {"and", "result"}:
        return LEGAL_PUSH_BB
    if phase == "captures":
        return CAPTURE_BB
    return WHITE_BB | LEGAL_PUSH_BB | CAPTURE_BB


def mask_info_for_phase(phase: str) -> tuple[str, int, tuple[int, int, int]]:
    if phase in {"intro", "board", "bitboard"}:
        return "whitePawns", WHITE_BB, PAWN_BLUE
    if phase == "shift":
        return "pushed", PUSHED_BB, SHIFT_YELLOW
    if phase == "empty":
        return "emptySquares", EMPTY_BB, MASK_TEAL
    if phase in {"and", "result"}:
        return "legalPushes", LEGAL_PUSH_BB, TARGET_GREEN
    if phase == "captures":
        return "captures", CAPTURE_BB, CAPTURE_RED
    return "moveTargets", WHITE_BB | LEGAL_PUSH_BB | CAPTURE_BB, TARGET_GREEN


def compact_number(value: int) -> str:
    if value >= 1_000_000_000_000_000_000:
        return f"{value / 1_000_000_000_000_000_000:.1f} quint."
    if value >= 1_000_000_000_000_000:
        return f"{value / 1_000_000_000_000_000:.1f} quad."
    if value >= 1_000_000_000_000:
        return f"{value / 1_000_000_000_000:.2f}T"
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}B"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if value >= 1_000:
        return f"{value / 1_000:.1f}K"
    return str(value)


def square_to_rc(index: int) -> tuple[int, int]:
    rank = index // 8
    file_index = index % 8
    return 7 - rank, file_index


def rc_to_square(row: int, col: int) -> str:
    rank = 8 - row
    return f"{BOARD_FILES[col]}{rank}"


def board_cell_bounds(left: int, top: int, size: int, square: str) -> tuple[int, int, int, int]:
    file_index = BOARD_FILES.index(square[0])
    rank_index = int(square[1]) - 1
    row = 7 - rank_index
    col = file_index
    x0 = left + col * size
    y0 = top + row * size
    return x0, y0, x0 + size, y0 + size


def draw_title(draw: ImageDraw.ImageDraw, width: int, state: FrameState) -> None:
    title = "CHESS BITBOARDS"
    title_font = bubble.fit_font(draw, title, 72, int(width * 0.9), bold=True, min_size=46)
    draw.text(((width - bubble.text_width(draw, title, title_font)) / 2, 70), title, font=title_font, fill=bubble.TEXT)

    subtitle_by_phase = {
        "intro": "The secret behind ultra-fast move generation in Stockfish.",
        "board": "The secret behind ultra-fast move generation in Stockfish.",
        "bitboard": "A 1 means: there is a white pawn on this square.",
        "shift": "White pawn pushes are a bit shift: pawns << 8.",
        "empty": "But only empty destination squares are legal pushes.",
        "and": "AND keeps only squares that pass both tests.",
        "result": "All one-square pawn pushes appear at once.",
        "captures": "Captures use diagonal shifts, then mask enemy pieces.",
        "stockfish": "This is the core bitboard trick behind fast move generation.",
    }
    subtitle = subtitle_by_phase[state.phase]
    subtitle_font = bubble.fit_font(draw, subtitle, 31, int(width * 0.88), min_size=24)
    bubble.draw_centered_text(draw, 158, subtitle, subtitle_font, bubble.MUTED, width)


def draw_chessboard(draw: ImageDraw.ImageDraw, left: int, top: int, size: int, state: FrameState) -> None:
    board_size = size * 8
    bubble.rounded_rect(draw, (left - 14, top - 14, left + board_size + 14, top + board_size + 14), 18, (14, 18, 28), BOARD_EDGE, 2)

    for row in range(8):
        for col in range(8):
            x0 = left + col * size
            y0 = top + row * size
            fill = LIGHT if (row + col) % 2 == 0 else DARK
            draw.rectangle((x0, y0, x0 + size, y0 + size), fill=fill)
            if row == 7:
                draw.text((x0 + 8, y0 + size - 25), BOARD_FILES[col], font=bubble.font(17, bold=True), fill=(35, 42, 48))
            if col == 0:
                draw.text((x0 + 7, y0 + 6), str(8 - row), font=bubble.font(17, bold=True), fill=(35, 42, 48))

    for square in BLACK_PIECES:
        x0, y0, x1, y1 = board_cell_bounds(left, top, size, square)
        draw_pawn(draw, x0 + size / 2, y0 + size / 2, size, BLACK_PIECE, (10, 13, 20), bubble.TEXT)

    for square in WHITE_PAWNS:
        x0, y0, x1, y1 = board_cell_bounds(left, top, size, square)
        draw_pawn(draw, x0 + size / 2, y0 + size / 2, size, WHITE_PIECE, PAWN_BLUE, (31, 42, 55))

    if state.phase in {"result", "stockfish"}:
        draw_targets(draw, left, top, size, LEGAL_PUSH_BB, TARGET_GREEN)
    if state.phase in {"captures", "stockfish"}:
        draw_targets(draw, left, top, size, CAPTURE_BB, CAPTURE_RED)


def draw_pawn(
    draw: ImageDraw.ImageDraw,
    cx: float,
    cy: float,
    cell_size: int,
    fill: tuple[int, int, int],
    outline: tuple[int, int, int],
    accent: tuple[int, int, int],
) -> None:
    unit = cell_size / 80
    shadow = (0, 0, 0)
    shadow_offset = 3 * unit

    def scaled(box: tuple[float, float, float, float], dx: float = 0, dy: float = 0) -> tuple[float, float, float, float]:
        return (
            cx + box[0] * unit + dx,
            cy + box[1] * unit + dy,
            cx + box[2] * unit + dx,
            cy + box[3] * unit + dy,
        )

    # Shadow first, then a simple pawn silhouette: head, collar, body, and base.
    for box, radius in (
        ((-14, -31, 14, -3), None),
        ((-19, -6, 19, 8), 7),
        ((-23, 4, 23, 29), 11),
        ((-30, 24, 30, 36), 6),
    ):
        if radius is None:
            draw.ellipse(scaled(box, shadow_offset, shadow_offset), fill=shadow)
        else:
            draw.rounded_rectangle(scaled(box, shadow_offset, shadow_offset), radius=int(radius * unit), fill=shadow)

    draw.ellipse(scaled((-14, -31, 14, -3)), fill=fill, outline=outline, width=max(2, int(3 * unit)))
    draw.rounded_rectangle(scaled((-19, -6, 19, 8)), radius=int(7 * unit), fill=fill, outline=outline, width=max(2, int(3 * unit)))
    draw.rounded_rectangle(scaled((-23, 4, 23, 29)), radius=int(11 * unit), fill=fill, outline=outline, width=max(2, int(3 * unit)))
    draw.rounded_rectangle(scaled((-30, 24, 30, 36)), radius=int(6 * unit), fill=fill, outline=outline, width=max(2, int(3 * unit)))
    draw.arc(scaled((-17, -28, 17, 6)), 205, 335, fill=accent, width=max(2, int(3 * unit)))


def draw_targets(draw: ImageDraw.ImageDraw, left: int, top: int, size: int, mask: int, color: tuple[int, int, int]) -> None:
    for index in range(64):
        if not (mask & (1 << index)):
            continue
        row, col = square_to_rc(index)
        cx = left + col * size + size / 2
        cy = top + row * size + size / 2
        draw.ellipse((cx - 17, cy - 17, cx + 17, cy + 17), fill=color, outline=(255, 255, 255), width=2)


def draw_shift_arrows(draw: ImageDraw.ImageDraw, left: int, top: int, size: int, progress: float) -> None:
    if progress <= 0:
        return
    for square in WHITE_PAWNS:
        from_index = square_index(square)
        to_index = from_index + 8
        if to_index >= 64:
            continue
        from_row, from_col = square_to_rc(from_index)
        to_row, to_col = square_to_rc(to_index)
        x0 = left + from_col * size + size / 2
        y0 = top + from_row * size + size / 2
        x1 = left + to_col * size + size / 2
        y1 = top + to_row * size + size / 2
        x = bubble.lerp(x0, x1, min(1.0, progress * 1.15))
        y = bubble.lerp(y0, y1, min(1.0, progress * 1.15))
        draw.line((x0, y0, x, y), fill=SHIFT_YELLOW, width=5)
        draw.ellipse((x - 8, y - 8, x + 8, y + 8), fill=SHIFT_YELLOW)


def draw_bit_grid(draw: ImageDraw.ImageDraw, left: int, top: int, size: int, state: FrameState) -> None:
    mask = active_mask_for_phase(state.phase)
    label_font = bubble.font(19, bold=True)
    bit_font = bubble.font(24, bold=True)
    board_size = size * 8
    bubble.rounded_rect(draw, (left - 14, top - 14, left + board_size + 14, top + board_size + 14), 18, (14, 18, 28), BOARD_EDGE, 2)

    for row in range(8):
        for col in range(8):
            square = rc_to_square(row, col)
            index = square_index(square)
            bit = bool(mask & (1 << index))
            x0 = left + col * size
            y0 = top + row * size
            base = (27, 33, 47)
            if bit and state.phase in {"empty"}:
                base = (31, 77, 73)
            elif bit and state.phase in {"and", "result", "stockfish"}:
                base = (39, 85, 53)
            elif bit and state.phase == "captures":
                base = (88, 43, 44)
            elif bit:
                base = (40, 70, 98)
            draw.rounded_rectangle((x0 + 3, y0 + 3, x0 + size - 3, y0 + size - 3), radius=8, fill=base, outline=(53, 64, 84), width=1)
            color = bubble.TEXT if bit else (82, 92, 112)
            value = "1" if bit else "0"
            draw.text((x0 + size / 2 - bubble.text_width(draw, value, bit_font) / 2, y0 + 21), value, font=bit_font, fill=color)
            draw.text((x0 + 8, y0 + size - 23), str(index), font=label_font, fill=(120, 130, 148))

    if state.phase == "shift":
        draw_shift_arrows(draw, left, top, size, state.progress)


def draw_integer_sidecar(
    draw: ImageDraw.ImageDraw,
    x0: int,
    y0: int,
    value: int,
    color: tuple[int, int, int],
) -> None:
    bubble.rounded_rect(draw, (x0, y0, x0 + 245, y0 + 160), 16, (24, 29, 41), color, 2)
    label_font = bubble.font(21, bold=True)
    value_font = bubble.font(29, bold=True)
    note_font = bubble.font(19)
    draw.text((x0 + 16, y0 + 16), "same 64 bits", font=label_font, fill=color)
    draw.text((x0 + 16, y0 + 50), "if read as number:", font=note_font, fill=bubble.MUTED)
    draw.text((x0 + 16, y0 + 82), f"{value:,}", font=value_font, fill=bubble.TEXT)
    draw.text((x0 + 16, y0 + 124), "engine uses bits", font=note_font, fill=bubble.MUTED)


def draw_equation(draw: ImageDraw.ImageDraw, width: int, y: int, state: FrameState) -> None:
    margin = int(width * 0.065)
    panel_h = 245
    bubble.rounded_rect(draw, (margin, y, width - margin, y + panel_h), 18, (15, 19, 29), bubble.GRID, 2)
    x = margin + 28
    title_font = bubble.font(44, bold=True)
    body_font = bubble.font(30)
    small_font = bubble.font(25)

    if state.phase in {"intro", "board", "bitboard"}:
        title = "Bitboard = one 64-bit mask"
        lines = [
            "normal view: one ordinary integer value",
            "engine trick: each bit becomes a square",
        ]
    elif state.phase == "shift":
        title = "pushed = whitePawns << 8"
        lines = [
            "NORTH is +8 because each rank has 8 files",
            "this moves every pawn candidate at the same time",
        ]
    elif state.phase == "empty":
        title = "emptySquares = NOT occupied"
        lines = [
            "occupied = white pieces OR black pieces",
            "blocked forward squares must disappear",
        ]
    elif state.phase in {"and", "result"}:
        title = "legalPushes = pushed & emptySquares"
        lines = [
            "AND keeps squares that are both pushed-to and empty",
            f"legal one-step pushes found: {LEGAL_PUSH_BB.bit_count()}",
        ]
    elif state.phase == "captures":
        title = "captures = diagonalAttacks & enemies"
        lines = [
            "diagonalAttacks = (pawns << 7) OR (pawns << 9)",
            f"capture targets found: {CAPTURE_BB.bit_count()}",
        ]
    else:
        title = "Stockfish uses this pattern everywhere"
        lines = [
            "shift masks for pawns, attack masks for pieces",
            "then AND with targets to generate moves",
        ]

    draw.text((x, y + 26), title, font=title_font, fill=SHIFT_YELLOW)
    for line_index, line in enumerate(lines):
        draw.text((x, y + 92 + line_index * 42), line, font=body_font, fill=bubble.TEXT if line_index == 0 else bubble.MUTED)

    if state.phase in {"intro", "board", "bitboard"}:
        _, value, color = mask_info_for_phase(state.phase)
        draw_integer_sidecar(draw, width - margin - 275, y + 42, value, color)

    if state.phase in {"bitboard", "shift", "empty", "and", "result", "captures"}:
        name, value, color = mask_info_for_phase(state.phase)
        text = f"{name} if read as a number: {value:,}"
        draw.text((x, y + 182), text, font=small_font, fill=color)


def draw_mini_masks(draw: ImageDraw.ImageDraw, width: int, y: int, state: FrameState) -> None:
    margin = int(width * 0.065)
    cards = [
        ("whitePawns", WHITE_BB, PAWN_BLUE),
        ("pushed", PUSHED_BB, SHIFT_YELLOW),
        ("empty", EMPTY_BB, MASK_TEAL),
        ("result", LEGAL_PUSH_BB, TARGET_GREEN),
    ]
    if state.phase == "captures":
        cards = [
            ("whitePawns", WHITE_BB, PAWN_BLUE),
            ("diagonals", ATTACK_BB, SHIFT_YELLOW),
            ("enemies", BLACK_BB, CAPTURE_RED),
            ("captures", CAPTURE_BB, CAPTURE_RED),
        ]
    card_w = (width - 2 * margin - 42) / 4
    font = bubble.font(22, bold=True)
    value_font = bubble.font(19)
    for index, (label, value, color) in enumerate(cards):
        x0 = margin + index * (card_w + 14)
        bubble.rounded_rect(draw, (x0, y, x0 + card_w, y + 112), 16, (24, 29, 41), color, 2)
        draw.text((x0 + 16, y + 15), label, font=font, fill=color)
        draw.text((x0 + 16, y + 58), f"{value.bit_count()} bits set", font=value_font, fill=bubble.TEXT)
        draw.text((x0 + 16, y + 84), f"as number: {compact_number(value)}", font=value_font, fill=bubble.MUTED)


def draw_stockfish_note(draw: ImageDraw.ImageDraw, width: int, y: int) -> None:
    margin = int(width * 0.065)
    font = bubble.font(28)
    code_font = bubble.font(30, bold=True)
    bubble.rounded_rect(draw, (margin, y, width - margin, y + 180), 18, (15, 19, 29), bubble.GRID, 2)
    draw.text((margin + 26, y + 24), "Same shape as Stockfish pawn move generation:", font=font, fill=bubble.MUTED)
    draw.text((margin + 26, y + 74), "b1 = shift<Up>(pawns) & emptySquares", font=code_font, fill=TARGET_GREEN)
    draw.text((margin + 26, y + 122), "moveList = splat_pawn_moves(..., b1)", font=code_font, fill=SHIFT_YELLOW)


def draw_frame(width: int, height: int, state: FrameState, frame_number: int, total_frames: int) -> Image.Image:
    image = bubble.gradient_background(width, height).copy()
    draw = ImageDraw.Draw(image)
    draw_title(draw, width, state)

    board_size = 80
    left = int((width - board_size * 8) / 2)
    top = 275
    if state.phase in {"intro", "board"}:
        draw_chessboard(draw, left, top, board_size, state)
    elif state.phase == "stockfish":
        draw_chessboard(draw, left, top, board_size, state)
    else:
        draw_bit_grid(draw, left, top, board_size, state)

    draw_equation(draw, width, 965, state)
    draw_mini_masks(draw, width, 1240, state)
    if state.phase == "stockfish":
        draw_stockfish_note(draw, width, 1385)
    else:
        caption_font = bubble.font(28, bold=True)
        caption = "8 x 8 board = 64 bits; bitwise ops become board geometry"
        bubble.draw_centered_text(draw, 1418, caption, caption_font, bubble.TEXT, width)

    footer_font = bubble.font(22)
    footer = f"frame {frame_number + 1}/{total_frames} | bitboards turn board geometry into integer operations"
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
                if not thumbnail_saved and frame_state.phase == "and" and frame_state.progress > 0.35:
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
    print(f"White pawns: 0x{WHITE_BB:016X}")
    print(f"Legal pushes: {LEGAL_PUSH_BB.bit_count()} targets, 0x{LEGAL_PUSH_BB:016X}")
    print(f"Captures: {CAPTURE_BB.bit_count()} targets, 0x{CAPTURE_BB:016X}")


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
