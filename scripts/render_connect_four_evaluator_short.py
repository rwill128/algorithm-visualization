#!/usr/bin/env python3
"""Render a Shorts-ready Connect Four board-evaluator animation."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw

import render_bubble_sort_short as bubble


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "artifacts" / "shorts" / "connect_four_evaluator.mp4"
DEFAULT_THUMBNAIL = ROOT / "artifacts" / "shorts" / "connect_four_evaluator_thumbnail.png"

ROWS = 6
COLS = 7
EMPTY = 0
AI = 1
OPPONENT = 2

AI_COLOR = (255, 204, 72)
OPPONENT_COLOR = (255, 96, 88)
EMPTY_COLOR = (13, 17, 25)
BOARD_BLUE = (38, 92, 178)
BOARD_EDGE = (72, 142, 232)
GHOST_COLOR = (255, 224, 113)
WIN_COLOR = bubble.SORTED
NEGATIVE = (255, 126, 106)


Board = tuple[tuple[int, ...], ...]
Cell = tuple[int, int]


@dataclass(frozen=True)
class WindowScore:
    cells: tuple[Cell, ...]
    score: int
    reason: str
    ai_count: int
    opponent_count: int
    empty_count: int


@dataclass(frozen=True)
class Candidate:
    column: int
    row: int
    board: Board
    score: int
    center_score: int
    windows: tuple[WindowScore, ...]


@dataclass(frozen=True)
class Step:
    label: str
    status: str
    board: Board
    column: int | None
    candidate: Candidate | None
    window: WindowScore | None
    scored_columns: tuple[tuple[int, int], ...]
    best_column: int | None
    best_score: int | None


@dataclass(frozen=True)
class FrameState:
    step: Step
    drop_progress: float | None = None
    audio_event: str | None = None


def make_board(moves: list[int]) -> Board:
    grid = [[EMPTY for _ in range(COLS)] for _ in range(ROWS)]
    player = AI
    for column in moves:
        for row in range(ROWS - 1, -1, -1):
            if grid[row][column] == EMPTY:
                grid[row][column] = player
                break
        else:
            raise ValueError(f"column {column} is full")
        player = OPPONENT if player == AI else AI
    board = tuple(tuple(row) for row in grid)
    validate_board(board)
    return board


def set_cell(board: Board, row: int, column: int, value: int) -> Board:
    mutable = [list(line) for line in board]
    mutable[row][column] = value
    next_board = tuple(tuple(line) for line in mutable)
    validate_board(next_board)
    return next_board


def validate_board(board: Board) -> None:
    if len(board) != ROWS or any(len(row) != COLS for row in board):
        raise ValueError(f"expected a {ROWS}x{COLS} board")
    for column in range(COLS):
        seen_empty = False
        for row in range(ROWS - 1, -1, -1):
            value = board[row][column]
            if value not in {EMPTY, AI, OPPONENT}:
                raise ValueError(f"invalid cell value {value} at row {row}, column {column}")
            if value == EMPTY:
                seen_empty = True
            elif seen_empty:
                raise ValueError(f"floating disk at row {row}, column {column}")


def legal_row(board: Board, column: int) -> int | None:
    for row in range(ROWS - 1, -1, -1):
        if board[row][column] == EMPTY:
            return row
    return None


def windows() -> list[tuple[Cell, ...]]:
    all_windows: list[tuple[Cell, ...]] = []
    for row in range(ROWS):
        for column in range(COLS - 3):
            all_windows.append(tuple((row, column + offset) for offset in range(4)))
    for column in range(COLS):
        for row in range(ROWS - 3):
            all_windows.append(tuple((row + offset, column) for offset in range(4)))
    for row in range(ROWS - 3):
        for column in range(COLS - 3):
            all_windows.append(tuple((row + offset, column + offset) for offset in range(4)))
    for row in range(3, ROWS):
        for column in range(COLS - 3):
            all_windows.append(tuple((row - offset, column + offset) for offset in range(4)))
    return all_windows


ALL_WINDOWS = windows()


def score_window(board: Board, cells: tuple[Cell, ...]) -> WindowScore | None:
    values = [board[row][column] for row, column in cells]
    ai_count = values.count(AI)
    opponent_count = values.count(OPPONENT)
    empty_count = values.count(EMPTY)
    if ai_count and opponent_count:
        return None
    if ai_count == 4:
        return WindowScore(cells, 100_000, "win now", ai_count, opponent_count, empty_count)
    if ai_count == 3 and empty_count == 1:
        return WindowScore(cells, 120, "three + space", ai_count, opponent_count, empty_count)
    if ai_count == 2 and empty_count == 2:
        return WindowScore(cells, 16, "two + spaces", ai_count, opponent_count, empty_count)
    if opponent_count == 4:
        return WindowScore(cells, -100_000, "opponent win", ai_count, opponent_count, empty_count)
    if opponent_count == 3 and empty_count == 1:
        return WindowScore(cells, -140, "must block", ai_count, opponent_count, empty_count)
    if opponent_count == 2 and empty_count == 2:
        return WindowScore(cells, -16, "opponent pair", ai_count, opponent_count, empty_count)
    return None


def evaluate_candidate(board: Board, column: int) -> Candidate:
    row = legal_row(board, column)
    if row is None:
        raise ValueError(f"column {column} is full")
    candidate_board = set_cell(board, row, column, AI)
    center_count = sum(1 for r in range(ROWS) if candidate_board[r][COLS // 2] == AI)
    center_score = center_count * 6
    scored: list[WindowScore] = []
    total = center_score
    for cells in ALL_WINDOWS:
        score = score_window(candidate_board, cells)
        if score is None:
            continue
        scored.append(score)
        total += score.score
    scored.sort(key=lambda item: (abs(item.score), item.score), reverse=True)
    return Candidate(column, row, candidate_board, total, center_score, tuple(scored))


def default_board() -> Board:
    # AI is yellow and moves next. Dropping in column 4 completes a horizontal four.
    board = (
        (0, 0, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 0, 0),
        (0, 2, 1, 0, 0, 0, 0),
        (2, 1, 1, 0, 1, 2, 0),
    )
    validate_board(board)
    return board


def candidate_steps(board: Board) -> list[Step]:
    steps: list[Step] = [
        Step("start", "yellow checks legal drops", board, None, None, None, tuple(), None, None)
    ]
    scored: list[tuple[int, int]] = []
    best_column: int | None = None
    best_score: int | None = None
    terminal_candidate: Candidate | None = None
    terminal_window: WindowScore | None = None
    for column in range(COLS):
        if legal_row(board, column) is None:
            continue
        candidate = evaluate_candidate(board, column)
        steps.append(
            Step(
                "drop",
                f"try column {candidate.column + 1}",
                candidate.board,
                candidate.column,
                candidate,
                None,
                tuple(scored),
                best_column,
                best_score,
            )
        )
        winning_window = next((window for window in candidate.windows if window.score >= 100_000), None)
        important_windows = (winning_window,) if winning_window is not None else candidate.windows[:4]
        if candidate.center_score:
            pseudo_window = WindowScore(tuple(), candidate.center_score, "center control", 0, 0, 0)
            important_windows = (pseudo_window,) + important_windows
        for window in important_windows:
            steps.append(
                Step(
                    "scan",
                    window.reason,
                    candidate.board,
                    candidate.column,
                    candidate,
                    window,
                    tuple(scored),
                    best_column,
                    best_score,
                )
            )
        scored.append((candidate.column, candidate.score))
        if best_score is None or candidate.score > best_score:
            best_column = candidate.column
            best_score = candidate.score
        if winning_window is not None:
            terminal_candidate = candidate
            terminal_window = winning_window
            status_text = f"column {candidate.column + 1}: WIN"
        else:
            status_text = f"column {candidate.column + 1}: {candidate.score:+}"
        steps.append(
            Step(
                "score",
                status_text,
                candidate.board,
                candidate.column,
                candidate,
                None,
                tuple(scored),
                best_column,
                best_score,
            )
        )
        if terminal_candidate is not None:
            break

    best = terminal_candidate if terminal_candidate is not None else max(
        (evaluate_candidate(board, column) for column in range(COLS) if legal_row(board, column) is not None),
        key=lambda item: item.score,
    )
    final_window = terminal_window if terminal_window is not None else best.windows[0]
    steps.append(
        Step(
            "choose",
            f"play column {best.column + 1}",
            best.board,
            best.column,
            best,
            final_window,
            tuple(scored),
            best.column,
            best.score,
        )
    )
    return steps


def planned_frames(steps: list[Step], fps: int) -> list[FrameState]:
    timeline: list[FrameState] = []
    for step in steps:
        if step.label == "start":
            timeline.extend(FrameState(step) for _ in range(int(fps * 1.0)))
        elif step.label == "drop":
            frames = max(12, int(fps * 0.42))
            for idx in range(frames):
                timeline.append(FrameState(step, drop_progress=idx / max(1, frames - 1), audio_event="lock" if idx == 0 else None))
        elif step.label == "scan":
            timeline.extend(FrameState(step) for _ in range(int(fps * 0.42)))
        elif step.label == "score":
            timeline.extend(
                FrameState(step, audio_event="swap" if idx == 0 else None)
                for idx in range(int(fps * 0.62))
            )
        elif step.label == "choose":
            timeline.extend(
                FrameState(step, audio_event="sorted" if idx == 0 else None)
                for idx in range(int(fps * 1.9))
            )
    return timeline


def cell_geometry(width: int, height: int) -> tuple[int, int, int, int, int]:
    board_w = int(width * 0.84)
    cell = board_w // COLS
    board_w = cell * COLS
    board_h = cell * ROWS
    left = (width - board_w) // 2
    top = int(height * 0.225)
    return left, top, board_w, board_h, cell


def piece_color(value: int) -> tuple[int, int, int]:
    if value == AI:
        return AI_COLOR
    if value == OPPONENT:
        return OPPONENT_COLOR
    return EMPTY_COLOR


def draw_piece(
    draw: ImageDraw.ImageDraw,
    cx: float,
    cy: float,
    radius: float,
    value: int,
    *,
    outline: tuple[int, int, int] | None = None,
    width: int = 3,
) -> None:
    fill = piece_color(value)
    draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=fill, outline=outline, width=width)
    if value != EMPTY:
        shine_r = radius * 0.24
        draw.ellipse((cx - radius * 0.35 - shine_r, cy - radius * 0.35 - shine_r, cx - radius * 0.35 + shine_r, cy - radius * 0.35 + shine_r), fill=(255, 255, 255, 55))


def highlighted_cells(step: Step) -> set[Cell]:
    if step.window is None:
        return set()
    return set(step.window.cells)


def draw_board(draw: ImageDraw.ImageDraw, step: Step, width: int, height: int, drop_progress: float | None) -> None:
    left, top, board_w, board_h, cell = cell_geometry(width, height)
    radius = cell * 0.38
    bubble.rounded_rect(draw, (left - 14, top - 14, left + board_w + 14, top + board_h + 14), 28, BOARD_BLUE, BOARD_EDGE, 4)
    highlights = highlighted_cells(step)
    for row in range(ROWS):
        for column in range(COLS):
            cx = left + column * cell + cell / 2
            cy = top + row * cell + cell / 2
            outline = WIN_COLOR if (row, column) in highlights else (12, 28, 55)
            outline_width = 8 if (row, column) in highlights else 3
            value = step.board[row][column]
            if step.label == "drop" and step.candidate is not None and row == step.candidate.row and column == step.candidate.column:
                draw_piece(draw, cx, cy, radius, EMPTY, outline=outline, width=outline_width)
                start_y = top - cell * 0.6
                end_y = cy
                moving_y = bubble.lerp(start_y, end_y, bubble.ease(drop_progress or 0.0))
                draw_piece(draw, cx, moving_y, radius, AI, outline=GHOST_COLOR, width=5)
            else:
                draw_piece(draw, cx, cy, radius, value, outline=outline, width=outline_width)

    for column in range(COLS):
        label = str(column + 1)
        typeface = bubble.font(int(width * 0.026), bold=True)
        x = left + column * cell + (cell - bubble.text_width(draw, label, typeface)) / 2
        draw.text((x, top + board_h + int(height * 0.018)), label, font=typeface, fill=bubble.MUTED)


def score_bounds(scored: tuple[tuple[int, int], ...], candidate: Candidate | None) -> tuple[int, int]:
    values = [score for _, score in scored]
    if candidate is not None:
        values.append(candidate.score)
    if not values:
        return -1, 1
    low = min(values + [-160])
    high = max(values + [160])
    if low == high:
        return low - 1, high + 1
    return low, high


def draw_score_bars(draw: ImageDraw.ImageDraw, step: Step, width: int, height: int) -> None:
    margin = int(width * 0.065)
    panel_top = int(height * 0.695)
    panel_h = int(height * 0.172)
    bubble.rounded_rect(draw, (margin, panel_top, width - margin, panel_top + panel_h), 18, bubble.PANEL, bubble.GRID, 2)
    label_font = bubble.font(int(width * 0.026), bold=True)
    value_font = bubble.font(int(width * 0.024), bold=True)
    title = "candidate scores"
    draw.text((margin + 22, panel_top + 16), title, font=label_font, fill=bubble.TEXT)

    low, high = score_bounds(step.scored_columns, step.candidate)
    zero_x0 = margin + int(width * 0.19)
    bar_x0 = margin + int(width * 0.21)
    bar_x1 = width - margin - 22
    span = max(1, high - low)
    zero_x = bar_x0 + ((0 - low) / span) * (bar_x1 - bar_x0)
    row_h = int(panel_h * 0.13)
    start_y = panel_top + 56
    scores = dict(step.scored_columns)
    if step.candidate is not None and step.label in {"drop", "scan"}:
        scores.setdefault(step.candidate.column, step.candidate.score)
    for column in range(COLS):
        y = start_y + column * row_h
        text = f"{column + 1}"
        draw.text((margin + 22, y - 2), text, font=value_font, fill=bubble.MUTED)
        draw.line((bar_x0, y + row_h * 0.48, bar_x1, y + row_h * 0.48), fill=(12, 16, 24), width=8)
        draw.line((zero_x, y + 1, zero_x, y + row_h - 2), fill=bubble.GRID, width=2)
        if column not in scores:
            continue
        score = scores[column]
        score_x = bar_x0 + ((score - low) / span) * (bar_x1 - bar_x0)
        color = WIN_COLOR if column == step.best_column else bubble.COMPARE if score >= 0 else NEGATIVE
        draw.line((zero_x, y + row_h * 0.48, score_x, y + row_h * 0.48), fill=color, width=8)
        value = f"{score:+}"
        draw.text((bar_x1 - bubble.text_width(draw, value, value_font), y - 2), value, font=value_font, fill=color)


def draw_explanation(draw: ImageDraw.ImageDraw, step: Step, width: int, height: int) -> None:
    margin = int(width * 0.065)
    top = int(height * 0.882)
    row_font = bubble.font(int(width * 0.026), bold=True)
    small_font = bubble.font(int(width * 0.022))
    if step.label == "choose":
        color = WIN_COLOR
        heading = f"best move: column {step.best_column + 1}"
        body = "terminal win found; later columns do not matter"
    elif step.window is not None:
        score = step.window.score
        color = WIN_COLOR if score > 0 else NEGATIVE
        heading = f"{step.window.reason}: {score:+}"
        body = f"yellow {step.window.ai_count}, red {step.window.opponent_count}, empty {step.window.empty_count}"
    elif step.candidate is not None and step.label == "score":
        color = bubble.COMPARE if step.candidate.score >= 0 else NEGATIVE
        if any(window.score >= 100_000 for window in step.candidate.windows):
            heading = f"column {step.candidate.column + 1}: WIN"
            body = "a terminal position short-circuits the evaluator"
            color = WIN_COLOR
        else:
            heading = f"column {step.candidate.column + 1} total: {step.candidate.score:+}"
            body = f"center bonus {step.candidate.center_score:+}; open windows summed"
    else:
        color = bubble.TEXT
        heading = "check candidate drops"
        body = "windows of four are the basic unit of evaluation"
    draw.text((margin, top), heading, font=row_font, fill=color)
    draw.text((margin, top + int(height * 0.03)), body, font=small_font, fill=bubble.MUTED)


def draw_frame(
    *,
    width: int,
    height: int,
    step: Step,
    frame_number: int,
    total_frames: int,
    drop_progress: float | None = None,
) -> Image.Image:
    image = bubble.gradient_background(width, height).copy()
    draw = ImageDraw.Draw(image)
    margin = int(width * 0.065)

    title_font = bubble.fit_font(draw, "CONNECT FOUR EVALUATOR", int(width * 0.068), int(width * 0.9), bold=True, min_size=int(width * 0.045))
    subtitle_font = bubble.font(int(width * 0.033))
    label_font = bubble.font(int(width * 0.031), bold=True)
    draw.text(((width - bubble.text_width(draw, "CONNECT FOUR EVALUATOR", title_font)) / 2, int(height * 0.045)), "CONNECT FOUR EVALUATOR", font=title_font, fill=bubble.TEXT)
    bubble.draw_centered_text(draw, int(height * 0.096), "Score positions, but stop immediately on a win.", subtitle_font, bubble.MUTED, width)

    badge_y = int(height * 0.145)
    badge_h = int(height * 0.052)
    bubble.rounded_rect(draw, (margin, badge_y, width - margin, badge_y + badge_h), 18, bubble.PANEL, bubble.GRID, 2)
    color = WIN_COLOR if step.label == "choose" else bubble.COMPARE if step.label in {"drop", "scan"} else bubble.TEXT
    draw.text((margin + 24, badge_y + int(badge_h * 0.22)), step.status, font=label_font, fill=color)
    if step.best_column is not None:
        best = f"best: {step.best_column + 1} ({step.best_score:+})"
        draw.text((width - margin - 24 - bubble.text_width(draw, best, label_font), badge_y + int(badge_h * 0.22)), best, font=label_font, fill=WIN_COLOR)

    draw_board(draw, step, width, height, drop_progress)
    draw_score_bars(draw, step, width, height)
    draw_explanation(draw, step, width, height)

    footer_font = bubble.font(int(width * 0.02))
    footer = f"frame {frame_number + 1}/{total_frames} | possible windows: {len(ALL_WINDOWS)}"
    draw.text((margin, int(height * 0.965)), footer, font=footer_font, fill=bubble.MUTED)
    return image


def render_video(args: argparse.Namespace) -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required but was not found on PATH")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    board = default_board()
    steps = candidate_steps(board)
    timeline = planned_frames(steps, args.fps)
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
            for frame_number, frame_state in enumerate(timeline):
                image = draw_frame(
                    width=args.width,
                    height=args.height,
                    step=frame_state.step,
                    frame_number=frame_number,
                    total_frames=len(timeline),
                    drop_progress=frame_state.drop_progress,
                )
                if not thumbnail_saved and frame_number >= min(len(timeline) - 1, args.fps * 5):
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
            draw_frame(
                width=args.width,
                height=args.height,
                step=steps[-1],
                frame_number=len(timeline) - 1,
                total_frames=len(timeline),
            ).save(args.thumbnail)

        if audio_enabled:
            audio_output = Path(temp_dir) / "operations.wav"
            bubble.generate_audio_track(timeline, args.fps, audio_output)
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

    best = steps[-1]
    duration = len(timeline) / args.fps
    print(f"Rendered {args.output}")
    print(f"Rendered {args.thumbnail}")
    print(f"Frames: {len(timeline)} at {args.fps} fps ({duration:.1f}s)")
    print(f"Audio: {'on' if audio_enabled else 'off'}")
    print(f"Best column: {best.best_column + 1} ({best.best_score:+})")
    print(f"Windows available: {len(ALL_WINDOWS)}")


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
