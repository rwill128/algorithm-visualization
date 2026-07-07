#!/usr/bin/env python3
"""Render a Shorts-ready Connect Four alpha-beta minimax animation."""

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
import render_connect_four_evaluator_short as c4


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "artifacts" / "shorts" / "connect_four_minimax.mp4"
DEFAULT_THUMBNAIL = ROOT / "artifacts" / "shorts" / "connect_four_minimax_thumbnail.png"

SEARCH_DEPTH = 4
WIN_SCORE = 1_000_000
MOVE_ORDER = (3, 2, 4, 1, 5, 0, 6)
BAR_CAP = 600


@dataclass(frozen=True)
class SearchStats:
    nodes: int = 0
    leaves: int = 0
    prunes: int = 0

    def plus(self, other: "SearchStats") -> "SearchStats":
        return SearchStats(self.nodes + other.nodes, self.leaves + other.leaves, self.prunes + other.prunes)


@dataclass(frozen=True)
class RootResult:
    column: int
    row: int
    board: c4.Board
    static_score: int
    minimax_score: int
    stats: SearchStats


@dataclass(frozen=True)
class Step:
    label: str
    status: str
    board: c4.Board
    column: int | None
    row: int | None
    static_scores: tuple[tuple[int, int], ...]
    minimax_scores: tuple[tuple[int, int], ...]
    current_static_score: int | None
    current_minimax_score: int | None
    best_static_column: int | None
    best_minimax_column: int | None
    stats: SearchStats
    alpha: int
    beta: int


@dataclass(frozen=True)
class FrameState:
    step: Step
    drop_progress: float | None = None
    audio_event: str | None = None


def search_board() -> c4.Board:
    # Legal position with yellow to move. A static evaluator likes column 4,
    # but depth-4 alpha-beta finds that column can be refuted.
    return c4.make_board([6, 6, 0, 1, 4, 4, 2, 2, 6, 1, 6, 0])


def legal_moves(board: c4.Board) -> list[int]:
    return [column for column in MOVE_ORDER if c4.legal_row(board, column) is not None]


def drop_piece(board: c4.Board, column: int, player: int) -> tuple[c4.Board, int]:
    row = c4.legal_row(board, column)
    if row is None:
        raise ValueError(f"column {column} is full")
    return c4.set_cell(board, row, column, player), row


def winner(board: c4.Board) -> int:
    for cells in c4.ALL_WINDOWS:
        values = [board[row][column] for row, column in cells]
        if values.count(c4.AI) == 4:
            return c4.AI
        if values.count(c4.OPPONENT) == 4:
            return c4.OPPONENT
    return c4.EMPTY


def evaluate_position(board: c4.Board) -> int:
    won_by = winner(board)
    if won_by == c4.AI:
        return WIN_SCORE
    if won_by == c4.OPPONENT:
        return -WIN_SCORE

    score = 0
    score += sum(1 for row in range(c4.ROWS) if board[row][c4.COLS // 2] == c4.AI) * 6
    score -= sum(1 for row in range(c4.ROWS) if board[row][c4.COLS // 2] == c4.OPPONENT) * 6
    for cells in c4.ALL_WINDOWS:
        window_score = c4.score_window(board, cells)
        if window_score is not None:
            score += window_score.score
    return score


def alpha_beta(
    board: c4.Board,
    depth: int,
    alpha: int,
    beta: int,
    maximizing: bool,
) -> tuple[int, SearchStats]:
    won_by = winner(board)
    moves = legal_moves(board)
    if depth == 0 or won_by != c4.EMPTY or not moves:
        return evaluate_position(board), SearchStats(nodes=1, leaves=1)

    stats = SearchStats(nodes=1)
    if maximizing:
        value = -WIN_SCORE * 2
        for column in moves:
            child, _ = drop_piece(board, column, c4.AI)
            child_value, child_stats = alpha_beta(child, depth - 1, alpha, beta, False)
            stats = stats.plus(child_stats)
            value = max(value, child_value)
            alpha = max(alpha, value)
            if alpha >= beta:
                return value, SearchStats(stats.nodes, stats.leaves, stats.prunes + 1)
        return value, stats

    value = WIN_SCORE * 2
    for column in moves:
        child, _ = drop_piece(board, column, c4.OPPONENT)
        child_value, child_stats = alpha_beta(child, depth - 1, alpha, beta, True)
        stats = stats.plus(child_stats)
        value = min(value, child_value)
        beta = min(beta, value)
        if alpha >= beta:
            return value, SearchStats(stats.nodes, stats.leaves, stats.prunes + 1)
    return value, stats


def root_results(board: c4.Board, depth: int) -> list[RootResult]:
    results: list[RootResult] = []
    for column in legal_moves(board):
        child, row = drop_piece(board, column, c4.AI)
        static_score = evaluate_position(child)
        minimax_score, stats = alpha_beta(child, depth - 1, -WIN_SCORE * 2, WIN_SCORE * 2, False)
        results.append(RootResult(column, row, child, static_score, minimax_score, stats))
    return results


def build_steps(board: c4.Board, results: list[RootResult]) -> list[Step]:
    static_scores = tuple((result.column, result.static_score) for result in results)
    best_static = max(results, key=lambda result: result.static_score).column
    steps = [
        Step(
            "start",
            f"alpha-beta search depth {SEARCH_DEPTH}",
            board,
            None,
            None,
            static_scores,
            tuple(),
            None,
            None,
            best_static,
            None,
            SearchStats(),
            -WIN_SCORE * 2,
            WIN_SCORE * 2,
        )
    ]

    completed: list[tuple[int, int]] = []
    total_stats = SearchStats()
    alpha = -WIN_SCORE * 2
    for result in results:
        steps.append(
            Step(
                "drop",
                f"search column {result.column + 1}",
                result.board,
                result.column,
                result.row,
                static_scores,
                tuple(completed),
                result.static_score,
                None,
                best_static,
                max(completed, key=lambda item: item[1])[0] if completed else None,
                total_stats,
                alpha,
                WIN_SCORE * 2,
            )
        )
        completed.append((result.column, result.minimax_score))
        total_stats = total_stats.plus(result.stats)
        alpha = max(alpha, result.minimax_score)
        best_minimax = max(completed, key=lambda item: item[1])[0]
        status = f"column {result.column + 1}: {format_score(result.minimax_score)}"
        if result.minimax_score <= -WIN_SCORE:
            status = f"column {result.column + 1}: refuted"
        steps.append(
            Step(
                "score",
                status,
                result.board,
                result.column,
                result.row,
                static_scores,
                tuple(completed),
                result.static_score,
                result.minimax_score,
                best_static,
                best_minimax,
                total_stats,
                alpha,
                WIN_SCORE * 2,
            )
        )

    best_result = max(results, key=lambda result: result.minimax_score)
    steps.append(
        Step(
            "choose",
            f"play column {best_result.column + 1}",
            best_result.board,
            best_result.column,
            best_result.row,
            static_scores,
            tuple(completed),
            best_result.static_score,
            best_result.minimax_score,
            best_static,
            best_result.column,
            total_stats,
            alpha,
            WIN_SCORE * 2,
        )
    )
    return steps


def planned_frames(steps: list[Step], fps: int) -> list[FrameState]:
    timeline: list[FrameState] = []
    for step in steps:
        if step.label == "start":
            timeline.extend(FrameState(step) for _ in range(int(fps * 1.4)))
        elif step.label == "drop":
            frames = max(12, int(fps * 0.44))
            for idx in range(frames):
                timeline.append(FrameState(step, drop_progress=idx / max(1, frames - 1), audio_event="lock" if idx == 0 else None))
        elif step.label == "score":
            timeline.extend(
                FrameState(step, audio_event="swap" if idx == 0 else None)
                for idx in range(int(fps * 0.9))
            )
        elif step.label == "choose":
            timeline.extend(
                FrameState(step, audio_event="sorted" if idx == 0 else None)
                for idx in range(int(fps * 2.0))
            )
    return timeline


def format_score(score: int | None) -> str:
    if score is None:
        return "-"
    if score >= WIN_SCORE:
        return "WIN"
    if score <= -WIN_SCORE:
        return "LOSS"
    return f"{score:+}"


def format_bound(score: int) -> str:
    if score >= WIN_SCORE * 2:
        return "+inf"
    if score <= -WIN_SCORE * 2:
        return "-inf"
    return format_score(score)


def clamp_score(score: int) -> int:
    return max(-BAR_CAP, min(BAR_CAP, score))


def draw_board(draw: ImageDraw.ImageDraw, step: Step, width: int, height: int, drop_progress: float | None) -> None:
    left, top, board_w, board_h, cell = c4.cell_geometry(width, height)
    radius = cell * 0.38
    bubble.rounded_rect(draw, (left - 14, top - 14, left + board_w + 14, top + board_h + 14), 28, c4.BOARD_BLUE, c4.BOARD_EDGE, 4)
    for row in range(c4.ROWS):
        for column in range(c4.COLS):
            cx = left + column * cell + cell / 2
            cy = top + row * cell + cell / 2
            outline = (12, 28, 55)
            outline_width = 3
            if step.label == "choose" and row == step.row and column == step.column:
                outline = c4.WIN_COLOR
                outline_width = 8
            value = step.board[row][column]
            if (
                step.label == "drop"
                and row == step.row
                and column == step.column
                and step.column is not None
            ):
                c4.draw_piece(draw, cx, cy, radius, c4.EMPTY, outline=outline, width=outline_width)
                start_y = top - cell * 0.6
                moving_y = bubble.lerp(start_y, cy, bubble.ease(drop_progress or 0.0))
                c4.draw_piece(draw, cx, moving_y, radius, c4.AI, outline=c4.GHOST_COLOR, width=5)
            else:
                c4.draw_piece(draw, cx, cy, radius, value, outline=outline, width=outline_width)

    typeface = bubble.font(int(width * 0.026), bold=True)
    for column in range(c4.COLS):
        label = str(column + 1)
        x = left + column * cell + (cell - bubble.text_width(draw, label, typeface)) / 2
        draw.text((x, top + board_h + int(height * 0.018)), label, font=typeface, fill=bubble.MUTED)


def draw_score_panel(draw: ImageDraw.ImageDraw, step: Step, width: int, height: int) -> None:
    margin = int(width * 0.065)
    panel_top = int(height * 0.695)
    panel_h = int(height * 0.172)
    bubble.rounded_rect(draw, (margin, panel_top, width - margin, panel_top + panel_h), 18, bubble.PANEL, bubble.GRID, 2)
    label_font = bubble.font(int(width * 0.024), bold=True)
    value_font = bubble.font(int(width * 0.021), bold=True)
    draw.text((margin + 22, panel_top + 16), "root move scores", font=label_font, fill=bubble.TEXT)
    draw.text((width - margin - 340, panel_top + 16), "static", font=value_font, fill=bubble.COMPARE)
    draw.text((width - margin - 190, panel_top + 16), f"depth {SEARCH_DEPTH}", font=value_font, fill=c4.WIN_COLOR)

    bar_x0 = margin + int(width * 0.23)
    bar_x1 = width - margin - 210
    zero_x = (bar_x0 + bar_x1) / 2
    row_h = int(panel_h * 0.13)
    start_y = panel_top + 56
    static_scores = dict(step.static_scores)
    minimax_scores = dict(step.minimax_scores)
    for column in range(c4.COLS):
        y = start_y + column * row_h
        draw.text((margin + 22, y - 2), str(column + 1), font=value_font, fill=bubble.MUTED)
        draw.line((bar_x0, y + row_h * 0.48, bar_x1, y + row_h * 0.48), fill=(12, 16, 24), width=8)
        draw.line((zero_x, y + 1, zero_x, y + row_h - 2), fill=bubble.GRID, width=2)
        if column in static_scores:
            static = clamp_score(static_scores[column])
            x = zero_x + (static / BAR_CAP) * ((bar_x1 - bar_x0) / 2)
            draw.line((zero_x, y + row_h * 0.35, x, y + row_h * 0.35), fill=bubble.COMPARE, width=4)
            value = format_score(static_scores[column])
            draw.text((width - margin - 340, y - 2), value, font=value_font, fill=bubble.COMPARE)
        if column in minimax_scores:
            minimax = minimax_scores[column]
            shown = clamp_score(minimax)
            x = zero_x + (shown / BAR_CAP) * ((bar_x1 - bar_x0) / 2)
            color = c4.WIN_COLOR if column == step.best_minimax_column else c4.NEGATIVE if minimax < 0 else bubble.BAR
            draw.line((zero_x, y + row_h * 0.65, x, y + row_h * 0.65), fill=color, width=6)
            value = format_score(minimax)
            draw.text((width - margin - 190, y - 2), value, font=value_font, fill=color)


def draw_metrics(draw: ImageDraw.ImageDraw, step: Step, width: int, height: int) -> None:
    margin = int(width * 0.065)
    top = int(height * 0.878)
    stat_font = bubble.font(int(width * 0.031), bold=True)
    small_font = bubble.font(int(width * 0.022))
    if step.label == "start":
        heading = f"static best: {step.best_static_column + 1}"
        body = "minimax searches replies before trusting the evaluator"
        color = bubble.COMPARE
    elif step.label == "choose":
        heading = f"depth {SEARCH_DEPTH} best: column {step.best_minimax_column + 1}"
        body = f"static wanted {step.best_static_column + 1}; search avoided the trap"
        color = c4.WIN_COLOR
    elif step.current_minimax_score is None:
        heading = f"try column {step.column + 1}: static {format_score(step.current_static_score)}"
        body = "now red replies, yellow replies, and leaf boards get scored"
        color = bubble.COMPARE
    else:
        heading = f"minimax result: {format_score(step.current_minimax_score)}"
        body = f"nodes {step.stats.nodes} | leaves {step.stats.leaves} | prunes {step.stats.prunes}"
        color = c4.NEGATIVE if step.current_minimax_score < 0 else c4.WIN_COLOR
    draw.text((margin, top), heading, font=stat_font, fill=color)
    draw.text((margin, top + int(height * 0.031)), body, font=small_font, fill=bubble.MUTED)

    bounds = f"alpha {format_bound(step.alpha)}   beta {format_bound(step.beta)}"
    draw.text((margin, top + int(height * 0.061)), bounds, font=small_font, fill=bubble.MUTED)


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

    title = "CONNECT FOUR MINIMAX"
    title_font = bubble.fit_font(draw, title, int(width * 0.072), int(width * 0.9), bold=True, min_size=int(width * 0.048))
    subtitle_font = bubble.font(int(width * 0.033))
    label_font = bubble.font(int(width * 0.031), bold=True)
    draw.text(((width - bubble.text_width(draw, title, title_font)) / 2, int(height * 0.045)), title, font=title_font, fill=bubble.TEXT)
    bubble.draw_centered_text(draw, int(height * 0.096), "Use the evaluator at leaf positions, not just after one move.", subtitle_font, bubble.MUTED, width)

    badge_y = int(height * 0.145)
    badge_h = int(height * 0.052)
    bubble.rounded_rect(draw, (margin, badge_y, width - margin, badge_y + badge_h), 18, bubble.PANEL, bubble.GRID, 2)
    color = c4.WIN_COLOR if step.label == "choose" else bubble.COMPARE
    draw.text((margin + 24, badge_y + int(badge_h * 0.22)), step.status, font=label_font, fill=color)
    if step.best_minimax_column is not None:
        best = f"best: {step.best_minimax_column + 1} ({format_score(dict(step.minimax_scores).get(step.best_minimax_column))})"
        draw.text((width - margin - 24 - bubble.text_width(draw, best, label_font), badge_y + int(badge_h * 0.22)), best, font=label_font, fill=c4.WIN_COLOR)

    draw_board(draw, step, width, height, drop_progress)
    draw_score_panel(draw, step, width, height)
    draw_metrics(draw, step, width, height)

    footer_font = bubble.font(int(width * 0.02))
    footer = f"frame {frame_number + 1}/{total_frames} | alpha-beta minimax"
    draw.text((margin, int(height * 0.965)), footer, font=footer_font, fill=bubble.MUTED)
    return image


def render_video(args: argparse.Namespace) -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required but was not found on PATH")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    board = search_board()
    c4.validate_board(board)
    results = root_results(board, args.depth)
    steps = build_steps(board, results)
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

    best_static = max(results, key=lambda result: result.static_score)
    best_search = max(results, key=lambda result: result.minimax_score)
    total_stats = SearchStats()
    for result in results:
        total_stats = total_stats.plus(result.stats)
    print(f"Rendered {args.output}")
    print(f"Rendered {args.thumbnail}")
    print(f"Frames: {len(timeline)} at {args.fps} fps ({len(timeline) / args.fps:.1f}s)")
    print(f"Audio: {'on' if audio_enabled else 'off'}")
    print(f"Static best column: {best_static.column + 1} ({best_static.static_score:+})")
    print(f"Depth {args.depth} best column: {best_search.column + 1} ({format_score(best_search.minimax_score)})")
    print(f"Nodes: {total_stats.nodes}, leaves: {total_stats.leaves}, prunes: {total_stats.prunes}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--thumbnail", type=Path, default=DEFAULT_THUMBNAIL)
    parser.add_argument("--width", type=int, default=1080)
    parser.add_argument("--height", type=int, default=1920)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--depth", type=int, default=SEARCH_DEPTH)
    parser.add_argument("--no-audio", action="store_true")
    args = parser.parse_args()
    if args.width <= 0 or args.height <= 0 or args.fps <= 0:
        parser.error("--width, --height, and --fps must be positive")
    if args.depth < 1:
        parser.error("--depth must be at least 1")
    return args


if __name__ == "__main__":
    render_video(parse_args())
