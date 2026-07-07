#!/usr/bin/env python3
"""Render a Shorts-ready Dijkstra math walkthrough."""

from __future__ import annotations

import argparse
import heapq
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
DEFAULT_OUTPUT = SHORTS_DIR / "dijkstra_math_walkthrough.mp4"
DEFAULT_THUMBNAIL = SHORTS_DIR / "dijkstra_math_walkthrough_thumbnail.png"

DEFAULT_EXPANSIONS = 10
ROWS = 9
COLS = 11
START = (5, 2)
GOAL = (5, 9)
WALLS = frozenset({(2, 6), (3, 6), (4, 6), (5, 6), (6, 6), (7, 6)})
TERRAIN_COSTS = {
    (5, 3): 4,
    (5, 4): 4,
    (5, 5): 4,
    (4, 4): 3,
    (6, 4): 3,
}

Coord = tuple[int, int]

CELL_OPEN = (28, 34, 48)
CELL_EDGE = (48, 58, 78)
WALL = (72, 80, 98)
HEAVY_TERRAIN = (38, 92, 97)
EXPLORED = (58, 119, 203)
FRONTIER = (255, 202, 77)
CURRENT = (255, 104, 91)
CHOSEN = (126, 220, 135)
START_COLOR = (86, 176, 255)
GOAL_COLOR = (251, 132, 76)
TEXT_BLUE = (86, 176, 255)
TEXT_GREEN = (126, 220, 135)
TEXT_YELLOW = (255, 202, 77)
TEXT_RED = (255, 104, 91)


@dataclass(frozen=True)
class Candidate:
    cell: Coord
    status: str
    g: int | None = None
    h: int | None = None
    f: int | None = None
    previous_f: int | None = None


@dataclass(frozen=True)
class Iteration:
    index: int
    current: Coord
    current_g: int
    candidates: tuple[Candidate, ...]
    explored_after: frozenset[Coord]
    frontier_after: dict[Coord, tuple[int, int, int]]
    chosen_next: Coord | None
    chosen_score: tuple[int, int, int] | None


@dataclass(frozen=True)
class FrameState:
    mode: str
    iteration_index: int = 0
    reveal_count: int = 0
    audio_event: str | None = None


def heuristic(cell: Coord) -> int:
    return abs(cell[0] - GOAL[0]) + abs(cell[1] - GOAL[1])


def move_cost(cell: Coord) -> int:
    return TERRAIN_COSTS.get(cell, 1)


def neighbors(cell: Coord) -> tuple[Coord, ...]:
    row, col = cell
    candidates = ((row - 1, col), (row, col + 1), (row + 1, col), (row, col - 1))
    return tuple(candidate for candidate in candidates if 0 <= candidate[0] < ROWS and 0 <= candidate[1] < COLS)


def build_iterations(count: int = DEFAULT_EXPANSIONS) -> tuple[Iteration, ...]:
    frontier: list[tuple[int, int, int, Coord]] = []
    frontier_scores: dict[Coord, tuple[int, int, int]] = {}
    g_score: dict[Coord, int] = {START: 0}
    explored: set[Coord] = set()
    counter = 0
    heapq.heappush(frontier, (0, counter, counter, START))
    frontier_scores[START] = (0, counter, counter)
    iterations: list[Iteration] = []

    for index in range(count):
        while frontier:
            current_priority, tie_order, insert_order, current = heapq.heappop(frontier)
            if frontier_scores.get(current) == (current_priority, tie_order, insert_order):
                break
        else:
            raise RuntimeError("frontier unexpectedly empty")

        frontier_scores.pop(current, None)
        explored.add(current)
        current_g = g_score[current]
        candidate_rows: list[Candidate] = []
        for neighbor in neighbors(current):
            if neighbor in WALLS:
                candidate_rows.append(Candidate(neighbor, "wall"))
                continue
            if neighbor in explored:
                candidate_rows.append(Candidate(neighbor, "explored"))
                continue
            step_cost = move_cost(neighbor)
            tentative_g = current_g + step_cost
            h_score = heuristic(neighbor)
            previous = frontier_scores.get(neighbor)
            previous_g = previous[0] if previous else None
            if previous is None or tentative_g < previous[0]:
                counter += 1
                g_score[neighbor] = tentative_g
                frontier_scores[neighbor] = (tentative_g, counter, counter)
                heapq.heappush(frontier, (tentative_g, counter, counter, neighbor))
            candidate_rows.append(Candidate(neighbor, "open", tentative_g, h_score, tentative_g, previous_g))

        chosen_next = None
        chosen_score = None
        if frontier_scores:
            chosen_next, chosen_score = min(frontier_scores.items(), key=lambda item: item[1])
        iterations.append(
            Iteration(
                index + 1,
                current,
                current_g,
                tuple(candidate_rows),
                frozenset(explored),
                dict(frontier_scores),
                chosen_next,
                chosen_score,
            )
        )
    return tuple(iterations)


def timeline(iterations: tuple[Iteration, ...], fps: int) -> list[FrameState]:
    frames: list[FrameState] = []
    frames.extend(FrameState("intro", audio_event="lock" if frame == 0 else None) for frame in range(int(fps * 2.2)))
    for iteration_index, iteration in enumerate(iterations):
        frames.extend(FrameState("focus", iteration_index, 0) for _ in range(int(fps * 1.0)))
        frames.extend(FrameState("formula", iteration_index, 0) for _ in range(int(fps * 1.15)))
        for reveal in range(1, len(iteration.candidates) + 1):
            frames.extend(
                FrameState("score", iteration_index, reveal, audio_event="swap" if frame == 0 else None)
                for frame in range(int(fps * 1.25))
            )
        frames.extend(
            FrameState("choose", iteration_index, len(iteration.candidates), audio_event="lock" if frame == 0 else None)
            for frame in range(int(fps * 1.7))
        )
    frames.extend(FrameState("outro", len(iterations) - 1, len(iterations[-1].candidates), audio_event="sorted" if frame == 0 else None) for frame in range(int(fps * 3.0)))
    return frames


def cell_bounds(cell: Coord, left: int, top: int, size: int) -> tuple[int, int, int, int]:
    row, col = cell
    x0 = left + col * size
    y0 = top + row * size
    return x0, y0, x0 + size, y0 + size


def draw_grid(
    draw: ImageDraw.ImageDraw,
    *,
    width: int,
    state: FrameState,
    iterations: tuple[Iteration, ...],
    left: int,
    top: int,
    size: int,
) -> None:
    iteration = iterations[min(state.iteration_index, len(iterations) - 1)]
    explored = set()
    frontier: dict[Coord, tuple[int, int, int]] = {}
    if state.mode != "intro":
        previous_index = max(0, state.iteration_index - 1)
        if state.iteration_index > 0:
            explored = set(iterations[previous_index].explored_after)
            frontier = dict(iterations[previous_index].frontier_after)
        if state.mode in {"score", "choose", "outro"}:
            explored = set(iteration.explored_after)
            frontier = dict(iteration.frontier_after)

    bubble.rounded_rect(
        draw,
        (left - 12, top - 12, left + COLS * size + 12, top + ROWS * size + 12),
        18,
        (13, 17, 26),
        bubble.GRID,
        2,
    )

    revealed_candidates = {candidate.cell for candidate in iteration.candidates[: state.reveal_count]}
    for row in range(ROWS):
        for col in range(COLS):
            cell = (row, col)
            x0, y0, x1, y1 = cell_bounds(cell, left, top, size)
            inset = max(3, size // 14)
            fill = CELL_OPEN
            outline = CELL_EDGE
            if cell in WALLS:
                fill = WALL
            elif cell in TERRAIN_COSTS:
                fill = HEAVY_TERRAIN
            elif cell == GOAL:
                fill = GOAL_COLOR
            elif cell == START:
                fill = START_COLOR
            elif cell == iteration.current and state.mode not in {"intro", "outro"}:
                fill = CURRENT
            elif cell == iteration.chosen_next and state.mode in {"choose", "outro"}:
                fill = CHOSEN
            elif cell in revealed_candidates:
                fill = (65, 58, 32)
                outline = FRONTIER
            elif cell in frontier:
                fill = (54, 47, 29)
                outline = FRONTIER
            elif cell in explored:
                fill = EXPLORED

            bubble.rounded_rect(draw, (x0 + inset, y0 + inset, x1 - inset, y1 - inset), max(5, size // 8), fill, outline, 2)

            label = ""
            label_color = bubble.TEXT
            if cell == START:
                label = "S"
            elif cell == GOAL:
                label = "G"
            elif cell == iteration.current and state.mode not in {"intro", "outro"}:
                label = "C"
            elif cell in frontier and size >= 48:
                label = str(frontier[cell][0])
                label_color = FRONTIER
            elif cell in TERRAIN_COSTS and size >= 48:
                label = f"+{move_cost(cell)}"
                label_color = bubble.MUTED
            if label:
                font = bubble.font(int(size * 0.38), bold=True)
                draw.text(
                    (x0 + (size - bubble.text_width(draw, label, font)) / 2, y0 + int(size * 0.28)),
                    label,
                    font=font,
                    fill=label_color,
                )

    current_x0, current_y0, current_x1, current_y1 = cell_bounds(iteration.current, left, top, size)
    if state.mode not in {"intro", "outro"}:
        draw.rectangle((current_x0 + 4, current_y0 + 4, current_x1 - 4, current_y1 - 4), outline=CURRENT, width=5)
    if iteration.chosen_next and state.mode in {"choose", "outro"}:
        x0, y0, x1, y1 = cell_bounds(iteration.chosen_next, left, top, size)
        draw.rectangle((x0 + 2, y0 + 2, x1 - 2, y1 - 2), outline=CHOSEN, width=5)


def draw_formula(draw: ImageDraw.ImageDraw, x: int, y: int, width: int) -> None:
    font = bubble.font(46, bold=True)
    small = bubble.font(27)
    parts = (("priority", TEXT_YELLOW), (" = ", bubble.TEXT), ("g", TEXT_BLUE))
    cursor = x
    for text, color in parts:
        draw.text((cursor, y), text, font=font, fill=color)
        cursor += bubble.text_width(draw, text, font)
    draw.text((x, y + 62), "g: exact cost already paid from S", font=small, fill=TEXT_BLUE)
    draw.text((x, y + 100), "heavier cells add more cost", font=small, fill=TEXT_GREEN)
    draw.text((x, y + 138), "Dijkstra expands the frontier cell with the smallest g.", font=small, fill=bubble.MUTED)


def candidate_label(candidate: Candidate) -> str:
    row, col = candidate.cell
    return f"({row},{col})"


def draw_candidate_cards(
    draw: ImageDraw.ImageDraw,
    *,
    width: int,
    y: int,
    iteration: Iteration,
    reveal_count: int,
) -> None:
    margin = int(width * 0.065)
    gap = 12
    card_w = int((width - 2 * margin - 3 * gap) / 4)
    card_h = 210
    title_font = bubble.font(28, bold=True)
    line_font = bubble.font(23)
    value_font = bubble.font(25, bold=True)
    for index, candidate in enumerate(iteration.candidates):
        x0 = margin + index * (card_w + gap)
        shown = index < reveal_count
        border = FRONTIER if shown else bubble.GRID
        bubble.rounded_rect(draw, (x0, y, x0 + card_w, y + card_h), 18, bubble.PANEL, border, 2)
        if not shown:
            draw.text((x0 + 18, y + 82), "neighbor", font=title_font, fill=bubble.MUTED)
            continue
        color = TEXT_YELLOW
        if candidate.status == "wall":
            color = WALL
        elif candidate.status == "explored":
            color = EXPLORED
        draw.text((x0 + 16, y + 16), candidate_label(candidate), font=title_font, fill=color)
        if candidate.status == "wall":
            draw.text((x0 + 16, y + 78), "wall", font=value_font, fill=WALL)
            draw.text((x0 + 16, y + 118), "skip", font=line_font, fill=bubble.MUTED)
        elif candidate.status == "explored":
            draw.text((x0 + 16, y + 78), "closed", font=value_font, fill=EXPLORED)
            draw.text((x0 + 16, y + 118), "skip", font=line_font, fill=bubble.MUTED)
        else:
            assert candidate.g is not None and candidate.h is not None and candidate.f is not None
            step_cost = move_cost(candidate.cell)
            draw.text((x0 + 16, y + 58), f"cost = {step_cost}", font=line_font, fill=TEXT_GREEN)
            draw.text((x0 + 16, y + 96), f"g = {iteration.current_g}+{step_cost}", font=line_font, fill=TEXT_BLUE)
            draw.text((x0 + 16, y + 134), f"  = {candidate.g}", font=line_font, fill=TEXT_BLUE)
            draw.text((x0 + 16, y + 172), f"priority = {candidate.g}", font=value_font, fill=TEXT_YELLOW)


def draw_frontier_panel(
    draw: ImageDraw.ImageDraw,
    *,
    width: int,
    y: int,
    iteration: Iteration,
    state: FrameState,
) -> None:
    margin = int(width * 0.065)
    panel_h = 145
    bubble.rounded_rect(draw, (margin, y, width - margin, y + panel_h), 18, bubble.PANEL, bubble.GRID, 2)
    label_font = bubble.font(28, bold=True)
    small_font = bubble.font(24)
    draw.text((margin + 20, y + 18), "frontier priority queue", font=label_font, fill=bubble.TEXT)
    if state.mode in {"intro", "focus", "formula"}:
        draw.text((margin + 20, y + 72), "score neighbors, then choose the smallest g", font=small_font, fill=bubble.MUTED)
        return

    entries = sorted(iteration.frontier_after.items(), key=lambda item: item[1])[:5]
    cursor = margin + 20
    entry_y = y + 72
    for cell, (g_cost, _, _) in entries:
        text = f"{cell}: g{g_cost}"
        color = CHOSEN if cell == iteration.chosen_next and state.mode in {"choose", "outro"} else FRONTIER
        pill_w = bubble.text_width(draw, text, small_font) + 28
        bubble.rounded_rect(draw, (cursor, entry_y - 8, cursor + pill_w, entry_y + 36), 14, (35, 37, 44), color, 2)
        draw.text((cursor + 14, entry_y), text, font=small_font, fill=color)
        cursor += pill_w + 12
        if cursor > width - margin - 180:
            break

    if iteration.chosen_next is not None and iteration.chosen_score is not None and state.mode in {"choose", "outro"}:
        lowest_g = iteration.chosen_score[0]
        tie_count = sum(1 for score in iteration.frontier_after.values() if score[0] == lowest_g)
        if tie_count > 1:
            chosen = f"next: {iteration.chosen_next} by tie order"
        else:
            chosen = f"next: {iteration.chosen_next} has lowest g"
        draw.text((width - margin - 20 - bubble.text_width(draw, chosen, small_font), y + 20), chosen, font=small_font, fill=CHOSEN)


def draw_frame(width: int, height: int, iterations: tuple[Iteration, ...], state: FrameState, frame_number: int, total_frames: int) -> Image.Image:
    image = bubble.gradient_background(width, height).copy()
    draw = ImageDraw.Draw(image)
    margin = int(width * 0.065)
    title = "DIJKSTRA STEP BY STEP"
    title_font = bubble.fit_font(draw, title, 68, int(width * 0.9), bold=True, min_size=44)
    subtitle_font = bubble.font(32)
    draw.text(((width - bubble.text_width(draw, title, title_font)) / 2, 72), title, font=title_font, fill=bubble.TEXT)
    subtitle = f"{len(iterations)} expansions, choosing the smallest g."
    bubble.draw_centered_text(draw, 154, subtitle, subtitle_font, bubble.MUTED, width)

    grid_size = 72
    grid_left = int((width - COLS * grid_size) / 2)
    grid_top = 275
    draw_grid(draw, width=width, state=state, iterations=iterations, left=grid_left, top=grid_top, size=grid_size)

    iteration = iterations[min(state.iteration_index, len(iterations) - 1)]
    badge_font = bubble.font(31, bold=True)
    badge_y = 215
    if state.mode == "intro":
        badge = "Start with S in the frontier."
    elif state.mode == "outro":
        badge = "Repeat until G is chosen from the frontier."
    else:
        badge = f"Expansion {iteration.index}: current cell {iteration.current}, g = {iteration.current_g}"
    bubble.draw_centered_text(draw, badge_y, badge, badge_font, TEXT_YELLOW if state.mode != "outro" else CHOSEN, width)

    math_y = 965
    bubble.rounded_rect(draw, (margin, math_y, width - margin, math_y + 205), 18, (15, 19, 29), bubble.GRID, 2)
    draw_formula(draw, margin + 26, math_y + 28, width - 2 * margin - 52)

    candidate_y = 1205
    reveal_count = state.reveal_count if state.mode in {"score", "choose", "outro"} else 0
    draw_candidate_cards(draw, width=width, y=candidate_y, iteration=iteration, reveal_count=reveal_count)

    draw_frontier_panel(draw, width=width, y=1455, iteration=iteration, state=state)

    footer_font = bubble.font(22)
    footer = f"frame {frame_number + 1}/{total_frames} | Dijkstra priority uses exact cost g"
    draw.text((margin, int(height * 0.965)), footer, font=footer_font, fill=bubble.MUTED)
    return image


def render_video(args: argparse.Namespace) -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required but was not found on PATH")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    iterations = build_iterations(args.expansions)
    frames = timeline(iterations, args.fps)
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
                image = draw_frame(args.width, args.height, iterations, frame_state, frame_number, len(frames))
                if not thumbnail_saved and frame_state.mode == "choose" and frame_state.iteration_index == 1:
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
            draw_frame(args.width, args.height, iterations, frames[-1], len(frames) - 1, len(frames)).save(args.thumbnail)

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
    for iteration in iterations:
        print(f"Expansion {iteration.index}: current={iteration.current} g={iteration.current_g} next={iteration.chosen_next} frontier={len(iteration.frontier_after)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--thumbnail", type=Path, default=DEFAULT_THUMBNAIL)
    parser.add_argument("--width", type=int, default=1080)
    parser.add_argument("--height", type=int, default=1920)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--expansions", type=int, default=DEFAULT_EXPANSIONS)
    parser.add_argument("--no-audio", action="store_true")
    args = parser.parse_args()
    if args.width <= 0 or args.height <= 0 or args.fps <= 0 or args.expansions <= 0:
        parser.error("--width, --height, --fps, and --expansions must be positive")
    return args


if __name__ == "__main__":
    render_video(parse_args())
