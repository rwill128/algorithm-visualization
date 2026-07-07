#!/usr/bin/env python3
"""Render Shorts-ready grid pathfinding animations."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import dataclass
from heapq import heappop, heappush
from pathlib import Path
import shutil
import subprocess
import tempfile

from PIL import Image, ImageDraw

import render_bubble_sort_short as bubble


ROOT = Path(__file__).resolve().parents[1]
SHORTS_DIR = ROOT / "artifacts" / "shorts"

ROWS = 23
COLS = 15
START = (20, 1)
GOAL = (2, 13)
Coord = tuple[int, int]

CELL_DARK = (17, 22, 32)
CELL_OPEN = (30, 37, 51)
WALL = (70, 78, 96)
EXPLORED = (58, 119, 203)
FRONTIER = (255, 202, 77)
CURRENT = (255, 104, 91)
PATH = (126, 220, 135)
START_COLOR = (86, 176, 255)
GOAL_COLOR = (251, 132, 76)
GRID_EDGE = (49, 58, 76)


@dataclass(frozen=True)
class AlgorithmSpec:
    slug: str
    title: str
    subtitle: str
    start_status: str
    expand_status: str
    found_status: str
    done_status: str
    cost_caption: str
    footer: str
    optimal: bool


ALGORITHMS = {
    "astar": AlgorithmSpec(
        "a_star_pathfinding",
        "A-STAR PATHFINDING",
        "Search the cheapest route through walls.",
        "frontier starts at S",
        "expand lowest f = g + h",
        "goal reached; rebuild path",
        "shortest path length",
        "frontier chooses the smallest estimated total cost",
        "A* with Manhattan heuristic",
        True,
    ),
    "bfs": AlgorithmSpec(
        "breadth_first_search",
        "BREADTH-FIRST SEARCH",
        "Expand every nearest cell before going deeper.",
        "queue starts at S",
        "dequeue shallowest cell",
        "goal reached; rebuild path",
        "shortest path length",
        "FIFO queue guarantees shortest paths on uniform grids",
        "BFS on an unweighted grid",
        True,
    ),
    "dfs": AlgorithmSpec(
        "depth_first_search",
        "DEPTH-FIRST SEARCH",
        "Follow one route until it dead-ends.",
        "stack starts at S",
        "pop deepest cell",
        "goal found; rebuild route",
        "route length",
        "stack order drives the search, not shortest-distance proof",
        "DFS with a stack",
        False,
    ),
    "dijkstra": AlgorithmSpec(
        "dijkstra_pathfinding",
        "DIJKSTRA PATHFINDING",
        "Expand the lowest known distance first.",
        "priority queue starts at S",
        "expand lowest distance",
        "goal reached; rebuild path",
        "shortest path length",
        "priority queue chooses the smallest known distance",
        "Dijkstra on a uniform-cost grid",
        True,
    ),
    "greedy": AlgorithmSpec(
        "greedy_best_first",
        "GREEDY BEST-FIRST",
        "Chase the cell that looks closest to the goal.",
        "frontier starts at S",
        "expand lowest h",
        "goal found; rebuild route",
        "route length",
        "heuristic-only search can be fast without proving optimality",
        "Greedy best-first with Manhattan heuristic",
        False,
    ),
}


@dataclass(frozen=True)
class SearchSnapshot:
    label: str
    status: str
    current: Coord | None
    frontier: frozenset[Coord]
    explored: frozenset[Coord]
    path: tuple[Coord, ...]
    g_cost: int | None
    h_cost: int | None
    f_cost: int | None
    expanded: int
    edge_checks: int
    discovered: int
    path_cost: int | None


@dataclass(frozen=True)
class FrameState:
    snapshot: SearchSnapshot
    pulse: float = 0.0
    audio_event: str | None = None


def obstacle_grid() -> frozenset[Coord]:
    walls: set[Coord] = set()

    for row in range(3, 22):
        if row not in (5, 16):
            walls.add((row, 3))
    for row in range(1, 19):
        if row not in (3, 12):
            walls.add((row, 6))
    for row in range(4, 23):
        if row not in (7, 18):
            walls.add((row, 9))
    for row in range(1, 20):
        if row not in (10, 14):
            walls.add((row, 12))

    for col in range(2, 14):
        if col not in (4, 10):
            walls.add((6, col))
    for col in range(1, 12):
        if col not in (2, 8):
            walls.add((11, col))
    for col in range(4, 15):
        if col not in (5, 11):
            walls.add((16, col))

    walls.discard(START)
    walls.discard(GOAL)
    return frozenset(walls)


def heuristic(left: Coord, right: Coord) -> int:
    return abs(left[0] - right[0]) + abs(left[1] - right[1])


def neighbors(cell: Coord) -> tuple[Coord, ...]:
    row, col = cell
    candidates = ((row - 1, col), (row, col + 1), (row + 1, col), (row, col - 1))
    return tuple(candidate for candidate in candidates if 0 <= candidate[0] < ROWS and 0 <= candidate[1] < COLS)


def reconstruct_path(came_from: dict[Coord, Coord], current: Coord) -> tuple[Coord, ...]:
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return tuple(path)


def frontier_priority(algorithm: str, cell: Coord, g_cost: int, counter: int) -> tuple[int, int, int, Coord]:
    h_cost = heuristic(cell, GOAL)
    if algorithm == "astar":
        return g_cost + h_cost, h_cost, counter, cell
    if algorithm == "dijkstra":
        return g_cost, 0, counter, cell
    if algorithm == "greedy":
        return h_cost, g_cost, counter, cell
    if algorithm == "bfs":
        return g_cost, counter, 0, cell
    raise ValueError(f"{algorithm} does not use heap priority")


def push_frontier(container: list[Coord] | list[tuple[int, int, int, Coord]], algorithm: str, cell: Coord, g_cost: int, counter: int) -> None:
    if algorithm == "dfs":
        container.append(cell)  # type: ignore[arg-type]
    else:
        heappush(container, frontier_priority(algorithm, cell, g_cost, counter))  # type: ignore[arg-type]


def pop_frontier(container: list[Coord] | list[tuple[int, int, int, Coord]], algorithm: str) -> Coord:
    if algorithm == "dfs":
        return container.pop()  # type: ignore[return-value]
    return heappop(container)[3]  # type: ignore[arg-type]


def search_snapshots(walls: frozenset[Coord], algorithm: str) -> list[SearchSnapshot]:
    spec = ALGORITHMS[algorithm]
    frontier_heap: list[tuple[int, int, int, Coord]] = []
    frontier_stack: list[Coord] = []
    frontier_set: set[Coord] = {START}
    came_from: dict[Coord, Coord] = {}
    g_score: dict[Coord, int] = {START: 0}
    explored: set[Coord] = set()
    counter = 0
    edge_checks = 0

    container: list[Coord] | list[tuple[int, int, int, Coord]]
    if algorithm == "dfs":
        container = frontier_stack
    else:
        container = frontier_heap
    push_frontier(container, algorithm, START, 0, counter)
    snapshots = [
        SearchSnapshot(
            "start",
            spec.start_status,
            START,
            frozenset(frontier_set),
            frozenset(),
            tuple(),
            0,
            heuristic(START, GOAL),
            heuristic(START, GOAL),
            0,
            0,
            1,
            None,
        )
    ]

    while container:
        current = pop_frontier(container, algorithm)
        if current in explored or current not in frontier_set:
            continue

        frontier_set.remove(current)
        explored.add(current)

        if current == GOAL:
            path = reconstruct_path(came_from, current)
            snapshots.append(
                SearchSnapshot(
                    "found",
                    spec.found_status,
                    current,
                    frozenset(frontier_set),
                    frozenset(explored),
                    path,
                    g_score[current],
                    0,
                    g_score[current],
                    len(explored),
                    edge_checks,
                    len(g_score),
                    g_score[current],
                )
            )
            for index in range(1, len(path) + 1):
                snapshots.append(
                    SearchSnapshot(
                        "path",
                        "trace parents back to S",
                        current,
                        frozenset(frontier_set),
                        frozenset(explored),
                        path[:index],
                        g_score[current],
                        0,
                        g_score[current],
                        len(explored),
                        edge_checks,
                        len(g_score),
                        g_score[current],
                    )
                )
            snapshots.append(
                SearchSnapshot(
                    "done",
                    f"{spec.done_status} {g_score[current]}",
                    current,
                    frozenset(frontier_set),
                    frozenset(explored),
                    path,
                    g_score[current],
                    0,
                    g_score[current],
                    len(explored),
                    edge_checks,
                    len(g_score),
                    g_score[current],
                )
            )
            return snapshots

        candidates = neighbors(current)
        if algorithm == "dfs":
            candidates = tuple(reversed(candidates))

        for neighbor in candidates:
            edge_checks += 1
            if neighbor in walls or neighbor in explored:
                continue
            tentative_g = g_score[current] + 1
            if algorithm in {"bfs", "dfs"}:
                if neighbor in g_score:
                    continue
            elif tentative_g >= g_score.get(neighbor, 1_000_000):
                continue
            came_from[neighbor] = current
            g_score[neighbor] = tentative_g
            frontier_set.add(neighbor)
            counter += 1
            push_frontier(container, algorithm, neighbor, tentative_g, counter)

        g_cost = g_score[current]
        h_cost = heuristic(current, GOAL)
        snapshots.append(
            SearchSnapshot(
                "expand",
                spec.expand_status,
                current,
                frozenset(frontier_set),
                frozenset(explored),
                tuple(),
                g_cost,
                h_cost,
                g_cost + h_cost,
                len(explored),
                edge_checks,
                len(g_score),
                None,
            )
        )

    raise RuntimeError(f"{spec.title} did not find a path through the configured grid")


def planned_frames(snapshots: list[SearchSnapshot], fps: int) -> list[FrameState]:
    timeline: list[FrameState] = []
    timeline.extend(FrameState(snapshots[0]) for _ in range(int(fps * 1.1)))

    for index, snapshot in enumerate(snapshots[1:], start=1):
        if snapshot.label == "expand":
            for frame in range(2):
                event = "swap" if frame == 0 and index % 7 == 0 else None
                timeline.append(FrameState(snapshot, pulse=frame / 1, audio_event=event))
        elif snapshot.label == "found":
            timeline.extend(FrameState(snapshot, audio_event="lock" if frame == 0 else None) for frame in range(int(fps * 0.7)))
        elif snapshot.label == "path":
            for frame in range(3):
                event = "lock" if frame == 0 and len(snapshot.path) % 6 == 0 else None
                timeline.append(FrameState(snapshot, pulse=frame / 2, audio_event=event))
        elif snapshot.label == "done":
            timeline.extend(FrameState(snapshot, audio_event="sorted" if frame == 0 else None) for frame in range(int(fps * 2.4)))
        else:
            timeline.append(FrameState(snapshot))
    return timeline


def grid_geometry(width: int, height: int) -> tuple[int, int, int, int]:
    max_w = int(width * 0.86)
    max_h = int(height * 0.56)
    cell = max(7, min(max_w // COLS, max_h // ROWS))
    grid_w = cell * COLS
    grid_h = cell * ROWS
    left = int((width - grid_w) / 2)
    top = int(height * 0.215)
    return left, top, cell, max(2, int(cell * 0.11))


def blend(a: tuple[int, int, int], b: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    return tuple(int(bubble.lerp(a[channel], b[channel], t)) for channel in range(3))


def cell_color(cell: Coord, snapshot: SearchSnapshot, walls: frozenset[Coord]) -> tuple[int, int, int]:
    if cell in walls:
        return WALL
    if cell == START:
        return START_COLOR
    if cell == GOAL:
        return GOAL_COLOR
    if cell in snapshot.path:
        return PATH
    if cell == snapshot.current and snapshot.label in {"expand", "found"}:
        return CURRENT
    if cell in snapshot.frontier:
        return FRONTIER
    if cell in snapshot.explored:
        return EXPLORED
    return CELL_OPEN


def draw_legend(draw: ImageDraw.ImageDraw, width: int, height: int, y: int) -> None:
    labels = (
        ("wall", WALL),
        ("frontier", FRONTIER),
        ("explored", EXPLORED),
        ("current", CURRENT),
        ("path", PATH),
    )
    typeface = bubble.font(int(width * 0.022), bold=True)
    x = int(width * 0.08)
    swatch = int(width * 0.022)
    for label, color in labels:
        bubble.rounded_rect(draw, (x, y, x + swatch, y + swatch), 5, color)
        draw.text((x + swatch + 8, y - 2), label, font=typeface, fill=bubble.MUTED)
        x += swatch + 10 + bubble.text_width(draw, label, typeface) + int(width * 0.025)


def draw_grid(
    draw: ImageDraw.ImageDraw,
    snapshot: SearchSnapshot,
    walls: frozenset[Coord],
    width: int,
    height: int,
    pulse: float,
) -> None:
    left, top, cell, gap = grid_geometry(width, height)
    grid_w = cell * COLS
    grid_h = cell * ROWS
    bubble.rounded_rect(draw, (left - 14, top - 14, left + grid_w + 14, top + grid_h + 14), 22, bubble.PANEL, GRID_EDGE, 3)

    for row in range(ROWS):
        for col in range(COLS):
            coord = (row, col)
            x0 = left + col * cell + gap
            y0 = top + row * cell + gap
            x1 = left + (col + 1) * cell - gap
            y1 = top + (row + 1) * cell - gap
            color = cell_color(coord, snapshot, walls)
            if coord == snapshot.current and snapshot.label == "expand":
                color = blend(color, bubble.TEXT, 0.12 + 0.1 * pulse)
            bubble.rounded_rect(draw, (x0, y0, x1, y1), max(3, int(cell * 0.12)), color, None)

    label_font = bubble.font(int(width * 0.026), bold=True)
    for coord, label, color in ((START, "S", bubble.TEXT), (GOAL, "G", bubble.TEXT)):
        row, col = coord
        x0 = left + col * cell
        y0 = top + row * cell
        label_w = bubble.text_width(draw, label, label_font)
        draw.text((x0 + (cell - label_w) / 2, y0 + int(cell * 0.17)), label, font=label_font, fill=color)


def draw_metric_card(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    label: str,
    value: str,
    color: tuple[int, int, int],
    width: int,
) -> None:
    x0, y0, x1, y1 = box
    value_font = bubble.font(int(width * 0.038), bold=True)
    label_font = bubble.font(int(width * 0.024))
    bubble.rounded_rect(draw, box, 16, bubble.PANEL, bubble.GRID, 2)
    draw.text((x0 + 18, y0 + int((y1 - y0) * 0.15)), value, font=value_font, fill=color)
    draw.text((x0 + 18, y0 + int((y1 - y0) * 0.58)), label, font=label_font, fill=bubble.MUTED)


def cost_text_for(snapshot: SearchSnapshot, spec: AlgorithmSpec) -> str:
    if spec.slug == "a_star_pathfinding":
        return f"g {snapshot.g_cost}   h {snapshot.h_cost}   f {snapshot.f_cost}"
    if spec.slug == "greedy_best_first":
        return f"h {snapshot.h_cost}   depth {snapshot.g_cost}"
    if spec.slug == "dijkstra_pathfinding":
        return f"distance {snapshot.g_cost}"
    if spec.slug == "breadth_first_search":
        return f"depth {snapshot.g_cost}"
    if spec.slug == "depth_first_search":
        return f"depth {snapshot.g_cost}"
    return f"cost {snapshot.g_cost}"


def draw_metrics(draw: ImageDraw.ImageDraw, snapshot: SearchSnapshot, spec: AlgorithmSpec, width: int, height: int) -> None:
    margin = int(width * 0.065)
    top = int(height * 0.792)
    gap = int(width * 0.024)
    card_h = int(height * 0.055)
    card_w = int((width - margin * 2 - gap * 2) / 3)
    cards = (
        ("expanded", str(snapshot.expanded), EXPLORED),
        ("edge checks", str(snapshot.edge_checks), FRONTIER),
        ("discovered", str(snapshot.discovered), bubble.BAR),
    )
    for index, (label, value, color) in enumerate(cards):
        x0 = margin + index * (card_w + gap)
        draw_metric_card(draw, (x0, top, x0 + card_w, top + card_h), label, value, color, width)

    cost_top = top + card_h + int(height * 0.014)
    cost_h = int(height * 0.05)
    bubble.rounded_rect(draw, (margin, cost_top, width - margin, cost_top + cost_h), 16, bubble.PANEL, bubble.GRID, 2)
    cost_font = bubble.font(int(width * 0.031), bold=True)
    small_font = bubble.font(int(width * 0.024))
    if snapshot.path_cost is None:
        cost_text = cost_text_for(snapshot, spec)
        caption = spec.cost_caption
        color = CURRENT if snapshot.label == "expand" else FRONTIER
    else:
        cost_text = f"path length {snapshot.path_cost}"
        caption = "parent links reconstruct the route"
        color = PATH
    draw.text((margin + 22, cost_top + int(cost_h * 0.18)), cost_text, font=cost_font, fill=color)
    draw.text((width - margin - 22 - bubble.text_width(draw, caption, small_font), cost_top + int(cost_h * 0.28)), caption, font=small_font, fill=bubble.MUTED)

    progress_top = int(height * 0.948)
    progress_h = int(height * 0.014)
    path_total = snapshot.path_cost or 0
    path_fill = len(snapshot.path) / max(1, path_total + 1)
    label = "path reconstruction"
    value = f"{min(1.0, path_fill):.2f}"
    label_y = int(height * 0.92)
    draw.text((margin, label_y), label, font=small_font, fill=bubble.MUTED)
    draw.text((width - margin - bubble.text_width(draw, value, small_font), label_y), value, font=small_font, fill=PATH)
    bubble.rounded_rect(draw, (margin, progress_top, width - margin, progress_top + progress_h), 12, (15, 18, 25), bubble.GRID, 2)
    inner_x0 = margin + 4
    inner_x1 = width - margin - 4
    inner_y0 = progress_top + 4
    inner_y1 = progress_top + progress_h - 4
    if path_fill > 0 and inner_y1 > inner_y0:
        bubble.rounded_rect(draw, (inner_x0, inner_y0, bubble.lerp(inner_x0, inner_x1, min(1.0, path_fill)), inner_y1), 8, PATH)


def draw_frame(
    *,
    width: int,
    height: int,
    snapshot: SearchSnapshot,
    spec: AlgorithmSpec,
    walls: frozenset[Coord],
    frame_number: int,
    total_frames: int,
    pulse: float = 0.0,
) -> Image.Image:
    image = bubble.gradient_background(width, height).copy()
    draw = ImageDraw.Draw(image)
    margin = int(width * 0.065)

    title = spec.title
    title_font = bubble.fit_font(draw, title, int(width * 0.074), int(width * 0.9), bold=True, min_size=int(width * 0.048))
    subtitle_font = bubble.font(int(width * 0.033))
    label_font = bubble.font(int(width * 0.031), bold=True)
    draw.text(((width - bubble.text_width(draw, title, title_font)) / 2, int(height * 0.045)), title, font=title_font, fill=bubble.TEXT)
    bubble.draw_centered_text(draw, int(height * 0.096), spec.subtitle, subtitle_font, bubble.MUTED, width)

    badge_y = int(height * 0.145)
    badge_h = int(height * 0.052)
    bubble.rounded_rect(draw, (margin, badge_y, width - margin, badge_y + badge_h), 18, bubble.PANEL, bubble.GRID, 2)
    status_color = PATH if snapshot.label in {"path", "done"} else CURRENT if snapshot.label == "expand" else FRONTIER
    draw.text((margin + 24, badge_y + int(badge_h * 0.22)), snapshot.status, font=label_font, fill=status_color)
    right_text = f"frontier: {len(snapshot.frontier)}"
    if snapshot.label == "done":
        right_text = "route complete"
    draw.text(
        (width - margin - 24 - bubble.text_width(draw, right_text, label_font), badge_y + int(badge_h * 0.22)),
        right_text,
        font=label_font,
        fill=PATH if snapshot.label == "done" else bubble.MUTED,
    )

    draw_legend(draw, width, height, int(height * 0.187))
    draw_grid(draw, snapshot, walls, width, height, pulse)
    draw_metrics(draw, snapshot, spec, width, height)

    footer_font = bubble.font(int(width * 0.02))
    footer = f"frame {frame_number + 1}/{total_frames} | {spec.footer}"
    draw.text((margin, int(height * 0.965)), footer, font=footer_font, fill=bubble.MUTED)
    return image


def default_output_for(spec: AlgorithmSpec) -> Path:
    return SHORTS_DIR / f"{spec.slug}.mp4"


def default_thumbnail_for(spec: AlgorithmSpec) -> Path:
    return SHORTS_DIR / f"{spec.slug}_thumbnail.png"


def render_video(args: argparse.Namespace, algorithm: str) -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required but was not found on PATH")

    spec = ALGORITHMS[algorithm]
    output = args.output or default_output_for(spec)
    thumbnail = args.thumbnail or default_thumbnail_for(spec)

    output.parent.mkdir(parents=True, exist_ok=True)
    walls = obstacle_grid()
    snapshots = search_snapshots(walls, algorithm)
    timeline = planned_frames(snapshots, args.fps)
    audio_enabled = not args.no_audio
    temp_context = tempfile.TemporaryDirectory() if audio_enabled else nullcontext(None)

    with temp_context as temp_dir:
        video_output = Path(temp_dir) / "silent.mp4" if audio_enabled else output
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
                    snapshot=frame_state.snapshot,
                    spec=spec,
                    walls=walls,
                    frame_number=frame_number,
                    total_frames=len(timeline),
                    pulse=frame_state.pulse,
                )
                if not thumbnail_saved and frame_number >= min(len(timeline) - 1, args.fps * 5):
                    thumbnail.parent.mkdir(parents=True, exist_ok=True)
                    image.save(thumbnail)
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
                snapshot=snapshots[-1],
                spec=spec,
                walls=walls,
                frame_number=len(timeline) - 1,
                total_frames=len(timeline),
            ).save(thumbnail)

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
                    str(output),
                ],
                check=True,
            )

    final = snapshots[-1]
    print(f"Rendered {output}")
    print(f"Rendered {thumbnail}")
    print(f"Algorithm: {spec.title}")
    print(f"Frames: {len(timeline)} at {args.fps} fps ({len(timeline) / args.fps:.1f}s)")
    print(f"Audio: {'on' if audio_enabled else 'off'}")
    print(f"Expanded: {final.expanded}")
    print(f"Edge checks: {final.edge_checks}")
    print(f"Discovered: {final.discovered}")
    print(f"Path length: {final.path_cost}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", choices=tuple(ALGORITHMS), default="astar")
    parser.add_argument("--all", action="store_true", help="render every configured pathfinding algorithm")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--thumbnail", type=Path, default=None)
    parser.add_argument("--width", type=int, default=1080)
    parser.add_argument("--height", type=int, default=1920)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--no-audio", action="store_true")
    args = parser.parse_args()
    if args.width <= 0 or args.height <= 0 or args.fps <= 0:
        parser.error("--width, --height, and --fps must be positive")
    if args.all and (args.output is not None or args.thumbnail is not None):
        parser.error("--output and --thumbnail can only be used for a single --algorithm render")
    return args


if __name__ == "__main__":
    parsed = parse_args()
    algorithms = tuple(ALGORITHMS) if parsed.all else (parsed.algorithm,)
    for selected_algorithm in algorithms:
        render_video(parsed, selected_algorithm)
