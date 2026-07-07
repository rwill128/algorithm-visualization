#!/usr/bin/env python3
"""Render Shorts-ready pathfinding cost-distribution videos."""

from __future__ import annotations

import argparse
import math
import random
import shutil
import subprocess
import tempfile
from contextlib import nullcontext
from dataclasses import dataclass
from heapq import heappop, heappush
from pathlib import Path

from PIL import Image, ImageDraw

import render_bubble_sort_short as bubble


ROOT = Path(__file__).resolve().parents[1]
SHORTS_DIR = ROOT / "artifacts" / "shorts"

Coord = tuple[int, int]

CELL_OPEN = (30, 37, 51)
WALL = (70, 78, 96)
EXPLORED = (58, 119, 203)
FRONTIER = (255, 202, 77)
CURRENT = (255, 104, 91)
PATH = (126, 220, 135)
START_COLOR = (86, 176, 255)
GOAL_COLOR = (251, 132, 76)
BEST = PATH
WORST = CURRENT
POINT = (86, 176, 255)


@dataclass(frozen=True)
class AlgorithmSpec:
    slug: str
    title: str
    subtitle: str
    footer: str


ALGORITHMS = {
    "astar": AlgorithmSpec("a_star", "A-STAR", "Heuristic cost narrows the search.", "A* with Manhattan heuristic"),
    "bfs": AlgorithmSpec("breadth_first", "BREADTH-FIRST", "A queue expands uniformly outward.", "BFS on unweighted grids"),
    "dfs": AlgorithmSpec("depth_first", "DEPTH-FIRST", "A stack commits to one route.", "DFS with stack order"),
    "dijkstra": AlgorithmSpec("dijkstra", "DIJKSTRA", "Known distance expands uniformly.", "Dijkstra on uniform-cost grids"),
    "greedy": AlgorithmSpec("greedy_best_first", "GREEDY BEST-FIRST", "Heuristic-only search chases the goal.", "Greedy best-first search"),
}


@dataclass(frozen=True)
class World:
    rows: int
    cols: int
    walls: frozenset[Coord]
    start: Coord
    goal: Coord

    @property
    def area(self) -> int:
        return self.rows * self.cols


@dataclass(frozen=True)
class SearchResult:
    world: World
    explored: frozenset[Coord]
    frontier: frozenset[Coord]
    path: tuple[Coord, ...]
    expanded: int
    edge_checks: int
    discovered: int
    path_length: int


@dataclass(frozen=True)
class Trial:
    result: SearchResult

    @property
    def area(self) -> int:
        return self.result.world.area

    @property
    def expanded(self) -> int:
        return self.result.expanded

    @property
    def edge_checks(self) -> int:
        return self.result.edge_checks


@dataclass(frozen=True)
class FrameState:
    mode: str
    example: SearchResult | None = None
    solved: bool = False
    visible_count: int = 0
    audio_event: str | None = None


def neighbors(world: World, cell: Coord) -> tuple[Coord, ...]:
    row, col = cell
    candidates = ((row - 1, col), (row, col + 1), (row + 1, col), (row, col - 1))
    return tuple(
        candidate
        for candidate in candidates
        if 0 <= candidate[0] < world.rows and 0 <= candidate[1] < world.cols
    )


def heuristic(left: Coord, right: Coord) -> int:
    return abs(left[0] - right[0]) + abs(left[1] - right[1])


def reconstruct(came_from: dict[Coord, Coord], current: Coord) -> tuple[Coord, ...]:
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return tuple(path)


def frontier_priority(algorithm: str, world: World, cell: Coord, g_cost: int, counter: int) -> tuple[int, int, int, Coord]:
    h_cost = heuristic(cell, world.goal)
    if algorithm == "astar":
        return g_cost + h_cost, h_cost, counter, cell
    if algorithm == "dijkstra":
        return g_cost, 0, counter, cell
    if algorithm == "greedy":
        return h_cost, g_cost, counter, cell
    if algorithm == "bfs":
        return g_cost, counter, 0, cell
    raise ValueError(f"{algorithm} does not use heap priority")


def push_frontier(
    container: list[Coord] | list[tuple[int, int, int, Coord]],
    algorithm: str,
    world: World,
    cell: Coord,
    g_cost: int,
    counter: int,
) -> None:
    if algorithm == "dfs":
        container.append(cell)  # type: ignore[arg-type]
    else:
        heappush(container, frontier_priority(algorithm, world, cell, g_cost, counter))  # type: ignore[arg-type]


def pop_frontier(container: list[Coord] | list[tuple[int, int, int, Coord]], algorithm: str) -> Coord:
    if algorithm == "dfs":
        return container.pop()  # type: ignore[return-value]
    return heappop(container)[3]  # type: ignore[arg-type]


def run_search(world: World, algorithm: str) -> SearchResult:
    frontier_heap: list[tuple[int, int, int, Coord]] = []
    frontier_stack: list[Coord] = []
    container: list[Coord] | list[tuple[int, int, int, Coord]] = frontier_stack if algorithm == "dfs" else frontier_heap
    frontier: set[Coord] = {world.start}
    explored: set[Coord] = set()
    came_from: dict[Coord, Coord] = {}
    g_score: dict[Coord, int] = {world.start: 0}
    counter = 0
    edge_checks = 0
    push_frontier(container, algorithm, world, world.start, 0, counter)

    while container:
        current = pop_frontier(container, algorithm)
        if current in explored or current not in frontier:
            continue
        frontier.remove(current)
        explored.add(current)

        if current == world.goal:
            path = reconstruct(came_from, current)
            return SearchResult(
                world,
                frozenset(explored),
                frozenset(frontier),
                path,
                len(explored),
                edge_checks,
                len(g_score),
                len(path) - 1,
            )

        candidate_neighbors = neighbors(world, current)
        if algorithm == "dfs":
            candidate_neighbors = tuple(reversed(candidate_neighbors))

        for neighbor in candidate_neighbors:
            edge_checks += 1
            if neighbor in world.walls or neighbor in explored:
                continue
            tentative = g_score[current] + 1
            if algorithm in {"bfs", "dfs"}:
                if neighbor in g_score:
                    continue
            elif tentative >= g_score.get(neighbor, 1_000_000):
                continue
            came_from[neighbor] = current
            g_score[neighbor] = tentative
            frontier.add(neighbor)
            counter += 1
            push_frontier(container, algorithm, world, neighbor, tentative, counter)

    raise RuntimeError("generated world unexpectedly had no route")


def start_goal_pair(rows: int, cols: int, rng: random.Random) -> tuple[Coord, Coord]:
    pairs = (
        ((rows - 2, 1), (1, cols - 2)),
        ((1, 1), (rows - 2, cols - 2)),
        ((1, cols - 2), (rows - 2, 1)),
        ((rows - 2, cols - 2), (1, 1)),
        ((rows // 2, 1), (rows // 2, cols - 2)),
        ((rows - 2, cols // 2), (1, cols // 2)),
    )
    return rng.choice(pairs)


def add_recursive_barriers(
    walls: set[Coord],
    rng: random.Random,
    *,
    row0: int,
    row1: int,
    col0: int,
    col1: int,
    depth: int,
) -> None:
    if depth <= 0 or row1 - row0 < 4 or col1 - col0 < 4:
        return

    height = row1 - row0 + 1
    width = col1 - col0 + 1
    horizontal = height > width or (height == width and rng.random() < 0.5)
    if horizontal:
        wall_row = rng.randint(row0 + 1, row1 - 1)
        passages = {rng.randint(col0, col1)}
        if width >= 12 and rng.random() < 0.18:
            passages.add(rng.randint(col0, col1))
        for col in range(col0, col1 + 1):
            if col not in passages:
                walls.add((wall_row, col))
        add_recursive_barriers(walls, rng, row0=row0, row1=wall_row - 1, col0=col0, col1=col1, depth=depth - 1)
        add_recursive_barriers(walls, rng, row0=wall_row + 1, row1=row1, col0=col0, col1=col1, depth=depth - 1)
    else:
        wall_col = rng.randint(col0 + 1, col1 - 1)
        passages = {rng.randint(row0, row1)}
        if height >= 12 and rng.random() < 0.18:
            passages.add(rng.randint(row0, row1))
        for row in range(row0, row1 + 1):
            if row not in passages:
                walls.add((row, wall_col))
        add_recursive_barriers(walls, rng, row0=row0, row1=row1, col0=col0, col1=wall_col - 1, depth=depth - 1)
        add_recursive_barriers(walls, rng, row0=row0, row1=row1, col0=wall_col + 1, col1=col1, depth=depth - 1)


def add_segment_barriers(walls: set[Coord], rng: random.Random, rows: int, cols: int, density: float) -> None:
    segment_count = max(2, int(rows * cols * density / 42))
    max_length = max(4, min(rows, cols) // 2)
    for _ in range(segment_count):
        horizontal = rng.random() < 0.5
        length = rng.randint(3, max_length)
        if horizontal:
            row = rng.randint(2, rows - 3)
            col0 = rng.randint(1, max(1, cols - length - 1))
            for col in range(col0, min(cols - 1, col0 + length)):
                walls.add((row, col))
        else:
            col = rng.randint(2, cols - 3)
            row0 = rng.randint(1, max(1, rows - length - 1))
            for row in range(row0, min(rows - 1, row0 + length)):
                walls.add((row, col))


def build_structured_world(rows: int, cols: int, rng: random.Random, density: float) -> World:
    start, goal = start_goal_pair(rows, cols, rng)
    walls: set[Coord] = set()
    for row in range(rows):
        for col in range(cols):
            if row in {0, rows - 1} or col in {0, cols - 1}:
                walls.add((row, col))

    depth = rng.randint(4, 7) + int(density * 5)
    add_recursive_barriers(walls, rng, row0=1, row1=rows - 2, col0=1, col1=cols - 2, depth=depth)
    add_segment_barriers(walls, rng, rows, cols, density)

    walls.discard(start)
    walls.discard(goal)
    return World(rows, cols, frozenset(walls), start, goal)


def accept_world(world: World) -> bool:
    result = run_search(world, "bfs")
    direct = heuristic(world.start, world.goal)
    detour = result.path_length - direct
    area = world.area
    open_cells = area - len(world.walls)
    if open_cells < max(8, area * 0.3):
        return False
    return detour >= max(3, int(math.sqrt(area) * 0.25)) or result.expanded >= open_cells * 0.28


def random_world(rng: random.Random, min_side: int, max_side: int, density_min: float, density_max: float) -> World:
    fallback: World | None = None
    for attempt in range(80):
        base = rng.randint(min_side, max_side)
        rows = max(min_side, min(max_side, base + rng.randint(-4, 4)))
        cols = max(min_side, min(max_side, base + rng.randint(-4, 4)))
        density = rng.uniform(density_min, density_max)
        world = build_structured_world(rows, cols, rng, density)
        try:
            run_search(world, "bfs")
        except RuntimeError:
            continue
        fallback = world
        if accept_world(world) or attempt >= 24:
            return world
    if fallback is None:
        raise RuntimeError("could not generate a solvable random world")
    return fallback


def generate_trials(
    *,
    algorithm: str,
    samples: int,
    seed: int,
    min_side: int,
    max_side: int,
    density_min: float,
    density_max: float,
) -> list[Trial]:
    rng = random.Random(seed)
    trials: list[Trial] = []
    while len(trials) < samples:
        world = random_world(rng, min_side, max_side, density_min, density_max)
        trials.append(Trial(run_search(world, algorithm)))
    return trials


def metric_value(trial: Trial, metric: str) -> int:
    if metric == "edge-checks":
        return trial.edge_checks
    return trial.expanded


def open_grid_reference_bound(area: int, metric: str) -> int:
    cells = max(2, int(2 * math.sqrt(area) - 5))
    if metric == "edge-checks":
        return max(1, cells * 2)
    return cells


def worst_bound(area: int, metric: str) -> int:
    if metric == "edge-checks":
        return area * 4
    return area


def nice_axis_max(value: int) -> int:
    target_step = max(1, value / 5)
    magnitude = 10 ** math.floor(math.log10(target_step))
    normalized = target_step / magnitude
    if normalized <= 1:
        step = magnitude
    elif normalized <= 2:
        step = 2 * magnitude
    elif normalized <= 2.5:
        step = 2.5 * magnitude
    elif normalized <= 3:
        step = 3 * magnitude
    elif normalized <= 5:
        step = 5 * magnitude
    else:
        step = 10 * magnitude
    return int(step * 5)


def format_count(value: int) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if value >= 1_000:
        return f"{value / 1_000:.1f}K" if value % 1000 else f"{value // 1000}K"
    return str(value)


def graph_point(
    area: float,
    cost: float,
    *,
    plot_left: int,
    plot_right: int,
    plot_top: int,
    plot_bottom: int,
    max_area: int,
    max_cost: int,
) -> tuple[float, float]:
    x = bubble.lerp(plot_left, plot_right, area / max_area)
    y = bubble.lerp(plot_bottom, plot_top, cost / max_cost)
    return x, y


def curve_points(
    bound_fn,
    *,
    plot_left: int,
    plot_right: int,
    plot_top: int,
    plot_bottom: int,
    max_area: int,
    max_cost: int,
    metric: str,
) -> list[tuple[float, float]]:
    points = []
    step = max(10, max_area // 80)
    for area in range(1, max_area + 1, step):
        points.append(
            graph_point(
                area,
                bound_fn(area, metric),
                plot_left=plot_left,
                plot_right=plot_right,
                plot_top=plot_top,
                plot_bottom=plot_bottom,
                max_area=max_area,
                max_cost=max_cost,
            )
        )
    points.append(
        graph_point(
            max_area,
            bound_fn(max_area, metric),
            plot_left=plot_left,
            plot_right=plot_right,
            plot_top=plot_top,
            plot_bottom=plot_bottom,
            max_area=max_area,
            max_cost=max_cost,
        )
    )
    return points


def draw_grid(
    draw: ImageDraw.ImageDraw,
    result: SearchResult,
    box: tuple[int, int, int, int],
    *,
    solved: bool,
    show_labels: bool = True,
) -> None:
    x0, y0, x1, y1 = box
    world = result.world
    cell = max(3, min((x1 - x0) // world.cols, (y1 - y0) // world.rows))
    grid_w = cell * world.cols
    grid_h = cell * world.rows
    left = x0 + (x1 - x0 - grid_w) // 2
    top = y0 + (y1 - y0 - grid_h) // 2
    bubble.rounded_rect(draw, (left - 10, top - 10, left + grid_w + 10, top + grid_h + 10), 18, bubble.PANEL, bubble.GRID, 2)
    path = set(result.path) if solved else set()
    explored = result.explored if solved else frozenset()
    frontier = result.frontier if solved else frozenset()
    for row in range(world.rows):
        for col in range(world.cols):
            coord = (row, col)
            color = CELL_OPEN
            if coord in world.walls:
                color = WALL
            elif coord in path:
                color = PATH
            elif coord == world.start:
                color = START_COLOR
            elif coord == world.goal:
                color = GOAL_COLOR
            elif coord in frontier:
                color = FRONTIER
            elif coord in explored:
                color = EXPLORED
            cx0 = left + col * cell + max(1, cell // 12)
            cy0 = top + row * cell + max(1, cell // 12)
            cx1 = left + (col + 1) * cell - max(1, cell // 12)
            cy1 = top + (row + 1) * cell - max(1, cell // 12)
            bubble.rounded_rect(draw, (cx0, cy0, cx1, cy1), max(2, cell // 8), color)
    if show_labels and cell >= 15:
        label_font = bubble.font(max(10, int(cell * 0.48)), bold=True)
        for coord, label in ((world.start, "S"), (world.goal, "G")):
            row, col = coord
            lx = left + col * cell
            ly = top + row * cell
            draw.text(
                (lx + (cell - bubble.text_width(draw, label, label_font)) / 2, ly + int(cell * 0.18)),
                label,
                font=label_font,
                fill=bubble.TEXT,
            )


def draw_example_frame(
    *,
    width: int,
    height: int,
    spec: AlgorithmSpec,
    result: SearchResult,
    solved: bool,
    frame_number: int,
    total_frames: int,
) -> Image.Image:
    image = bubble.gradient_background(width, height).copy()
    draw = ImageDraw.Draw(image)
    margin = int(width * 0.065)
    title = f"{spec.title} RANDOM WORLDS"
    title_font = bubble.fit_font(draw, title, int(width * 0.07), int(width * 0.9), bold=True, min_size=int(width * 0.045))
    subtitle_font = bubble.font(int(width * 0.032))
    label_font = bubble.font(int(width * 0.031), bold=True)
    draw.text(((width - bubble.text_width(draw, title, title_font)) / 2, int(height * 0.045)), title, font=title_font, fill=bubble.TEXT)
    bubble.draw_centered_text(draw, int(height * 0.096), "Same algorithm, many random wall layouts.", subtitle_font, bubble.MUTED, width)

    badge_y = int(height * 0.145)
    badge_h = int(height * 0.052)
    bubble.rounded_rect(draw, (margin, badge_y, width - margin, badge_y + badge_h), 18, bubble.PANEL, bubble.GRID, 2)
    status = "unresolved grid" if not solved else "resolved search"
    draw.text((margin + 24, badge_y + int(badge_h * 0.22)), status, font=label_font, fill=FRONTIER if not solved else PATH)
    size_text = f"{result.world.rows}x{result.world.cols}  area {result.world.area}"
    draw.text((width - margin - 24 - bubble.text_width(draw, size_text, label_font), badge_y + int(badge_h * 0.22)), size_text, font=label_font, fill=bubble.MUTED)

    draw_grid(draw, result, (margin, int(height * 0.235), width - margin, int(height * 0.735)), solved=solved)

    stats_y = int(height * 0.785)
    stats_h = int(height * 0.06)
    gap = int(width * 0.024)
    card_w = int((width - 2 * margin - 2 * gap) / 3)
    stats = (
        ("expanded", result.expanded, EXPLORED),
        ("edge checks", result.edge_checks, FRONTIER),
        ("path length", result.path_length, PATH),
    )
    value_font = bubble.font(int(width * 0.038), bold=True)
    small_font = bubble.font(int(width * 0.024))
    for index, (label, value, color) in enumerate(stats):
        x0 = margin + index * (card_w + gap)
        bubble.rounded_rect(draw, (x0, stats_y, x0 + card_w, stats_y + stats_h), 16, bubble.PANEL, bubble.GRID, 2)
        shown = str(value) if solved else "-"
        draw.text((x0 + 18, stats_y + int(stats_h * 0.14)), shown, font=value_font, fill=color if solved else bubble.MUTED)
        draw.text((x0 + 18, stats_y + int(stats_h * 0.58)), label, font=small_font, fill=bubble.MUTED)

    footer_font = bubble.font(int(width * 0.02))
    draw.text((margin, int(height * 0.965)), f"frame {frame_number + 1}/{total_frames} | {spec.footer}", font=footer_font, fill=bubble.MUTED)
    return image


def draw_graph_frame(
    *,
    width: int,
    height: int,
    spec: AlgorithmSpec,
    trials: list[Trial],
    visible_count: int,
    metric: str,
    max_area: int,
    max_cost: int,
    frame_number: int,
    total_frames: int,
) -> Image.Image:
    image = bubble.gradient_background(width, height).copy()
    draw = ImageDraw.Draw(image)
    margin = int(width * 0.065)
    title = f"{spec.title} COST DISTRIBUTION"
    title_font = bubble.fit_font(draw, title, int(width * 0.072), int(width * 0.9), bold=True, min_size=int(width * 0.045))
    subtitle_font = bubble.font(int(width * 0.032))
    small_font = bubble.font(int(width * 0.023))
    label_font = bubble.font(int(width * 0.03), bold=True)
    draw.text(((width - bubble.text_width(draw, title, title_font)) / 2, int(height * 0.045)), title, font=title_font, fill=bubble.TEXT)
    metric_label = "expanded cells" if metric == "expanded" else "edge checks"
    bubble.draw_centered_text(draw, int(height * 0.096), f"Random wall worlds plotted by area vs {metric_label}.", subtitle_font, bubble.MUTED, width)

    plot_left = int(width * 0.15)
    plot_right = int(width * 0.92)
    plot_top = int(height * 0.18)
    plot_bottom = int(height * 0.78)
    draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill=bubble.GRID, width=3)
    draw.line((plot_left, plot_top, plot_left, plot_bottom), fill=bubble.GRID, width=3)

    for tick in range(6):
        value = int(max_cost * tick / 5)
        y = bubble.lerp(plot_bottom, plot_top, tick / 5)
        draw.line((plot_left - 8, y, plot_right, y), fill=bubble.GRID, width=1)
        label = format_count(value)
        draw.text((plot_left - 18 - bubble.text_width(draw, label, small_font), y - 10), label, font=small_font, fill=bubble.MUTED)

    for tick in range(5):
        area = int(max_area * tick / 4)
        x = bubble.lerp(plot_left, plot_right, tick / 4)
        draw.line((x, plot_bottom, x, plot_bottom + 8), fill=bubble.GRID, width=2)
        label = format_count(area)
        draw.text((x - bubble.text_width(draw, label, small_font) / 2, plot_bottom + 16), label, font=small_font, fill=bubble.MUTED)

    worst = curve_points(worst_bound, plot_left=plot_left, plot_right=plot_right, plot_top=plot_top, plot_bottom=plot_bottom, max_area=max_area, max_cost=max_cost, metric=metric)
    best = curve_points(open_grid_reference_bound, plot_left=plot_left, plot_right=plot_right, plot_top=plot_top, plot_bottom=plot_bottom, max_area=max_area, max_cost=max_cost, metric=metric)
    draw.line(worst, fill=WORST, width=5)
    draw.line(best, fill=BEST, width=5)

    for trial in trials[:visible_count]:
        x, y = graph_point(
            trial.area,
            metric_value(trial, metric),
            plot_left=plot_left,
            plot_right=plot_right,
            plot_top=plot_top,
            plot_bottom=plot_bottom,
            max_area=max_area,
            max_cost=max_cost,
        )
        radius = 4 if trial is not trials[min(visible_count - 1, len(trials) - 1)] else 8
        color = POINT
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)

    axis_font = bubble.font(int(width * 0.025), bold=True)
    draw.text((plot_left, plot_bottom + 56), "grid area", font=axis_font, fill=bubble.MUTED)
    draw.text((plot_left, plot_top - 42), metric_label, font=axis_font, fill=bubble.MUTED)
    draw.text((plot_left + 12, plot_top + 10), "worst: all cells", font=label_font, fill=WORST)
    draw.text((plot_left + 12, plot_top + 48), "open-grid reference", font=label_font, fill=BEST)

    panel_y = int(height * 0.835)
    bubble.rounded_rect(draw, (margin, panel_y, width - margin, panel_y + int(height * 0.075)), 18, bubble.PANEL, bubble.GRID, 2)
    if visible_count > 0:
        current = trials[min(visible_count - 1, len(trials) - 1)]
        summary = f"area {current.area}   exp {current.expanded}   edges {current.edge_checks}   path {current.result.path_length}"
    else:
        summary = "sampling random solvable grid worlds"
    summary_font = bubble.fit_font(draw, summary, int(width * 0.03), int(width * 0.5), bold=True, min_size=int(width * 0.018))
    draw.text((margin + 24, panel_y + int(height * 0.023)), summary, font=summary_font, fill=POINT)

    count_text = f"samples: {visible_count}/{len(trials)}"
    draw.text((width - margin - 24 - bubble.text_width(draw, count_text, label_font), panel_y + int(height * 0.021)), count_text, font=label_font, fill=bubble.MUTED)

    footer_font = bubble.font(int(width * 0.02))
    draw.text((margin, int(height * 0.965)), f"frame {frame_number + 1}/{total_frames} | {spec.footer}", font=footer_font, fill=bubble.MUTED)
    return image


def planned_frames(examples: list[SearchResult], trials: list[Trial], fps: int, graph_seconds: float) -> list[FrameState]:
    timeline: list[FrameState] = []
    for result in examples:
        timeline.extend(FrameState("example", result, False) for _ in range(int(fps * 0.5)))
        timeline.extend(FrameState("example", result, True, audio_event="lock" if frame == 0 else None) for frame in range(int(fps * 0.85)))
    timeline.extend(FrameState("graph", visible_count=0) for _ in range(int(fps * 0.75)))
    reveal_frames = max(1, int(fps * graph_seconds))
    for frame in range(reveal_frames):
        visible = min(len(trials), 1 + int((frame / max(1, reveal_frames - 1)) * len(trials)))
        event = "swap" if frame % max(1, fps // 8) == 0 else None
        timeline.append(FrameState("graph", visible_count=visible, audio_event=event))
    timeline.extend(FrameState("graph", visible_count=len(trials), audio_event="sorted" if frame == 0 else None) for frame in range(int(fps * 2.0)))
    return timeline


def default_output_for(spec: AlgorithmSpec, metric: str) -> Path:
    suffix = "expanded" if metric == "expanded" else "edge_checks"
    return SHORTS_DIR / f"{spec.slug}_distribution_{suffix}.mp4"


def default_thumbnail_for(spec: AlgorithmSpec, metric: str) -> Path:
    suffix = "expanded" if metric == "expanded" else "edge_checks"
    return SHORTS_DIR / f"{spec.slug}_distribution_{suffix}_thumbnail.png"


def render_video(args: argparse.Namespace, algorithm: str) -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required but was not found on PATH")
    spec = ALGORITHMS[algorithm]
    output = args.output or default_output_for(spec, args.metric)
    thumbnail = args.thumbnail or default_thumbnail_for(spec, args.metric)
    output.parent.mkdir(parents=True, exist_ok=True)

    trials = generate_trials(
        algorithm=algorithm,
        samples=args.samples,
        seed=args.seed,
        min_side=args.min_side,
        max_side=args.max_side,
        density_min=args.density_min,
        density_max=args.density_max,
    )
    examples = [trial.result for trial in trials[: args.examples]]
    graph_trials = trials[args.examples :]
    max_area = args.max_side * args.max_side
    max_cost = nice_axis_max(max(worst_bound(max_area, args.metric), max(metric_value(trial, args.metric) for trial in graph_trials)))
    timeline = planned_frames(examples, graph_trials, args.fps, args.graph_seconds)
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
                if frame_state.mode == "example":
                    assert frame_state.example is not None
                    image = draw_example_frame(
                        width=args.width,
                        height=args.height,
                        spec=spec,
                        result=frame_state.example,
                        solved=frame_state.solved,
                        frame_number=frame_number,
                        total_frames=len(timeline),
                    )
                else:
                    image = draw_graph_frame(
                        width=args.width,
                        height=args.height,
                        spec=spec,
                        trials=graph_trials,
                        visible_count=frame_state.visible_count,
                        metric=args.metric,
                        max_area=max_area,
                        max_cost=max_cost,
                        frame_number=frame_number,
                        total_frames=len(timeline),
                    )
                if not thumbnail_saved and frame_state.mode == "graph" and frame_state.visible_count >= max(1, len(graph_trials) // 2):
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
            draw_graph_frame(
                width=args.width,
                height=args.height,
                spec=spec,
                trials=graph_trials,
                visible_count=len(graph_trials),
                metric=args.metric,
                max_area=max_area,
                max_cost=max_cost,
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

    values = [metric_value(trial, args.metric) for trial in graph_trials]
    print(f"Rendered {output}")
    print(f"Rendered {thumbnail}")
    print(f"Algorithm: {spec.title}")
    print(f"Frames: {len(timeline)} at {args.fps} fps ({len(timeline) / args.fps:.1f}s)")
    print(f"Audio: {'on' if audio_enabled else 'off'}")
    print(f"Metric: {args.metric}")
    print(f"Samples: {len(graph_trials)} graph + {len(examples)} examples")
    print(f"Expanded range: {min(trial.expanded for trial in graph_trials)}..{max(trial.expanded for trial in graph_trials)}")
    print(f"Edge-check range: {min(trial.edge_checks for trial in graph_trials)}..{max(trial.edge_checks for trial in graph_trials)}")
    print(f"Plotted metric range: {min(values)}..{max(values)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", choices=tuple(ALGORITHMS), default="astar")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--metric", choices=("expanded", "edge-checks"), default="expanded")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--thumbnail", type=Path, default=None)
    parser.add_argument("--width", type=int, default=1080)
    parser.add_argument("--height", type=int, default=1920)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--samples", type=int, default=520)
    parser.add_argument("--examples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--min-side", type=int, default=8)
    parser.add_argument("--max-side", type=int, default=38)
    parser.add_argument("--density-min", type=float, default=0.12)
    parser.add_argument("--density-max", type=float, default=0.34)
    parser.add_argument("--graph-seconds", type=float, default=8.0)
    parser.add_argument("--no-audio", action="store_true")
    args = parser.parse_args()
    if args.width <= 0 or args.height <= 0 or args.fps <= 0:
        parser.error("--width, --height, and --fps must be positive")
    if args.samples <= args.examples:
        parser.error("--samples must be greater than --examples")
    if args.examples < 1:
        parser.error("--examples must be at least 1")
    if args.min_side < 5 or args.max_side < args.min_side:
        parser.error("--min-side must be at least 5 and --max-side must be >= --min-side")
    if not 0 <= args.density_min <= args.density_max < 0.8:
        parser.error("wall densities must satisfy 0 <= min <= max < 0.8")
    if args.all and (args.output is not None or args.thumbnail is not None):
        parser.error("--output and --thumbnail can only be used for a single algorithm render")
    return args


if __name__ == "__main__":
    parsed = parse_args()
    selected = tuple(ALGORITHMS) if parsed.all else (parsed.algorithm,)
    for algorithm_name in selected:
        render_video(parsed, algorithm_name)
