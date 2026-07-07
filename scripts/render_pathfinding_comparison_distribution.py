#!/usr/bin/env python3
"""Render a shared-axis pathfinding cost-distribution comparison short."""

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
import render_pathfinding_distribution as dist


ROOT = Path(__file__).resolve().parents[1]
SHORTS_DIR = ROOT / "artifacts" / "shorts"
DEFAULT_OUTPUT = SHORTS_DIR / "pathfinding_algorithm_comparison_distribution.mp4"
DEFAULT_THUMBNAIL = SHORTS_DIR / "pathfinding_algorithm_comparison_distribution_thumbnail.png"

ALGORITHM_ORDER = ("astar", "bfs", "dfs", "dijkstra", "greedy")

COMPLETED_POINT = (80, 88, 102)
COMPLETED_LINE = (70, 78, 92)
CURRENT_POINT = (86, 176, 255)
CURRENT_ACCENT = (255, 202, 77)
FINAL_COLORS = {
    "astar": (86, 176, 255),
    "bfs": (126, 220, 135),
    "dfs": (183, 132, 255),
    "dijkstra": (255, 202, 77),
    "greedy": (255, 104, 91),
}


@dataclass(frozen=True)
class AlgorithmPlot:
    key: str
    spec: dist.AlgorithmSpec
    examples: list[dist.SearchResult]
    trials: list[dist.Trial]
    points: list[tuple[float, float]]


@dataclass(frozen=True)
class FrameState:
    mode: str
    algorithm_index: int
    example: dist.SearchResult | None = None
    solved: bool = False
    visible_count: int = 0
    final_hold: bool = False
    audio_event: str | None = None


def build_plots(
    *,
    metric: str,
    samples: int,
    examples: int,
    seed: int,
    min_side: int,
    max_side: int,
    density_min: float,
    density_max: float,
    plot_left: int,
    plot_right: int,
    plot_top: int,
    plot_bottom: int,
    max_area: int,
    max_cost: int,
) -> list[AlgorithmPlot]:
    plots: list[AlgorithmPlot] = []
    for algorithm in ALGORITHM_ORDER:
        trials = dist.generate_trials(
            algorithm=algorithm,
            samples=samples,
            seed=seed,
            min_side=min_side,
            max_side=max_side,
            density_min=density_min,
            density_max=density_max,
        )
        graph_trials = trials[examples:]
        points = [
            dist.graph_point(
                trial.area,
                dist.metric_value(trial, metric),
                plot_left=plot_left,
                plot_right=plot_right,
                plot_top=plot_top,
                plot_bottom=plot_bottom,
                max_area=max_area,
                max_cost=max_cost,
            )
            for trial in graph_trials
        ]
        plots.append(
            AlgorithmPlot(
                key=algorithm,
                spec=dist.ALGORITHMS[algorithm],
                examples=[trial.result for trial in trials[:examples]],
                trials=graph_trials,
                points=points,
            )
        )
    return plots


def draw_axes(
    draw: ImageDraw.ImageDraw,
    *,
    width: int,
    plot_left: int,
    plot_right: int,
    plot_top: int,
    plot_bottom: int,
    max_area: int,
    max_cost: int,
    metric: str,
) -> None:
    small_font = bubble.font(int(width * 0.023))
    axis_font = bubble.font(int(width * 0.025), bold=True)
    bubble.rounded_rect(draw, (plot_left - 18, plot_top - 18, plot_right + 18, plot_bottom + 18), 8, (12, 15, 22), bubble.GRID, 2)

    for tick in range(6):
        value = int(max_cost * tick / 5)
        y = bubble.lerp(plot_bottom, plot_top, tick / 5)
        draw.line((plot_left, y, plot_right, y), fill=bubble.GRID, width=2)
        label = dist.format_count(value)
        draw.text((plot_left - 18 - bubble.text_width(draw, label, small_font), y - 11), label, font=small_font, fill=bubble.MUTED)

    for tick in range(5):
        area = int(max_area * tick / 4)
        x = bubble.lerp(plot_left, plot_right, tick / 4)
        draw.line((x, plot_top, x, plot_bottom), fill=(29, 35, 47), width=1)
        label = dist.format_count(area)
        draw.text((x - bubble.text_width(draw, label, small_font) / 2, plot_bottom + 22), label, font=small_font, fill=bubble.MUTED)

    draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill=bubble.MUTED, width=3)
    draw.line((plot_left, plot_top, plot_left, plot_bottom), fill=bubble.MUTED, width=3)
    metric_label = "expanded cells" if metric == "expanded" else "edge checks"
    draw.text((plot_left + 14, plot_bottom + 60), "grid area", font=axis_font, fill=bubble.MUTED)
    draw.text((plot_left, plot_top - 42), metric_label, font=axis_font, fill=bubble.MUTED)


def draw_bounds(
    draw: ImageDraw.ImageDraw,
    *,
    plot_left: int,
    plot_right: int,
    plot_top: int,
    plot_bottom: int,
    max_area: int,
    max_cost: int,
    metric: str,
    active: bool,
) -> None:
    worst_color = dist.WORST if active else COMPLETED_LINE
    best_color = dist.BEST if active else (74, 83, 94)
    worst = dist.curve_points(
        dist.worst_bound,
        plot_left=plot_left,
        plot_right=plot_right,
        plot_top=plot_top,
        plot_bottom=plot_bottom,
        max_area=max_area,
        max_cost=max_cost,
        metric=metric,
    )
    best = dist.curve_points(
        dist.open_grid_reference_bound,
        plot_left=plot_left,
        plot_right=plot_right,
        plot_top=plot_top,
        plot_bottom=plot_bottom,
        max_area=max_area,
        max_cost=max_cost,
        metric=metric,
    )
    draw.line(worst, fill=worst_color, width=5 if active else 3)
    draw.line(best, fill=best_color, width=5 if active else 3)


def draw_completed_points(draw: ImageDraw.ImageDraw, plot: AlgorithmPlot, *, final_hold: bool) -> None:
    color = FINAL_COLORS[plot.key] if final_hold else COMPLETED_POINT
    radius = 3 if final_hold else 2
    for x, y in plot.points:
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)


def draw_current_points(draw: ImageDraw.ImageDraw, plot: AlgorithmPlot, visible_count: int) -> None:
    for x, y in plot.points[: max(0, visible_count - 1)]:
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=CURRENT_POINT)
    if visible_count:
        x, y = plot.points[visible_count - 1]
        draw.ellipse((x - 9, y - 9, x + 9, y + 9), fill=CURRENT_ACCENT)
        draw.ellipse((x - 15, y - 15, x + 15, y + 15), outline=CURRENT_ACCENT, width=3)


def draw_legend(
    draw: ImageDraw.ImageDraw,
    *,
    plots: list[AlgorithmPlot],
    width: int,
    y: int,
) -> None:
    small_font = bubble.font(int(width * 0.02), bold=True)
    margin = int(width * 0.065)
    x = margin
    for plot in plots:
        label = plot.spec.title.replace("BREADTH-FIRST", "BFS").replace("DEPTH-FIRST", "DFS").replace("GREEDY BEST-FIRST", "GREEDY")
        draw.ellipse((x, y + 7, x + 16, y + 23), fill=FINAL_COLORS[plot.key])
        draw.text((x + 24, y), label, font=small_font, fill=bubble.MUTED)
        x += 24 + bubble.text_width(draw, label, small_font) + 28


def draw_cards(
    draw: ImageDraw.ImageDraw,
    *,
    width: int,
    height: int,
    plot: AlgorithmPlot | None,
    visible_count: int,
    completed_count: int,
    max_area: int,
) -> None:
    margin = int(width * 0.065)
    label_font = bubble.font(int(width * 0.032), bold=True)
    small_font = bubble.font(int(width * 0.024))
    panel_y = int(height * 0.835)
    panel_h = int(height * 0.073)
    gap = int(width * 0.022)
    panel_w = int((width - margin * 2 - gap * 2) / 3)

    current = plot.spec.title if plot else "ALL SEARCHES"
    if current == "GREEDY BEST-FIRST":
        current = "GREEDY"
    stats = (
        ("current", current, CURRENT_ACCENT),
        ("samples", str(visible_count if plot else completed_count), CURRENT_POINT),
        ("max area", str(max_area), bubble.TEXT),
    )
    for index, (label, value, color) in enumerate(stats):
        x0 = margin + index * (panel_w + gap)
        bubble.rounded_rect(draw, (x0, panel_y, x0 + panel_w, panel_y + panel_h), 16, bubble.PANEL, bubble.GRID, 2)
        value_font = bubble.fit_font(draw, value, int(width * 0.032), panel_w - 34, bold=True, min_size=int(width * 0.021))
        draw.text((x0 + 17, panel_y + 12), value, font=value_font, fill=color)
        draw.text((x0 + 17, panel_y + int(panel_h * 0.58)), label, font=small_font, fill=bubble.MUTED)


def draw_graph_frame(
    *,
    width: int,
    height: int,
    plots: list[AlgorithmPlot],
    current_index: int,
    visible_count: int,
    metric: str,
    max_area: int,
    max_cost: int,
    progress: float,
    final_hold: bool,
) -> Image.Image:
    image = bubble.gradient_background(width, height).copy()
    draw = ImageDraw.Draw(image)
    margin = int(width * 0.065)
    title = "PATHFINDING COST DISTRIBUTION"
    title_font = bubble.fit_font(draw, title, int(width * 0.064), int(width * 0.9), bold=True, min_size=int(width * 0.045))
    subtitle_font = bubble.font(int(width * 0.031))
    draw.text(((width - bubble.text_width(draw, title, title_font)) / 2, int(height * 0.045)), title, font=title_font, fill=bubble.TEXT)
    if final_hold:
        subtitle = "All algorithms on one shared random-world scale."
    else:
        subtitle = f"{plots[current_index].spec.title}: old distributions grey out while new samples land."
    bubble.draw_centered_text(draw, int(height * 0.096), subtitle, subtitle_font, bubble.MUTED, width)

    plot_left = int(width * 0.15)
    plot_right = int(width * 0.92)
    plot_top = int(height * 0.18)
    plot_bottom = int(height * 0.78)
    draw_axes(
        draw,
        width=width,
        plot_left=plot_left,
        plot_right=plot_right,
        plot_top=plot_top,
        plot_bottom=plot_bottom,
        max_area=max_area,
        max_cost=max_cost,
        metric=metric,
    )
    draw_bounds(
        draw,
        plot_left=plot_left,
        plot_right=plot_right,
        plot_top=plot_top,
        plot_bottom=plot_bottom,
        max_area=max_area,
        max_cost=max_cost,
        metric=metric,
        active=not final_hold,
    )

    completed_range = range(len(plots)) if final_hold else range(current_index)
    for index in completed_range:
        draw_completed_points(draw, plots[index], final_hold=final_hold)

    current_plot = None if final_hold else plots[current_index]
    if current_plot is not None:
        draw_current_points(draw, current_plot, visible_count)
        label_font = bubble.font(int(width * 0.027), bold=True)
        label = current_plot.spec.title
        draw.text((plot_right - bubble.text_width(draw, label, label_font) - 10, plot_top + 12), label, font=label_font, fill=CURRENT_ACCENT)
        bound_font = bubble.font(int(width * 0.027), bold=True)
        draw.text((plot_left + 12, plot_top + 12), "worst: all cells", font=bound_font, fill=dist.WORST)
        draw.text((plot_left + 12, plot_top + 50), "open-grid reference", font=bound_font, fill=dist.BEST)

    if final_hold:
        draw_legend(draw, plots=plots, width=width, y=int(height * 0.135))

    draw_cards(
        draw,
        width=width,
        height=height,
        plot=current_plot,
        visible_count=visible_count,
        completed_count=sum(len(plot.trials) for plot in plots),
        max_area=max_area,
    )

    bar_x0 = margin
    bar_x1 = width - margin
    bar_y = int(height * 0.945)
    bar_h = max(10, int(height * 0.01))
    inset = max(2, min(4, bar_h // 3))
    bubble.rounded_rect(draw, (bar_x0, bar_y, bar_x1, bar_y + bar_h), 12, (15, 18, 25), bubble.GRID, 2)
    bubble.rounded_rect(
        draw,
        (bar_x0 + inset, bar_y + inset, bubble.lerp(bar_x0 + inset, bar_x1 - inset, progress), bar_y + bar_h - inset),
        8,
        CURRENT_POINT,
    )
    return image


def planned_frames(
    plots: list[AlgorithmPlot],
    *,
    fps: int,
    graph_seconds: float,
    algorithm_hold: float,
    final_hold: float,
) -> list[FrameState]:
    timeline: list[FrameState] = []
    for algorithm_index, plot in enumerate(plots):
        for result in plot.examples:
            timeline.extend(FrameState("example", algorithm_index, result, False) for _ in range(int(fps * 0.5)))
            timeline.extend(
                FrameState("example", algorithm_index, result, True, audio_event="lock" if frame == 0 else None)
                for frame in range(int(fps * 0.85))
            )
        timeline.extend(FrameState("graph", algorithm_index, visible_count=0) for _ in range(int(fps * 0.65)))
        reveal_frames = max(1, int(fps * graph_seconds))
        for frame in range(reveal_frames):
            visible = min(len(plot.trials), 1 + int((frame / max(1, reveal_frames - 1)) * len(plot.trials)))
            event = "swap" if frame % max(1, fps // 8) == 0 else None
            timeline.append(FrameState("graph", algorithm_index, visible_count=visible, audio_event=event))
        timeline.extend(
            FrameState("graph", algorithm_index, visible_count=len(plot.trials), audio_event="sorted" if frame == 0 else None)
            for frame in range(int(fps * algorithm_hold))
        )
    timeline.extend(
        FrameState("graph", len(plots) - 1, visible_count=len(plots[-1].trials), final_hold=True, audio_event="sorted" if frame == 0 else None)
        for frame in range(int(fps * final_hold))
    )
    return timeline


def render_video(args: argparse.Namespace) -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required but was not found on PATH")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    max_area = args.max_side * args.max_side
    max_cost = dist.nice_axis_max(dist.worst_bound(max_area, args.metric))
    plot_left = int(args.width * 0.15)
    plot_right = int(args.width * 0.92)
    plot_top = int(args.height * 0.18)
    plot_bottom = int(args.height * 0.78)
    plots = build_plots(
        metric=args.metric,
        samples=args.samples,
        examples=args.examples,
        seed=args.seed,
        min_side=args.min_side,
        max_side=args.max_side,
        density_min=args.density_min,
        density_max=args.density_max,
        plot_left=plot_left,
        plot_right=plot_right,
        plot_top=plot_top,
        plot_bottom=plot_bottom,
        max_area=max_area,
        max_cost=max_cost,
    )
    timeline = planned_frames(plots, fps=args.fps, graph_seconds=args.graph_seconds, algorithm_hold=args.algorithm_hold, final_hold=args.final_hold)
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
                progress = frame_number / max(1, len(timeline) - 1)
                if frame_state.mode == "example":
                    assert frame_state.example is not None
                    image = dist.draw_example_frame(
                        width=args.width,
                        height=args.height,
                        spec=plots[frame_state.algorithm_index].spec,
                        result=frame_state.example,
                        solved=frame_state.solved,
                        frame_number=frame_number,
                        total_frames=len(timeline),
                    )
                else:
                    image = draw_graph_frame(
                        width=args.width,
                        height=args.height,
                        plots=plots,
                        current_index=frame_state.algorithm_index,
                        visible_count=frame_state.visible_count,
                        metric=args.metric,
                        max_area=max_area,
                        max_cost=max_cost,
                        progress=progress,
                        final_hold=frame_state.final_hold,
                    )
                if not thumbnail_saved and frame_state.final_hold:
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
            image.save(args.thumbnail)

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

    print(f"Rendered {args.output}")
    print(f"Rendered {args.thumbnail}")
    print(f"Algorithms: {', '.join(plot.spec.title for plot in plots)}")
    print(f"Frames: {len(timeline)} at {args.fps} fps ({len(timeline) / args.fps:.1f}s)")
    print(f"Audio: {'on' if audio_enabled else 'off'}")
    print(f"Metric: {args.metric}")
    print(f"Samples: {len(plots[0].trials)} graph + {len(plots[0].examples)} examples per algorithm")
    print(f"Axis max: {max_cost}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--thumbnail", type=Path, default=DEFAULT_THUMBNAIL)
    parser.add_argument("--metric", choices=("expanded", "edge-checks"), default="expanded")
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
    parser.add_argument("--graph-seconds", type=float, default=3.3)
    parser.add_argument("--algorithm-hold", type=float, default=3.0)
    parser.add_argument("--final-hold", type=float, default=2.4)
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
    if args.graph_seconds <= 0 or args.algorithm_hold <= 0 or args.final_hold <= 0:
        parser.error("--graph-seconds, --algorithm-hold, and --final-hold must be positive")
    return args


if __name__ == "__main__":
    render_video(parse_args())
