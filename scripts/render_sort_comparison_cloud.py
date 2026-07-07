#!/usr/bin/env python3
"""Render a shared-axis sorting algorithm operation comparison short."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw

import render_bubble_sort_operation_cloud as cloud


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "artifacts" / "shorts" / "sorting_algorithm_comparison_cloud.mp4"
DEFAULT_THUMBNAIL = ROOT / "artifacts" / "shorts" / "sorting_algorithm_comparison_cloud_thumbnail.png"

ALGORITHM_ORDER = ["merge", "selection", "insertion", "bubble", "cocktail", "odd-even", "gnome"]

COMPLETED_LINE = (94, 104, 118)
COMPLETED_POINT = (75, 84, 96)
CURRENT_POINT = (86, 176, 255)
CURRENT_BEST = (126, 220, 135)
CURRENT_WORST = (255, 104, 91)
CURRENT_ACCENT = (255, 202, 77)


@dataclass(frozen=True)
class AlgorithmPlot:
    slug: str
    name: str
    trials: list[cloud.Trial]
    best_curve: list[tuple[float, float]]
    worst_curve: list[tuple[float, float]]
    points: list[tuple[float, float]]


def curve_points_for_algorithm(
    algorithm: cloud.SortAlgorithm,
    *,
    plot_left: int,
    plot_right: int,
    plot_top: int,
    plot_bottom: int,
    max_n: int,
    axis_max: int,
    which: str,
) -> list[tuple[float, float]]:
    fn = algorithm.best_operations if which == "best" else algorithm.worst_operations
    points = []
    for n in range(2, max_n + 1):
        points.append(
            cloud.graph_point(
                n,
                fn(n),
                plot_left=plot_left,
                plot_right=plot_right,
                plot_top=plot_top,
                plot_bottom=plot_bottom,
                max_n=max_n,
                max_ops=axis_max,
                y_scale="linear",
            )
        )
    return points


def build_plots(
    *,
    plot_left: int,
    plot_right: int,
    plot_top: int,
    plot_bottom: int,
    max_n: int,
    samples: int,
    seed: int,
    axis_max: int,
) -> list[AlgorithmPlot]:
    plots = []
    for slug in ALGORITHM_ORDER:
        algorithm = cloud.ALGORITHMS[slug]
        trials = cloud.generate_trials(samples, max_n, seed, "random-permutation", algorithm)
        points = [
            cloud.graph_point(
                trial.n,
                trial.operations,
                plot_left=plot_left,
                plot_right=plot_right,
                plot_top=plot_top,
                plot_bottom=plot_bottom,
                max_n=max_n,
                max_ops=axis_max,
                y_scale="linear",
            )
            for trial in trials
        ]
        plots.append(
            AlgorithmPlot(
                slug=slug,
                name=algorithm.name,
                trials=trials,
                best_curve=curve_points_for_algorithm(
                    algorithm,
                    plot_left=plot_left,
                    plot_right=plot_right,
                    plot_top=plot_top,
                    plot_bottom=plot_bottom,
                    max_n=max_n,
                    axis_max=axis_max,
                    which="best",
                ),
                worst_curve=curve_points_for_algorithm(
                    algorithm,
                    plot_left=plot_left,
                    plot_right=plot_right,
                    plot_top=plot_top,
                    plot_bottom=plot_bottom,
                    max_n=max_n,
                    axis_max=axis_max,
                    which="worst",
                ),
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
    max_n: int,
    axis_max: int,
) -> None:
    small_font = cloud.font(int(width * 0.026))
    cloud.rounded_rect(draw, (plot_left - 18, plot_top - 18, plot_right + 18, plot_bottom + 18), 8, (12, 15, 22), cloud.GRID, 2)

    for value in cloud.y_axis_ticks(axis_max, "linear"):
        _, y = cloud.graph_point(
            2,
            value,
            plot_left=plot_left,
            plot_right=plot_right,
            plot_top=plot_top,
            plot_bottom=plot_bottom,
            max_n=max_n,
            max_ops=axis_max,
            y_scale="linear",
        )
        draw.line((plot_left, y, plot_right, y), fill=cloud.GRID, width=2)
        label = cloud.format_count(value)
        draw.text((plot_left - cloud.text_width(draw, label, small_font) - 18, y - 15), label, font=small_font, fill=cloud.MUTED)

    for idx in range(6):
        t = idx / 5
        x = cloud.lerp(plot_left, plot_right, t)
        draw.line((x, plot_top, x, plot_bottom), fill=(29, 35, 47), width=1)
        n_label = str(int(2 + (max_n - 2) * t))
        draw.text((x - cloud.text_width(draw, n_label, small_font) / 2, plot_bottom + 28), n_label, font=small_font, fill=cloud.MUTED)

    draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill=cloud.MUTED, width=3)
    draw.line((plot_left, plot_top, plot_left, plot_bottom), fill=cloud.MUTED, width=3)
    draw.text((plot_left + 16, plot_bottom + 68), "elements in list", font=small_font, fill=cloud.MUTED)
    draw.text((plot_left - 92, plot_top - 48), "operations", font=small_font, fill=cloud.MUTED)


def draw_completed_plot(draw: ImageDraw.ImageDraw, plot: AlgorithmPlot) -> None:
    draw.line(plot.worst_curve, fill=COMPLETED_LINE, width=3)
    draw.line(plot.best_curve, fill=(74, 83, 94), width=2)
    for x, y in plot.points:
        draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=COMPLETED_POINT)


def draw_current_plot(draw: ImageDraw.ImageDraw, plot: AlgorithmPlot, visible_count: int) -> None:
    draw.line(plot.worst_curve, fill=CURRENT_WORST, width=5)
    draw.line(plot.best_curve, fill=CURRENT_BEST, width=5)
    for x, y in plot.points[: max(0, visible_count - 1)]:
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=CURRENT_POINT)
    if visible_count:
        x, y = plot.points[visible_count - 1]
        draw.ellipse((x - 9, y - 9, x + 9, y + 9), fill=CURRENT_ACCENT)
        draw.ellipse((x - 15, y - 15, x + 15, y + 15), outline=CURRENT_ACCENT, width=3)


def draw_cards(
    draw: ImageDraw.ImageDraw,
    *,
    width: int,
    height: int,
    plot: AlgorithmPlot | None,
    visible_count: int,
    completed_count: int,
    max_n: int,
) -> None:
    margin = int(width * 0.065)
    label_font = cloud.font(int(width * 0.033), bold=True)
    small_font = cloud.font(int(width * 0.026))
    panel_y = int(height * 0.835)
    panel_h = int(height * 0.072)
    gap = int(width * 0.022)
    panel_w = int((width - margin * 2 - gap * 2) / 3)

    current_name = plot.name if plot else "All Algorithms"
    if len(current_name) > 18:
        current_name = current_name.replace(" Shaker", "")
    count_label = "samples" if plot else "algorithms"
    count_value = str(visible_count if plot else completed_count)
    stats = [
        ("current", current_name, CURRENT_ACCENT),
        (count_label, count_value, CURRENT_POINT),
        ("max n", str(max_n), cloud.TEXT),
    ]
    for i, (name, value, color) in enumerate(stats):
        x0 = margin + i * (panel_w + gap)
        cloud.rounded_rect(draw, (x0, panel_y, x0 + panel_w, panel_y + panel_h), 16, cloud.PANEL, cloud.GRID, 2)
        draw.text((x0 + 18, panel_y + 12), value, font=label_font, fill=color)
        draw.text((x0 + 18, panel_y + int(panel_h * 0.58)), name, font=small_font, fill=cloud.MUTED)


def draw_frame(
    *,
    width: int,
    height: int,
    plots: list[AlgorithmPlot],
    current_index: int,
    visible_count: int,
    progress: float,
    max_n: int,
    axis_max: int,
    final_hold: bool,
) -> Image.Image:
    image = cloud.gradient_background(width, height).copy()
    draw = ImageDraw.Draw(image)

    title_font = cloud.font(int(width * 0.06), bold=True)
    subtitle_font = cloud.font(int(width * 0.034))
    small_font = cloud.font(int(width * 0.026))
    margin = int(width * 0.065)
    plot_left = int(width * 0.14)
    plot_right = int(width * 0.91)
    plot_top = int(height * 0.23)
    plot_bottom = int(height * 0.74)

    cloud.draw_centered_text(draw, int(height * 0.052), "SORTING COSTS COMPARED", title_font, cloud.TEXT, width)
    subtitle = "Each algorithm animates, then fades into the shared cost map."
    if final_hold:
        subtitle = "All algorithms on one shared operation scale."
    cloud.draw_centered_text(draw, int(height * 0.108), subtitle, subtitle_font, cloud.MUTED, width)

    draw_axes(
        draw,
        width=width,
        plot_left=plot_left,
        plot_right=plot_right,
        plot_top=plot_top,
        plot_bottom=plot_bottom,
        max_n=max_n,
        axis_max=axis_max,
    )

    completed_range = range(len(plots)) if final_hold else range(current_index)
    for idx in completed_range:
        draw_completed_plot(draw, plots[idx])

    current_plot = None if final_hold else plots[current_index]
    if current_plot:
        draw_current_plot(draw, current_plot, visible_count)
        label = current_plot.name
        draw.text((plot_right - cloud.text_width(draw, label, small_font) - 10, plot_top + 12), label, font=small_font, fill=CURRENT_ACCENT)

    draw_cards(
        draw,
        width=width,
        height=height,
        plot=current_plot,
        visible_count=visible_count,
        completed_count=len(plots),
        max_n=max_n,
    )

    bar_x0 = margin
    bar_x1 = width - margin
    bar_y = int(height * 0.945)
    bar_h = max(10, int(height * 0.01))
    inset = max(2, min(4, bar_h // 3))
    cloud.rounded_rect(draw, (bar_x0, bar_y, bar_x1, bar_y + bar_h), 12, (15, 18, 25), cloud.GRID, 2)
    cloud.rounded_rect(
        draw,
        (bar_x0 + inset, bar_y + inset, cloud.lerp(bar_x0 + inset, bar_x1 - inset, progress), bar_y + bar_h - inset),
        8,
        CURRENT_POINT,
    )
    return image


def render_video(args: argparse.Namespace) -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required but was not found on PATH")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    width = args.width
    height = args.height
    plot_left = int(width * 0.14)
    plot_right = int(width * 0.91)
    plot_top = int(height * 0.23)
    plot_bottom = int(height * 0.74)
    max_worst = max(cloud.ALGORITHMS[slug].worst_operations(args.max_n) for slug in ALGORITHM_ORDER)
    axis_max = cloud.axis_max_operations(max_worst, "linear")
    plots = build_plots(
        plot_left=plot_left,
        plot_right=plot_right,
        plot_top=plot_top,
        plot_bottom=plot_bottom,
        max_n=args.max_n,
        samples=args.samples,
        seed=args.seed,
        axis_max=axis_max,
    )

    reveal_frames = int(args.fps * args.seconds_per_algorithm)
    final_frames = int(args.fps * args.final_hold)
    total_frames = reveal_frames * len(plots) + final_frames

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
        f"{width}x{height}",
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
        str(args.output),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    assert process.stdin is not None

    thumbnail_saved = False
    try:
        for frame_number in range(total_frames):
            final_hold = frame_number >= reveal_frames * len(plots)
            if final_hold:
                current_index = len(plots) - 1
                visible_count = args.samples
            else:
                current_index = frame_number // reveal_frames
                local_frame = frame_number % reveal_frames
                visible_count = max(1, min(args.samples, int((local_frame + 1) / reveal_frames * args.samples)))
            progress = frame_number / max(1, total_frames - 1)
            image = draw_frame(
                width=width,
                height=height,
                plots=plots,
                current_index=current_index,
                visible_count=visible_count,
                progress=progress,
                max_n=args.max_n,
                axis_max=axis_max,
                final_hold=final_hold,
            )
            if not thumbnail_saved and final_hold:
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

    print(f"Rendered {args.output}")
    print(f"Rendered {args.thumbnail}")
    print(f"Algorithms: {', '.join(plot.name for plot in plots)}")
    print(f"Frames: {total_frames} at {args.fps} fps ({total_frames / args.fps:.1f}s)")
    print(f"Max n: {args.max_n}")
    print(f"Axis max: {axis_max}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--thumbnail", type=Path, default=DEFAULT_THUMBNAIL)
    parser.add_argument("--width", type=int, default=1080)
    parser.add_argument("--height", type=int, default=1920)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--max-n", type=int, default=100)
    parser.add_argument("--samples", type=int, default=520)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--seconds-per-algorithm", type=float, default=4.2)
    parser.add_argument("--final-hold", type=float, default=3.0)
    args = parser.parse_args()

    if args.max_n < 10:
        parser.error("--max-n must be at least 10")
    if args.samples < 10:
        parser.error("--samples must be at least 10")
    if args.width <= 0 or args.height <= 0 or args.fps <= 0:
        parser.error("--width, --height, and --fps must be positive")
    if args.seconds_per_algorithm <= 0 or args.final_hold <= 0:
        parser.error("--seconds-per-algorithm and --final-hold must be positive")
    return args


if __name__ == "__main__":
    render_video(parse_args())
