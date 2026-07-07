#!/usr/bin/env python3
"""Render a Shorts-ready Bubble Sort operation-distribution graph."""

from __future__ import annotations

import argparse
import math
import random
import shutil
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "artifacts" / "shorts" / "bubble_sort_operation_cloud.mp4"
DEFAULT_THUMBNAIL = ROOT / "artifacts" / "shorts" / "bubble_sort_operation_cloud_thumbnail.png"
DEFAULT_LOG_OUTPUT = ROOT / "artifacts" / "shorts" / "bubble_sort_operation_cloud_log.mp4"
DEFAULT_LOG_THUMBNAIL = ROOT / "artifacts" / "shorts" / "bubble_sort_operation_cloud_log_thumbnail.png"
DEFAULT_OUTPUT_DIR = ROOT / "artifacts" / "shorts"

FONT_REGULAR = Path("/System/Library/Fonts/Supplemental/Arial.ttf")
FONT_BOLD = Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf")

BG_TOP = (10, 12, 18)
BG_BOTTOM = (18, 20, 30)
TEXT = (245, 242, 232)
MUTED = (152, 164, 176)
GRID = (40, 48, 62)
PANEL = (23, 28, 39)
BEST = (126, 220, 135)
WORST = (255, 104, 91)
POINT = (86, 176, 255)
CURRENT = (255, 202, 77)


@dataclass(frozen=True)
class Trial:
    n: int
    comparisons: int
    moves: int

    @property
    def operations(self) -> int:
        return self.comparisons + self.moves


@dataclass(frozen=True)
class SortAlgorithm:
    slug: str
    name: str
    counter: object
    best_operations: object
    worst_operations: object


@lru_cache(maxsize=None)
def font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    path = FONT_BOLD if bold else FONT_REGULAR
    if path.exists():
        return ImageFont.truetype(str(path), size)
    return ImageFont.load_default(size=size)


def text_width(draw: ImageDraw.ImageDraw, content: str, typeface: ImageFont.ImageFont) -> int:
    left, _, right, _ = draw.textbbox((0, 0), content, font=typeface)
    return right - left


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


@lru_cache(maxsize=8)
def gradient_background(width: int, height: int) -> Image.Image:
    image = Image.new("RGB", (width, height), BG_TOP)
    draw = ImageDraw.Draw(image)
    for y in range(height):
        t = y / max(1, height - 1)
        row = tuple(int(lerp(BG_TOP[c], BG_BOTTOM[c], t)) for c in range(3))
        draw.line((0, y, width, y), fill=row)
    return image


def rounded_rect(
    draw: ImageDraw.ImageDraw,
    box: tuple[float, float, float, float],
    radius: int,
    fill: tuple[int, int, int],
    outline: tuple[int, int, int] | None = None,
    width: int = 1,
) -> None:
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def draw_centered_text(
    draw: ImageDraw.ImageDraw,
    y: int,
    content: str,
    typeface: ImageFont.ImageFont,
    fill: tuple[int, int, int],
    canvas_width: int,
) -> None:
    draw.text(((canvas_width - text_width(draw, content, typeface)) / 2, y), content, font=typeface, fill=fill)


def fitted_font(
    draw: ImageDraw.ImageDraw,
    content: str,
    max_width: int,
    start_size: int,
    min_size: int,
    *,
    bold: bool = False,
) -> ImageFont.ImageFont:
    size = start_size
    while size > min_size and text_width(draw, content, font(size, bold=bold)) > max_width:
        size -= 2
    return font(size, bold=bold)


def best_operations(n: int) -> int:
    return max(0, n - 1)


def bubble_worst_operations(n: int) -> int:
    # Optimized bubble sort: comparisons plus two array writes for each reverse-order swap.
    return 3 * n * (n - 1) // 2


def bubble_sort_operation_count(values: list[int]) -> tuple[int, int]:
    arr = values[:]
    comparisons = 0
    moves = 0
    for end in range(len(arr) - 1, 0, -1):
        swapped = False
        for i in range(end):
            comparisons += 1
            if arr[i] > arr[i + 1]:
                arr[i], arr[i + 1] = arr[i + 1], arr[i]
                moves += 2
                swapped = True
        if not swapped:
            break
    return comparisons, moves


def insertion_sort_operation_count(values: list[int]) -> tuple[int, int]:
    arr = values[:]
    comparisons = 0
    moves = 0
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        shifted = False
        while j >= 0:
            comparisons += 1
            if arr[j] <= key:
                break
            arr[j + 1] = arr[j]
            moves += 1
            shifted = True
            j -= 1
        if shifted:
            arr[j + 1] = key
            moves += 1
    return comparisons, moves


def insertion_worst_operations(n: int) -> int:
    if n <= 1:
        return 0
    comparisons = n * (n - 1) // 2
    moves = comparisons + n - 1
    return comparisons + moves


def selection_sort_operation_count(values: list[int]) -> tuple[int, int]:
    arr = values[:]
    comparisons = 0
    moves = 0
    for i in range(len(arr) - 1):
        min_index = i
        for j in range(i + 1, len(arr)):
            comparisons += 1
            if arr[j] < arr[min_index]:
                min_index = j
        if min_index != i:
            arr[i], arr[min_index] = arr[min_index], arr[i]
            moves += 2
    return comparisons, moves


def selection_best_operations(n: int) -> int:
    return n * (n - 1) // 2


def selection_worst_operations(n: int) -> int:
    return selection_best_operations(n) + 2 * max(0, n - 1)


def merge_sort_operation_count(values: list[int]) -> tuple[int, int]:
    arr = values[:]
    comparisons = 0
    moves = 0

    def sort_range(left: int, right: int) -> None:
        nonlocal comparisons, moves
        if right - left <= 1:
            return
        mid = (left + right) // 2
        sort_range(left, mid)
        sort_range(mid, right)

        merged: list[int] = []
        i = left
        j = mid
        while i < mid and j < right:
            comparisons += 1
            if arr[i] <= arr[j]:
                merged.append(arr[i])
                i += 1
            else:
                merged.append(arr[j])
                j += 1
        merged.extend(arr[i:mid])
        merged.extend(arr[j:right])
        for offset, value in enumerate(merged):
            arr[left + offset] = value
            moves += 1

    sort_range(0, len(arr))
    return comparisons, moves


@lru_cache(maxsize=None)
def merge_best_operations(n: int) -> int:
    if n <= 1:
        return 0
    left = n // 2
    right = n - left
    return merge_best_operations(left) + merge_best_operations(right) + min(left, right) + n


@lru_cache(maxsize=None)
def merge_worst_operations(n: int) -> int:
    if n <= 1:
        return 0
    left = n // 2
    right = n - left
    return merge_worst_operations(left) + merge_worst_operations(right) + n - 1 + n


def gnome_sort_operation_count(values: list[int]) -> tuple[int, int]:
    arr = values[:]
    comparisons = 0
    moves = 0
    index = 0
    while index < len(arr):
        if index == 0:
            index += 1
            continue
        comparisons += 1
        if arr[index] >= arr[index - 1]:
            index += 1
        else:
            arr[index], arr[index - 1] = arr[index - 1], arr[index]
            moves += 2
            index -= 1
    return comparisons, moves


def cocktail_shaker_sort_operation_count(values: list[int]) -> tuple[int, int]:
    arr = values[:]
    comparisons = 0
    moves = 0
    start = 0
    end = len(arr) - 1
    swapped = True
    while swapped:
        swapped = False
        for i in range(start, end):
            comparisons += 1
            if arr[i] > arr[i + 1]:
                arr[i], arr[i + 1] = arr[i + 1], arr[i]
                moves += 2
                swapped = True
        if not swapped:
            break
        swapped = False
        end -= 1
        for i in range(end - 1, start - 1, -1):
            comparisons += 1
            if arr[i] > arr[i + 1]:
                arr[i], arr[i + 1] = arr[i + 1], arr[i]
                moves += 2
                swapped = True
        start += 1
    return comparisons, moves


def odd_even_sort_operation_count(values: list[int]) -> tuple[int, int]:
    arr = values[:]
    comparisons = 0
    moves = 0
    sorted_pass = False
    while not sorted_pass:
        sorted_pass = True
        for i in range(1, len(arr) - 1, 2):
            comparisons += 1
            if arr[i] > arr[i + 1]:
                arr[i], arr[i + 1] = arr[i + 1], arr[i]
                moves += 2
                sorted_pass = False
        for i in range(0, len(arr) - 1, 2):
            comparisons += 1
            if arr[i] > arr[i + 1]:
                arr[i], arr[i + 1] = arr[i + 1], arr[i]
                moves += 2
                sorted_pass = False
    return comparisons, moves


def reversed_input_operations(counter, n: int) -> int:
    comparisons, moves = counter(list(range(n - 1, -1, -1)))
    return comparisons + moves


@lru_cache(maxsize=None)
def gnome_worst_operations(n: int) -> int:
    return reversed_input_operations(gnome_sort_operation_count, n)


@lru_cache(maxsize=None)
def cocktail_worst_operations(n: int) -> int:
    return reversed_input_operations(cocktail_shaker_sort_operation_count, n)


@lru_cache(maxsize=None)
def odd_even_worst_operations(n: int) -> int:
    return reversed_input_operations(odd_even_sort_operation_count, n)


ALGORITHMS = {
    "bubble": SortAlgorithm("bubble", "Bubble Sort", bubble_sort_operation_count, best_operations, bubble_worst_operations),
    "insertion": SortAlgorithm("insertion", "Insertion Sort", insertion_sort_operation_count, best_operations, insertion_worst_operations),
    "selection": SortAlgorithm("selection", "Selection Sort", selection_sort_operation_count, selection_best_operations, selection_worst_operations),
    "merge": SortAlgorithm("merge", "Merge Sort", merge_sort_operation_count, merge_best_operations, merge_worst_operations),
    "gnome": SortAlgorithm("gnome", "Gnome Sort", gnome_sort_operation_count, best_operations, gnome_worst_operations),
    "cocktail": SortAlgorithm("cocktail", "Cocktail Shaker Sort", cocktail_shaker_sort_operation_count, best_operations, cocktail_worst_operations),
    "odd-even": SortAlgorithm("odd-even", "Odd-Even Sort", odd_even_sort_operation_count, best_operations, odd_even_worst_operations),
}


def make_values_with_inversions(n: int, target_inversions: int, rng: random.Random) -> list[int]:
    """Build a permutation with an exact inversion count using a Lehmer code."""
    max_inversions = n * (n - 1) // 2
    remaining = min(max(0, target_inversions), max_inversions)
    later_capacity = max_inversions
    code: list[int] = []

    for i in range(n):
        max_digit = n - 1 - i
        later_capacity -= max_digit
        low = max(0, remaining - later_capacity)
        high = min(max_digit, remaining)
        digit = rng.randint(low, high) if low < high else low
        code.append(digit)
        remaining -= digit

    available = list(range(n))
    values = []
    for digit in code:
        values.append(available.pop(digit))
    return values


def make_random_permutation(n: int, rng: random.Random) -> list[int]:
    values = list(range(n))
    rng.shuffle(values)
    return values


def make_inversion_uniform_values(n: int, rng: random.Random) -> list[int]:
    max_inversions = n * (n - 1) // 2
    roll = rng.random()
    if roll < 0.07:
        target_inversions = 0
    elif roll < 0.14:
        target_inversions = max_inversions
    else:
        target_inversions = int(rng.random() * max_inversions)
    return make_values_with_inversions(n, target_inversions, rng)


def make_sample_values(n: int, rng: random.Random, sample_mode: str) -> list[int]:
    if sample_mode == "inversion-uniform":
        return make_inversion_uniform_values(n, rng)
    return make_random_permutation(n, rng)


def generate_trials(count: int, max_n: int, seed: int, sample_mode: str, algorithm: SortAlgorithm) -> list[Trial]:
    rng = random.Random(seed)
    trials: list[Trial] = []
    for _ in range(count):
        n = rng.randint(2, max_n)
        values = make_sample_values(n, rng, sample_mode)
        comparisons, moves = algorithm.counter(values)
        trials.append(Trial(n, comparisons, moves))
    return trials


def graph_point(
    n: float,
    operations: float,
    *,
    plot_left: int,
    plot_right: int,
    plot_top: int,
    plot_bottom: int,
    max_n: int,
    max_ops: int,
    y_scale: str,
) -> tuple[float, float]:
    x = lerp(plot_left, plot_right, (n - 2) / max(1, max_n - 2))
    if y_scale == "log":
        y_fraction = math.log10(max(1, operations)) / math.log10(max_ops)
    else:
        y_fraction = operations / max_ops
    y = lerp(plot_bottom, plot_top, y_fraction)
    return x, y


def curve_points(
    fn,
    *,
    plot_left: int,
    plot_right: int,
    plot_top: int,
    plot_bottom: int,
    max_n: int,
    max_ops: int,
    y_scale: str,
) -> list[tuple[float, float]]:
    points = []
    for n in range(2, max_n + 1, 4):
        points.append(
            graph_point(
                n,
                fn(n),
                plot_left=plot_left,
                plot_right=plot_right,
                plot_top=plot_top,
                plot_bottom=plot_bottom,
                max_n=max_n,
                max_ops=max_ops,
                y_scale=y_scale,
            )
        )
    points.append(
        graph_point(
            max_n,
            fn(max_n),
            plot_left=plot_left,
            plot_right=plot_right,
            plot_top=plot_top,
            plot_bottom=plot_bottom,
            max_n=max_n,
            max_ops=max_ops,
            y_scale=y_scale,
        )
    )
    return points


def y_axis_ticks(max_ops: int, y_scale: str) -> list[int]:
    if y_scale == "log":
        ticks = [1]
        value = 10
        while value <= max_ops:
            ticks.append(value)
            value *= 10
        return ticks
    return [int(max_ops * idx / 5) for idx in range(6)]


def nice_linear_axis_max(value: int) -> int:
    target_step = value / 5
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
    elif normalized <= 4:
        step = 4 * magnitude
    elif normalized <= 5:
        step = 5 * magnitude
    else:
        step = 10 * magnitude
    return int(step * 5)


def axis_max_operations(max_ops: int, y_scale: str) -> int:
    if y_scale == "log":
        return max_ops
    return nice_linear_axis_max(max_ops)


def format_count(value: int) -> str:
    if value >= 1_000_000:
        if value % 1_000_000 == 0:
            return f"{value // 1_000_000}M"
        return f"{value / 1_000_000:.1f}M"
    if value >= 1_000:
        if value % 1_000 == 0:
            return f"{value // 1_000}K"
        return f"{value / 1_000:.1f}K"
    return str(value)


def draw_frame(
    *,
    width: int,
    height: int,
    algorithm: SortAlgorithm,
    trials: list[Trial],
    visible_count: int,
    max_n: int,
    sample_mode: str,
    y_scale: str,
    frame_number: int,
    total_frames: int,
) -> Image.Image:
    image = gradient_background(width, height).copy()
    draw = ImageDraw.Draw(image)

    title = f"{algorithm.name.upper()} COSTS"
    title_font = fitted_font(draw, title, int(width * 0.9), int(width * 0.073), int(width * 0.048), bold=True)
    subtitle_font = font(int(width * 0.034))
    label_font = font(int(width * 0.033), bold=True)
    small_font = font(int(width * 0.026))
    tiny_font = font(int(width * 0.022), bold=True)

    margin = int(width * 0.065)
    draw_centered_text(draw, int(height * 0.052), title, title_font, TEXT, width)
    subtitle = "Random permutations reveal where typical costs cluster."
    if sample_mode == "inversion-uniform":
        subtitle = "Randomized inversion levels fill the space between best and worst case."
    if y_scale == "log":
        subtitle = "Log scale shows typical random costs across orders of magnitude."
        if sample_mode == "inversion-uniform":
            subtitle = "Log scale reveals the full spread between best and worst case."
    draw_centered_text(draw, int(height * 0.108), subtitle, subtitle_font, MUTED, width)

    plot_left = int(width * 0.14)
    plot_right = int(width * 0.91)
    plot_top = int(height * 0.23)
    plot_bottom = int(height * 0.74)
    max_ops = axis_max_operations(algorithm.worst_operations(max_n), y_scale)

    rounded_rect(draw, (plot_left - 18, plot_top - 18, plot_right + 18, plot_bottom + 18), 8, (12, 15, 22), GRID, 2)

    for value in y_axis_ticks(max_ops, y_scale):
        _, y = graph_point(
            2,
            value,
            plot_left=plot_left,
            plot_right=plot_right,
            plot_top=plot_top,
            plot_bottom=plot_bottom,
            max_n=max_n,
            max_ops=max_ops,
            y_scale=y_scale,
        )
        draw.line((plot_left, y, plot_right, y), fill=GRID, width=2)
        label = format_count(value)
        draw.text((plot_left - text_width(draw, label, small_font) - 18, y - 15), label, font=small_font, fill=MUTED)

    for idx in range(6):
        t = idx / 5
        x = lerp(plot_left, plot_right, t)
        draw.line((x, plot_top, x, plot_bottom), fill=(29, 35, 47), width=1)
        n_label = str(int(2 + (max_n - 2) * t))
        draw.text((x - text_width(draw, n_label, small_font) / 2, plot_bottom + 28), n_label, font=small_font, fill=MUTED)

    draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill=MUTED, width=3)
    draw.line((plot_left, plot_top, plot_left, plot_bottom), fill=MUTED, width=3)

    worst = curve_points(
        algorithm.worst_operations,
        plot_left=plot_left,
        plot_right=plot_right,
        plot_top=plot_top,
        plot_bottom=plot_bottom,
        max_n=max_n,
        max_ops=max_ops,
        y_scale=y_scale,
    )
    best = curve_points(
        algorithm.best_operations,
        plot_left=plot_left,
        plot_right=plot_right,
        plot_top=plot_top,
        plot_bottom=plot_bottom,
        max_n=max_n,
        max_ops=max_ops,
        y_scale=y_scale,
    )
    draw.line(worst, fill=WORST, width=5, joint="curve")
    draw.line(best, fill=BEST, width=5, joint="curve")

    if y_scale == "log":
        worst_label_xy = (plot_right - 232, plot_top + 54)
    else:
        worst_label_xy = (plot_right - 178, plot_top + 12)
    worst_label_box = (
        worst_label_xy[0] - 10,
        worst_label_xy[1] - 6,
        worst_label_xy[0] + text_width(draw, "worst case", small_font) + 10,
        worst_label_xy[1] + 34,
    )
    rounded_rect(draw, worst_label_box, 8, (12, 15, 22), None)
    draw.text(worst_label_xy, "worst case", font=small_font, fill=WORST)
    draw.text((plot_left + 14, plot_bottom - 46), "best case", font=small_font, fill=BEST)
    draw.text((plot_left + 16, plot_bottom + 68), "elements in list", font=small_font, fill=MUTED)
    y_label = "operations" if y_scale == "linear" else "operations (log)"
    draw.text((plot_left - 92, plot_top - 48), y_label, font=small_font, fill=MUTED)

    visible_trials = trials[:visible_count]
    for trial in visible_trials[:-1]:
        x, y = graph_point(
            trial.n,
            trial.operations,
            plot_left=plot_left,
            plot_right=plot_right,
            plot_top=plot_top,
            plot_bottom=plot_bottom,
            max_n=max_n,
            max_ops=max_ops,
            y_scale=y_scale,
        )
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=POINT)

    current = visible_trials[-1] if visible_trials else None
    if current is not None:
        x, y = graph_point(
            current.n,
            current.operations,
            plot_left=plot_left,
            plot_right=plot_right,
            plot_top=plot_top,
            plot_bottom=plot_bottom,
            max_n=max_n,
            max_ops=max_ops,
            y_scale=y_scale,
        )
        draw.ellipse((x - 9, y - 9, x + 9, y + 9), fill=CURRENT)
        draw.ellipse((x - 15, y - 15, x + 15, y + 15), outline=CURRENT, width=3)

    panel_y = int(height * 0.835)
    panel_h = int(height * 0.072)
    gap = int(width * 0.022)
    panel_w = int((width - margin * 2 - gap * 2) / 3)
    stats = [
        ("samples", str(visible_count), POINT),
        ("current n", str(current.n if current else 0), CURRENT),
        ("operations", format_count(current.operations if current else 0), TEXT),
    ]
    for i, (name, value, color) in enumerate(stats):
        x0 = margin + i * (panel_w + gap)
        rounded_rect(draw, (x0, panel_y, x0 + panel_w, panel_y + panel_h), 16, PANEL, GRID, 2)
        draw.text((x0 + 18, panel_y + 12), value, font=label_font, fill=color)
        draw.text((x0 + 18, panel_y + int(panel_h * 0.58)), name, font=small_font, fill=MUTED)

    progress = frame_number / max(1, total_frames - 1)
    bar_x0 = margin
    bar_x1 = width - margin
    bar_y = int(height * 0.945)
    bar_h = int(height * 0.01)
    rounded_rect(draw, (bar_x0, bar_y, bar_x1, bar_y + bar_h), 12, (15, 18, 25), GRID, 2)
    rounded_rect(draw, (bar_x0 + 4, bar_y + 4, lerp(bar_x0 + 4, bar_x1 - 4, progress), bar_y + bar_h - 4), 8, POINT)

    return image


def default_output_paths(algorithm: SortAlgorithm, y_scale: str) -> tuple[Path, Path]:
    if algorithm.slug == "bubble" and y_scale == "linear":
        return DEFAULT_OUTPUT, DEFAULT_THUMBNAIL
    if algorithm.slug == "bubble" and y_scale == "log":
        return DEFAULT_LOG_OUTPUT, DEFAULT_LOG_THUMBNAIL

    suffix = "_log" if y_scale == "log" else ""
    base = f"{algorithm.slug}_sort_operation_cloud{suffix}"
    return DEFAULT_OUTPUT_DIR / f"{base}.mp4", DEFAULT_OUTPUT_DIR / f"{base}_thumbnail.png"


def render_video(args: argparse.Namespace) -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required but was not found on PATH")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    algorithm = ALGORITHMS[args.algorithm]
    trials = generate_trials(args.samples, args.max_n, args.seed, args.sample_mode, algorithm)
    hold_frames = int(args.fps * 1.5)
    reveal_frames = max(args.samples, int(args.fps * args.duration))
    total_frames = reveal_frames + hold_frames

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
        str(args.output),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    assert process.stdin is not None

    thumbnail_saved = False
    try:
        for frame_number in range(total_frames):
            if frame_number < reveal_frames:
                visible_count = max(1, min(args.samples, int((frame_number + 1) / reveal_frames * args.samples)))
            else:
                visible_count = args.samples

            image = draw_frame(
                width=args.width,
                height=args.height,
                algorithm=algorithm,
                trials=trials,
                visible_count=visible_count,
                max_n=args.max_n,
                sample_mode=args.sample_mode,
                y_scale=args.y_scale,
                frame_number=frame_number,
                total_frames=total_frames,
            )
            if not thumbnail_saved and visible_count >= min(args.samples, int(args.samples * 0.65)):
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
    print(f"Algorithm: {algorithm.name}")
    print(f"Trials: {len(trials)}")
    print(f"Frames: {total_frames} at {args.fps} fps ({total_frames / args.fps:.1f}s)")
    print(f"Max n: {args.max_n}")
    print(f"Sample mode: {args.sample_mode}")
    print(f"Y scale: {args.y_scale}")
    print(f"Worst-case operations at max n: {algorithm.worst_operations(args.max_n)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", choices=sorted(ALGORITHMS), default="bubble")
    parser.add_argument("--all", action="store_true", help="Render every configured sorting algorithm")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--thumbnail", type=Path)
    parser.add_argument("--width", type=int, default=1080)
    parser.add_argument("--height", type=int, default=1920)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--duration", type=float, default=24.0)
    parser.add_argument("--samples", type=int, default=520)
    parser.add_argument("--max-n", type=int, default=100)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--sample-mode", choices=("random-permutation", "inversion-uniform"), default="random-permutation")
    parser.add_argument("--y-scale", choices=("linear", "log"), default="linear")
    args = parser.parse_args()

    if args.all and (args.output is not None or args.thumbnail is not None):
        parser.error("--output and --thumbnail cannot be used with --all")

    if not args.all:
        algorithm = ALGORITHMS[args.algorithm]
        default_output, default_thumbnail = default_output_paths(algorithm, args.y_scale)
        if args.output is None:
            args.output = default_output
        if args.thumbnail is None:
            args.thumbnail = default_thumbnail

    if args.max_n < 10:
        parser.error("--max-n must be at least 10")
    if args.samples < 10:
        parser.error("--samples must be at least 10")
    if args.width <= 0 or args.height <= 0 or args.fps <= 0 or args.duration <= 0:
        parser.error("--width, --height, --fps, and --duration must be positive")
    return args


if __name__ == "__main__":
    parsed_args = parse_args()
    if parsed_args.all:
        for algorithm_slug in ALGORITHMS:
            output, thumbnail = default_output_paths(ALGORITHMS[algorithm_slug], parsed_args.y_scale)
            render_args = argparse.Namespace(**vars(parsed_args))
            render_args.all = False
            render_args.algorithm = algorithm_slug
            render_args.output = output
            render_args.thumbnail = thumbnail
            render_video(render_args)
    else:
        render_video(parsed_args)
