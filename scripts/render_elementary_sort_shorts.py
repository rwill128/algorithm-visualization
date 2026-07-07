#!/usr/bin/env python3
"""Render Shorts-ready visualizations for elementary sorting algorithms."""

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
DEFAULT_OUTPUT_DIR = ROOT / "artifacts" / "shorts"


@dataclass(frozen=True)
class AlgorithmSpec:
    slug: str
    title: str
    subtitle: str


ALGORITHMS = {
    "insertion": AlgorithmSpec("insertion", "INSERTION SORT", "Grow a sorted prefix one value at a time."),
    "selection": AlgorithmSpec("selection", "SELECTION SORT", "Find the minimum. Lock it into place."),
    "gnome": AlgorithmSpec("gnome", "GNOME SORT", "Step forward if ordered. Swap back if not."),
    "cocktail": AlgorithmSpec("cocktail", "COCKTAIL SORT", "Bubble both directions through the list."),
    "odd-even": AlgorithmSpec("odd-even", "ODD-EVEN SORT", "Alternate odd and even neighbor pairs."),
}


@dataclass(frozen=True)
class Step:
    values: tuple[int, ...]
    pair: tuple[int, int] | None
    sorted_indices: tuple[int, ...]
    status_text: str
    label: str
    comparisons: int
    moves: int
    swaps: int
    inversions: int
    max_inversions: int
    swapped: bool = False
    before: tuple[int, ...] | None = None


@dataclass(frozen=True)
class FrameState:
    step: Step
    swap_progress: float | None = None
    audio_event: str | None = None


def make_step(
    arr: list[int],
    *,
    pair: tuple[int, int] | None,
    sorted_indices: set[int],
    status_text: str,
    label: str,
    comparisons: int,
    moves: int,
    swaps: int,
    max_inversions: int,
    swapped: bool = False,
    before: tuple[int, ...] | None = None,
) -> Step:
    return Step(
        tuple(arr),
        pair,
        tuple(sorted(sorted_indices)),
        status_text,
        label,
        comparisons,
        moves,
        swaps,
        bubble.inversion_count(arr),
        max_inversions,
        swapped=swapped,
        before=before,
    )


def insertion_sort_steps(values: list[int]) -> list[Step]:
    arr = values[:]
    comparisons = moves = swaps = 0
    max_inversions = len(arr) * (len(arr) - 1) // 2
    sorted_prefix = 1
    n = len(arr)
    steps = [
        make_step(
            arr,
            pair=None,
            sorted_indices=set(),
            status_text=f"sorted prefix: 1/{n}",
            label="start",
            comparisons=comparisons,
            moves=moves,
            swaps=swaps,
            max_inversions=max_inversions,
        )
    ]

    for i in range(1, n):
        j = i
        while j > 0:
            before = tuple(arr)
            comparisons += 1
            should_swap = arr[j - 1] > arr[j]
            if should_swap:
                arr[j - 1], arr[j] = arr[j], arr[j - 1]
                moves += 2
                swaps += 1
            steps.append(
                make_step(
                    arr,
                    pair=(j - 1, j),
                    sorted_indices=set(range(sorted_prefix)),
                    status_text=f"sorted prefix: {sorted_prefix}/{n}",
                    label="swap" if should_swap else "compare",
                    comparisons=comparisons,
                    moves=moves,
                    swaps=swaps,
                    max_inversions=max_inversions,
                    swapped=should_swap,
                    before=before,
                )
            )
            if not should_swap:
                break
            j -= 1
        sorted_prefix = i + 1
        steps.append(
            make_step(
                arr,
                pair=None,
                sorted_indices=set(range(sorted_prefix)),
                status_text=f"sorted prefix: {sorted_prefix}/{n}",
                label="lock",
                comparisons=comparisons,
                moves=moves,
                swaps=swaps,
                max_inversions=max_inversions,
            )
        )

    steps.append(
        make_step(
            arr,
            pair=None,
            sorted_indices=set(range(n)),
            status_text=f"sorted prefix: {n}/{n}",
            label="sorted",
            comparisons=comparisons,
            moves=moves,
            swaps=swaps,
            max_inversions=max_inversions,
        )
    )
    assert arr == sorted(values)
    return steps


def selection_sort_steps(values: list[int]) -> list[Step]:
    arr = values[:]
    comparisons = moves = swaps = 0
    max_inversions = len(arr) * (len(arr) - 1) // 2
    n = len(arr)
    locked: set[int] = set()
    steps = [
        make_step(
            arr,
            pair=None,
            sorted_indices=locked,
            status_text=f"locked bars: 0/{n}",
            label="start",
            comparisons=comparisons,
            moves=moves,
            swaps=swaps,
            max_inversions=max_inversions,
        )
    ]

    for i in range(n - 1):
        min_index = i
        for j in range(i + 1, n):
            before = tuple(arr)
            previous_min_index = min_index
            comparisons += 1
            if arr[j] < arr[min_index]:
                min_index = j
                label = "new_min"
            else:
                label = "compare"
            steps.append(
                make_step(
                    arr,
                    pair=(previous_min_index, j),
                    sorted_indices=locked,
                    status_text=f"locked bars: {len(locked)}/{n}",
                    label=label,
                    comparisons=comparisons,
                    moves=moves,
                    swaps=swaps,
                    max_inversions=max_inversions,
                    before=before,
                )
            )
        if min_index != i:
            before = tuple(arr)
            arr[i], arr[min_index] = arr[min_index], arr[i]
            moves += 2
            swaps += 1
            locked.add(i)
            steps.append(
                make_step(
                    arr,
                    pair=(i, min_index),
                    sorted_indices=locked,
                    status_text=f"locked bars: {len(locked)}/{n}",
                    label="swap",
                    comparisons=comparisons,
                    moves=moves,
                    swaps=swaps,
                    max_inversions=max_inversions,
                    swapped=True,
                    before=before,
                )
            )
        else:
            locked.add(i)
        steps.append(
            make_step(
                arr,
                pair=None,
                sorted_indices=locked,
                status_text=f"locked bars: {len(locked)}/{n}",
                label="lock",
                comparisons=comparisons,
                moves=moves,
                swaps=swaps,
                max_inversions=max_inversions,
            )
        )

    locked.add(n - 1)
    steps.append(
        make_step(
            arr,
            pair=None,
            sorted_indices=locked,
            status_text=f"locked bars: {n}/{n}",
            label="sorted",
            comparisons=comparisons,
            moves=moves,
            swaps=swaps,
            max_inversions=max_inversions,
        )
    )
    assert arr == sorted(values)
    return steps


def gnome_sort_steps(values: list[int]) -> list[Step]:
    arr = values[:]
    comparisons = moves = swaps = 0
    max_inversions = len(arr) * (len(arr) - 1) // 2
    n = len(arr)
    steps = [
        make_step(
            arr,
            pair=None,
            sorted_indices=set(),
            status_text=f"scan index: 0/{n}",
            label="start",
            comparisons=comparisons,
            moves=moves,
            swaps=swaps,
            max_inversions=max_inversions,
        )
    ]

    index = 0
    while index < n:
        if index == 0:
            index += 1
            continue
        before = tuple(arr)
        comparisons += 1
        should_swap = arr[index] < arr[index - 1]
        if should_swap:
            arr[index], arr[index - 1] = arr[index - 1], arr[index]
            moves += 2
            swaps += 1
        steps.append(
            make_step(
                arr,
                pair=(index - 1, index),
                sorted_indices=set(),
                status_text=f"scan index: {index}/{n}",
                label="swap" if should_swap else "compare",
                comparisons=comparisons,
                moves=moves,
                swaps=swaps,
                max_inversions=max_inversions,
                swapped=should_swap,
                before=before,
            )
        )
        if should_swap:
            index -= 1
        else:
            index += 1

    steps.append(
        make_step(
            arr,
            pair=None,
            sorted_indices=set(range(n)),
            status_text=f"scan index: {n}/{n}",
            label="sorted",
            comparisons=comparisons,
            moves=moves,
            swaps=swaps,
            max_inversions=max_inversions,
        )
    )
    assert arr == sorted(values)
    return steps


def cocktail_sort_steps(values: list[int]) -> list[Step]:
    arr = values[:]
    comparisons = moves = swaps = 0
    max_inversions = len(arr) * (len(arr) - 1) // 2
    n = len(arr)
    start = 0
    end = n - 1
    locked: set[int] = set()
    steps = [
        make_step(
            arr,
            pair=None,
            sorted_indices=locked,
            status_text=f"locked ends: 0/{n}",
            label="start",
            comparisons=comparisons,
            moves=moves,
            swaps=swaps,
            max_inversions=max_inversions,
        )
    ]

    swapped_this_pass = True
    while swapped_this_pass:
        swapped_this_pass = False
        for i in range(start, end):
            before = tuple(arr)
            comparisons += 1
            should_swap = arr[i] > arr[i + 1]
            if should_swap:
                arr[i], arr[i + 1] = arr[i + 1], arr[i]
                moves += 2
                swaps += 1
                swapped_this_pass = True
            steps.append(
                make_step(
                    arr,
                    pair=(i, i + 1),
                    sorted_indices=locked,
                    status_text=f"locked ends: {len(locked)}/{n}",
                    label="swap" if should_swap else "compare",
                    comparisons=comparisons,
                    moves=moves,
                    swaps=swaps,
                    max_inversions=max_inversions,
                    swapped=should_swap,
                    before=before,
                )
            )
        locked.add(end)
        steps.append(
            make_step(
                arr,
                pair=None,
                sorted_indices=locked,
                status_text=f"locked ends: {len(locked)}/{n}",
                label="lock",
                comparisons=comparisons,
                moves=moves,
                swaps=swaps,
                max_inversions=max_inversions,
            )
        )
        if not swapped_this_pass:
            break

        swapped_this_pass = False
        end -= 1
        for i in range(end - 1, start - 1, -1):
            before = tuple(arr)
            comparisons += 1
            should_swap = arr[i] > arr[i + 1]
            if should_swap:
                arr[i], arr[i + 1] = arr[i + 1], arr[i]
                moves += 2
                swaps += 1
                swapped_this_pass = True
            steps.append(
                make_step(
                    arr,
                    pair=(i, i + 1),
                    sorted_indices=locked,
                    status_text=f"locked ends: {len(locked)}/{n}",
                    label="swap" if should_swap else "compare",
                    comparisons=comparisons,
                    moves=moves,
                    swaps=swaps,
                    max_inversions=max_inversions,
                    swapped=should_swap,
                    before=before,
                )
            )
        locked.add(start)
        steps.append(
            make_step(
                arr,
                pair=None,
                sorted_indices=locked,
                status_text=f"locked ends: {len(locked)}/{n}",
                label="lock",
                comparisons=comparisons,
                moves=moves,
                swaps=swaps,
                max_inversions=max_inversions,
            )
        )
        start += 1

    steps.append(
        make_step(
            arr,
            pair=None,
            sorted_indices=set(range(n)),
            status_text=f"locked ends: {n}/{n}",
            label="sorted",
            comparisons=comparisons,
            moves=moves,
            swaps=swaps,
            max_inversions=max_inversions,
        )
    )
    assert arr == sorted(values)
    return steps


def odd_even_sort_steps(values: list[int]) -> list[Step]:
    arr = values[:]
    comparisons = moves = swaps = 0
    max_inversions = len(arr) * (len(arr) - 1) // 2
    n = len(arr)
    pass_number = 0
    steps = [
        make_step(
            arr,
            pair=None,
            sorted_indices=set(),
            status_text="phase: odd",
            label="start",
            comparisons=comparisons,
            moves=moves,
            swaps=swaps,
            max_inversions=max_inversions,
        )
    ]

    sorted_pass = False
    while not sorted_pass:
        sorted_pass = True
        pass_number += 1
        for phase_name, first in (("odd", 1), ("even", 0)):
            for i in range(first, n - 1, 2):
                before = tuple(arr)
                comparisons += 1
                should_swap = arr[i] > arr[i + 1]
                if should_swap:
                    arr[i], arr[i + 1] = arr[i + 1], arr[i]
                    moves += 2
                    swaps += 1
                    sorted_pass = False
                steps.append(
                    make_step(
                        arr,
                        pair=(i, i + 1),
                        sorted_indices=set(),
                        status_text=f"phase: {phase_name} {pass_number}",
                        label="swap" if should_swap else "compare",
                        comparisons=comparisons,
                        moves=moves,
                        swaps=swaps,
                        max_inversions=max_inversions,
                        swapped=should_swap,
                        before=before,
                    )
                )

    steps.append(
        make_step(
            arr,
            pair=None,
            sorted_indices=set(range(n)),
            status_text="phase: complete",
            label="sorted",
            comparisons=comparisons,
            moves=moves,
            swaps=swaps,
            max_inversions=max_inversions,
        )
    )
    assert arr == sorted(values)
    return steps


STEP_BUILDERS = {
    "insertion": insertion_sort_steps,
    "selection": selection_sort_steps,
    "gnome": gnome_sort_steps,
    "cocktail": cocktail_sort_steps,
    "odd-even": odd_even_sort_steps,
}


def bar_positions(
    values: tuple[int, ...],
    width: int,
    pair: tuple[int, int] | None,
    before: tuple[int, ...] | None,
    swap_progress: float | None,
) -> list[tuple[float, int, int]]:
    chart_left = int(width * 0.09)
    chart_right = int(width * 0.91)
    step = (chart_right - chart_left) / len(values)
    positions = [(chart_left + step * i + step * 0.12, value, i) for i, value in enumerate(values)]
    if pair is None or before is None or swap_progress is None:
        return positions

    i, j = pair
    eased = bubble.ease(swap_progress)
    base_i = chart_left + step * i + step * 0.12
    base_j = chart_left + step * j + step * 0.12
    animated = [(chart_left + step * idx + step * 0.12, value, idx) for idx, value in enumerate(before)]
    animated[i] = (bubble.lerp(base_i, base_j, eased), before[i], i)
    animated[j] = (bubble.lerp(base_j, base_i, eased), before[j], j)
    return animated


def draw_frame(
    *,
    width: int,
    height: int,
    spec: AlgorithmSpec,
    step: Step,
    frame_number: int,
    total_frames: int,
    swap_progress: float | None = None,
) -> Image.Image:
    image = bubble.gradient_background(width, height).copy()
    draw = ImageDraw.Draw(image)

    title_font = bubble.fit_font(draw, spec.title, int(width * 0.085), int(width * 0.88), bold=True, min_size=int(width * 0.056))
    subtitle_font = bubble.font(int(width * 0.038))
    label_font = bubble.font(int(width * 0.042), bold=True)
    small_font = bubble.font(int(width * 0.029))
    tiny_font = bubble.font(int(width * 0.024), bold=True)

    margin = int(width * 0.065)
    bubble.draw_centered_text(draw, int(height * 0.055), spec.title, title_font, bubble.TEXT, width)
    bubble.draw_centered_text(draw, int(height * 0.115), spec.subtitle, subtitle_font, bubble.MUTED, width)

    badge_y = int(height * 0.17)
    badge_h = int(height * 0.052)
    bubble.rounded_rect(draw, (margin, badge_y, width - margin, badge_y + badge_h), 18, bubble.PANEL, bubble.GRID, 2)

    labels = {
        "start": "Unsorted input",
        "compare": "Already ordered",
        "swap": "Swap values",
        "new_min": "New minimum found",
        "lock": "Position locked",
        "sorted": "Sorted",
    }
    label = labels[step.label]
    label_color = (
        bubble.SWAP
        if step.label == "swap"
        else bubble.SORTED
        if step.label in {"compare", "lock", "sorted"}
        else bubble.COMPARE
        if step.label == "new_min"
        else bubble.TEXT
    )
    status_x = width - margin - bubble.text_width(draw, step.status_text, small_font) - 24
    label_max_width = int(status_x - (margin + 24) - 28)
    fitted_label_font = bubble.fit_font(draw, label, int(width * 0.042), label_max_width, bold=True, min_size=int(width * 0.026))
    draw.text((margin + 24, badge_y + int(badge_h * 0.22)), label, font=fitted_label_font, fill=label_color)
    draw.text((status_x, badge_y + int(badge_h * 0.31)), step.status_text, font=small_font, fill=bubble.MUTED)

    chart_top = int(height * 0.27)
    chart_bottom = int(height * 0.755)
    chart_left = int(width * 0.09)
    chart_right = int(width * 0.91)
    chart_height = chart_bottom - chart_top
    max_value = max(step.values)
    bar_gap = max(4, int(width * 0.006))
    bar_step = (chart_right - chart_left) / len(step.values)
    bar_width = max(10, int(bar_step - bar_gap))

    for grid_idx in range(6):
        y = chart_top + chart_height * grid_idx / 5
        draw.line((chart_left, y, chart_right, y), fill=bubble.GRID, width=2)

    tau_fill = 1.0 - (step.inversions / step.max_inversions) if step.max_inversions else 1.0
    tau_fill = max(0.0, min(1.0, tau_fill))
    active = set(step.pair or ())
    sorted_set = set(step.sorted_indices)

    positions = bar_positions(step.values, width, step.pair, step.before, swap_progress)
    for visual_x, value, idx in sorted(positions, key=lambda item: item[0]):
        bar_h = int((value / max_value) * chart_height)
        y0 = chart_bottom - bar_h
        y1 = chart_bottom
        if step.label == "sorted" or idx in sorted_set:
            color = bubble.SORTED
        elif idx in active and step.label == "swap":
            color = bubble.SWAP
        elif idx in active:
            color = bubble.COMPARE
        else:
            color = bubble.BAR if idx % 2 == 0 else bubble.BAR_ALT
        bubble.rounded_rect(draw, (visual_x, y0, visual_x + bar_width, y1), 8, color)

        if idx in active or idx in sorted_set or step.label == "sorted":
            value_text = str(value)
            draw.text(
                (visual_x + (bar_width - bubble.text_width(draw, value_text, tiny_font)) / 2, y0 - int(height * 0.026)),
                value_text,
                font=tiny_font,
                fill=bubble.TEXT,
            )

    axis_y = chart_bottom + int(height * 0.018)
    draw.line((chart_left, axis_y, chart_right, axis_y), fill=bubble.GRID, width=3)

    compare_y = int(height * 0.787)
    compare_h = int(height * 0.052)
    compare_box_w = int(width * 0.29)
    compare_gap = int(width * 0.055)
    operator_w = int(width * 0.1)
    compare_total_w = compare_box_w * 2 + compare_gap * 2 + operator_w
    compare_left_x = int((width - compare_total_w) / 2)
    compare_operator_x = compare_left_x + compare_box_w + compare_gap
    compare_right_x = compare_operator_x + operator_w + compare_gap
    compare_value_font = bubble.font(int(width * 0.052), bold=True)
    operator_font = bubble.font(int(width * 0.062), bold=True)

    if step.pair is not None and step.before is not None:
        left_idx, right_idx = step.pair
        left_value = step.before[left_idx]
        right_value = step.before[right_idx]
        compare_color = bubble.SWAP if left_value > right_value else bubble.SORTED
    else:
        left_value = None
        right_value = None
        compare_color = bubble.SORTED if step.label in {"lock", "sorted"} else bubble.MUTED

    for box_x, value in ((compare_left_x, left_value), (compare_right_x, right_value)):
        bubble.rounded_rect(draw, (box_x, compare_y, box_x + compare_box_w, compare_y + compare_h), 16, bubble.PANEL, compare_color, 3)
        if value is not None:
            value_text = str(value)
            draw.text(
                (
                    box_x + (compare_box_w - bubble.text_width(draw, value_text, compare_value_font)) / 2,
                    compare_y + int(compare_h * 0.13),
                ),
                value_text,
                font=compare_value_font,
                fill=compare_color,
            )
    draw.text(
        (
            compare_operator_x + (operator_w - bubble.text_width(draw, ">", operator_font)) / 2,
            compare_y + int(compare_h * 0.05),
        ),
        ">",
        font=operator_font,
        fill=compare_color if left_value is not None else bubble.MUTED,
    )

    stats_y = int(height * 0.854)
    stats_h = int(height * 0.052)
    stats_gap = int(width * 0.025)
    stats_w = int((width - margin * 2 - stats_gap) / 2)
    stats = (
        ("comparisons", step.comparisons, bubble.COMPARE),
        ("moves", step.moves, bubble.SWAP if step.label == "swap" else bubble.BAR),
    )
    for stat_index, (name, value, color) in enumerate(stats):
        x0 = margin + stat_index * (stats_w + stats_gap)
        x1 = x0 + stats_w
        bubble.rounded_rect(draw, (x0, stats_y, x1, stats_y + stats_h), 16, bubble.PANEL, bubble.GRID, 2)
        value_text = str(value)
        draw.text((x0 + 22, stats_y + int(stats_h * 0.14)), value_text, font=label_font, fill=color)
        draw.text((x0 + 22, stats_y + int(stats_h * 0.58)), name, font=small_font, fill=bubble.MUTED)

    tau_label = "Kendall tau sortedness"
    tau_score = f"{tau_fill:.2f}"
    tau_label_y = int(height * 0.922)
    tau_bar_y = int(height * 0.95)
    tau_bar_h = int(height * 0.012)
    tau_bar_x0 = margin
    tau_bar_x1 = width - margin
    draw.text((tau_bar_x0, tau_label_y), tau_label, font=small_font, fill=bubble.MUTED)
    draw.text((tau_bar_x1 - bubble.text_width(draw, tau_score, small_font), tau_label_y), tau_score, font=small_font, fill=bubble.SORTED)
    bubble.rounded_rect(draw, (tau_bar_x0, tau_bar_y, tau_bar_x1, tau_bar_y + tau_bar_h), 12, (15, 18, 25), bubble.GRID, 2)
    inner_x0 = tau_bar_x0 + 4
    inner_x1 = tau_bar_x1 - 4
    inner_y0 = tau_bar_y + 4
    inner_y1 = tau_bar_y + tau_bar_h - 4
    if tau_fill > 0 and inner_y1 > inner_y0:
        bubble.rounded_rect(draw, (inner_x0, inner_y0, bubble.lerp(inner_x0, inner_x1, tau_fill), inner_y1), 8, bubble.SORTED)

    return image


def planned_frames(steps: list[Step], fps: int) -> list[FrameState]:
    timeline: list[FrameState] = []
    for _ in range(int(fps * 1.0)):
        timeline.append(FrameState(steps[0]))

    for step in steps[1:]:
        if step.swapped:
            for frame in range(5):
                timeline.append(FrameState(step, frame / 4, "swap" if frame == 0 else None))
            timeline.append(FrameState(step))
        elif step.label == "compare" or step.label == "new_min":
            timeline.append(FrameState(step))
        elif step.label == "lock":
            for frame in range(5):
                timeline.append(FrameState(step, audio_event="lock" if frame == 0 else None))
        elif step.label == "sorted":
            for frame in range(int(fps * 4.0)):
                timeline.append(FrameState(step, audio_event="sorted" if frame == 0 else None))
        else:
            timeline.append(FrameState(step))
    return timeline


def output_path_for(slug: str, output_dir: Path) -> Path:
    return output_dir / f"{slug}_sort.mp4"


def thumbnail_path_for(slug: str, output_dir: Path) -> Path:
    return output_dir / f"{slug}_sort_thumbnail.png"


def render_one(args: argparse.Namespace, slug: str) -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required but was not found on PATH")

    spec = ALGORITHMS[slug]
    output = args.output if args.output and not args.all else output_path_for(slug, args.output_dir)
    thumbnail = args.thumbnail if args.thumbnail and not args.all else thumbnail_path_for(slug, args.output_dir)
    output.parent.mkdir(parents=True, exist_ok=True)

    values = bubble.seeded_values(args.bars, args.seed)
    steps = STEP_BUILDERS[slug](values)
    timeline = planned_frames(steps, args.fps)
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
                    spec=spec,
                    step=frame_state.step,
                    frame_number=frame_number,
                    total_frames=len(timeline),
                    swap_progress=frame_state.swap_progress,
                )
                if not thumbnail_saved and frame_number >= min(len(timeline) - 1, args.fps * 3):
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
                spec=spec,
                step=steps[-1],
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

    duration = len(timeline) / args.fps
    print(f"Rendered {output}")
    print(f"Rendered {thumbnail}")
    print(f"Algorithm: {spec.title.title()}")
    print(f"Frames: {len(timeline)} at {args.fps} fps ({duration:.1f}s)")
    print(f"Audio: {'on' if audio_enabled else 'off'}")
    print(f"Initial values: {values}")
    print(f"Comparisons: {steps[-1].comparisons}")
    print(f"Moves: {steps[-1].moves}")
    print(f"Swaps: {steps[-1].swaps}")
    print(f"Final inversions: {steps[-1].inversions}/{steps[-1].max_inversions}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", choices=sorted(ALGORITHMS), default="insertion")
    parser.add_argument("--all", action="store_true", help="render all configured elementary algorithm shorts")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--thumbnail", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--width", type=int, default=1080)
    parser.add_argument("--height", type=int, default=1920)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--bars", type=int, default=24)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--no-audio", action="store_true", help="render without operation tones")
    args = parser.parse_args()

    if args.all and (args.output or args.thumbnail):
        parser.error("--output and --thumbnail can only be used with a single --algorithm")
    if args.bars < 6:
        parser.error("--bars must be at least 6")
    if args.width <= 0 or args.height <= 0 or args.fps <= 0:
        parser.error("--width, --height, and --fps must be positive")
    return args


def main() -> None:
    args = parse_args()
    slugs = list(ALGORITHMS) if args.all else [args.algorithm]
    for slug in slugs:
        render_one(args, slug)


if __name__ == "__main__":
    main()
