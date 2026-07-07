#!/usr/bin/env python3
"""Render a sorting animation with live operation and memory meters."""

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
DEFAULT_OUTPUT = ROOT / "artifacts" / "shorts" / "merge_sort_operation_meters.mp4"
DEFAULT_THUMBNAIL = ROOT / "artifacts" / "shorts" / "merge_sort_operation_meters_thumbnail.png"
DEFAULT_OUTPUT_DIR = ROOT / "artifacts" / "shorts"

METER_COLORS = {
    "comparisons": bubble.COMPARE,
    "reads": (86, 176, 255),
    "writes": (45, 208, 184),
    "swaps": bubble.SWAP,
    "aux memory": (190, 132, 255),
}


@dataclass(frozen=True)
class AlgorithmSpec:
    slug: str
    title: str
    subtitle: str


ALGORITHMS = {
    "bubble": AlgorithmSpec("bubble", "BUBBLE SORT COSTS", "Adjacent comparisons drive reads, writes, and swaps."),
    "insertion": AlgorithmSpec("insertion", "INSERTION SORT COSTS", "A sorted prefix grows as values move backward."),
    "selection": AlgorithmSpec("selection", "SELECTION SORT COSTS", "Many reads find each minimum; few writes lock positions."),
    "merge": AlgorithmSpec("merge", "MERGE SORT COSTS", "The list sorts while operations and memory accumulate."),
    "gnome": AlgorithmSpec("gnome", "GNOME SORT COSTS", "A local walk trades extra comparisons for simple swaps."),
    "cocktail": AlgorithmSpec("cocktail", "COCKTAIL SORT COSTS", "Bidirectional bubbling moves values from both ends."),
    "odd-even": AlgorithmSpec("odd-even", "ODD-EVEN SORT COSTS", "Alternating neighbor passes expose parallel-style work."),
}


@dataclass(frozen=True)
class Metrics:
    comparisons: int = 0
    reads: int = 0
    writes: int = 0
    swaps: int = 0
    aux_memory: int = 0
    peak_aux_memory: int = 0


@dataclass(frozen=True)
class Step:
    values: tuple[int, ...]
    pair: tuple[int, int] | None
    compare_values: tuple[int, int] | None
    target: int | None
    active_range: tuple[int, int] | None
    label: str
    status_text: str
    metrics: Metrics
    before: tuple[int, ...] | None = None
    took_right: bool = False


@dataclass(frozen=True)
class FrameState:
    step: Step
    write_progress: float | None = None
    audio_event: str | None = None


def update_metrics(metrics: Metrics, **updates: int) -> Metrics:
    data = metrics.__dict__ | updates
    return Metrics(**data)


def in_place_start(values: list[int], status_text: str) -> tuple[list[int], Metrics, list[Step]]:
    arr = values[:]
    metrics = Metrics()
    steps = [Step(tuple(arr), None, None, None, None, "start", status_text, metrics)]
    return arr, metrics, steps


def add_in_place_step(
    steps: list[Step],
    arr: list[int],
    metrics: Metrics,
    *,
    label: str,
    status_text: str,
    pair: tuple[int, int] | None = None,
    before: tuple[int, ...] | None = None,
) -> None:
    compare_values = None
    if pair is not None and before is not None:
        compare_values = (before[pair[0]], before[pair[1]])
    steps.append(
        Step(
            tuple(arr),
            pair,
            compare_values,
            None,
            None,
            label,
            status_text,
            metrics,
            before=before,
            took_right=bool(compare_values and compare_values[0] > compare_values[1]),
        )
    )


def bubble_meter_steps(values: list[int]) -> list[Step]:
    arr, metrics, steps = in_place_start(values, "unsorted input")
    n = len(arr)
    for end in range(n - 1, 0, -1):
        swapped_this_pass = False
        for i in range(end):
            before = tuple(arr)
            metrics = update_metrics(metrics, comparisons=metrics.comparisons + 1, reads=metrics.reads + 2)
            should_swap = arr[i] > arr[i + 1]
            if should_swap:
                arr[i], arr[i + 1] = arr[i + 1], arr[i]
                metrics = update_metrics(metrics, writes=metrics.writes + 2, swaps=metrics.swaps + 1)
                swapped_this_pass = True
            add_in_place_step(
                steps,
                arr,
                metrics,
                label="swap" if should_swap else "compare",
                status_text=f"unsorted end: {end + 1}",
                pair=(i, i + 1),
                before=before,
            )
        add_in_place_step(steps, arr, metrics, label="lock", status_text=f"locked largest: {n - end}/{n}")
        if not swapped_this_pass:
            break
    add_in_place_step(steps, arr, metrics, label="sorted", status_text="sorted")
    assert arr == sorted(values)
    return steps


def insertion_meter_steps(values: list[int]) -> list[Step]:
    arr, metrics, steps = in_place_start(values, "sorted prefix: 1")
    n = len(arr)
    for i in range(1, n):
        j = i
        while j > 0:
            before = tuple(arr)
            metrics = update_metrics(metrics, comparisons=metrics.comparisons + 1, reads=metrics.reads + 2)
            should_swap = arr[j - 1] > arr[j]
            if should_swap:
                arr[j - 1], arr[j] = arr[j], arr[j - 1]
                metrics = update_metrics(metrics, writes=metrics.writes + 2, swaps=metrics.swaps + 1)
            add_in_place_step(
                steps,
                arr,
                metrics,
                label="swap" if should_swap else "compare",
                status_text=f"insert index: {i}",
                pair=(j - 1, j),
                before=before,
            )
            if not should_swap:
                break
            j -= 1
        add_in_place_step(steps, arr, metrics, label="lock", status_text=f"sorted prefix: {i + 1}/{n}")
    add_in_place_step(steps, arr, metrics, label="sorted", status_text="sorted")
    assert arr == sorted(values)
    return steps


def selection_meter_steps(values: list[int]) -> list[Step]:
    arr, metrics, steps = in_place_start(values, "unsorted input")
    n = len(arr)
    for i in range(n - 1):
        min_index = i
        for j in range(i + 1, n):
            before = tuple(arr)
            previous_min = min_index
            metrics = update_metrics(metrics, comparisons=metrics.comparisons + 1, reads=metrics.reads + 2)
            if arr[j] < arr[min_index]:
                min_index = j
                label = "new_min"
            else:
                label = "compare"
            add_in_place_step(
                steps,
                arr,
                metrics,
                label=label,
                status_text=f"scan minimum for slot {i + 1}",
                pair=(previous_min, j),
                before=before,
            )
        if min_index != i:
            before = tuple(arr)
            arr[i], arr[min_index] = arr[min_index], arr[i]
            metrics = update_metrics(metrics, reads=metrics.reads + 2, writes=metrics.writes + 2, swaps=metrics.swaps + 1)
            add_in_place_step(
                steps,
                arr,
                metrics,
                label="swap",
                status_text=f"lock slot: {i + 1}",
                pair=(i, min_index),
                before=before,
            )
        add_in_place_step(steps, arr, metrics, label="lock", status_text=f"locked slots: {i + 1}/{n}")
    add_in_place_step(steps, arr, metrics, label="sorted", status_text="sorted")
    assert arr == sorted(values)
    return steps


def gnome_meter_steps(values: list[int]) -> list[Step]:
    arr, metrics, steps = in_place_start(values, "scan index: 0")
    n = len(arr)
    index = 0
    while index < n:
        if index == 0:
            index += 1
            continue
        before = tuple(arr)
        metrics = update_metrics(metrics, comparisons=metrics.comparisons + 1, reads=metrics.reads + 2)
        should_swap = arr[index] < arr[index - 1]
        if should_swap:
            arr[index], arr[index - 1] = arr[index - 1], arr[index]
            metrics = update_metrics(metrics, writes=metrics.writes + 2, swaps=metrics.swaps + 1)
        add_in_place_step(
            steps,
            arr,
            metrics,
            label="swap" if should_swap else "compare",
            status_text=f"scan index: {index}/{n}",
            pair=(index - 1, index),
            before=before,
        )
        index = index - 1 if should_swap else index + 1
    add_in_place_step(steps, arr, metrics, label="sorted", status_text="sorted")
    assert arr == sorted(values)
    return steps


def cocktail_meter_steps(values: list[int]) -> list[Step]:
    arr, metrics, steps = in_place_start(values, "unsorted input")
    n = len(arr)
    start = 0
    end = n - 1
    swapped_this_pass = True
    while swapped_this_pass:
        swapped_this_pass = False
        for i in range(start, end):
            before = tuple(arr)
            metrics = update_metrics(metrics, comparisons=metrics.comparisons + 1, reads=metrics.reads + 2)
            should_swap = arr[i] > arr[i + 1]
            if should_swap:
                arr[i], arr[i + 1] = arr[i + 1], arr[i]
                metrics = update_metrics(metrics, writes=metrics.writes + 2, swaps=metrics.swaps + 1)
                swapped_this_pass = True
            add_in_place_step(
                steps,
                arr,
                metrics,
                label="swap" if should_swap else "compare",
                status_text=f"forward pass: {start + 1}-{end + 1}",
                pair=(i, i + 1),
                before=before,
            )
        add_in_place_step(steps, arr, metrics, label="lock", status_text=f"locked high end: {n - end}/{n}")
        if not swapped_this_pass:
            break
        swapped_this_pass = False
        end -= 1
        for i in range(end - 1, start - 1, -1):
            before = tuple(arr)
            metrics = update_metrics(metrics, comparisons=metrics.comparisons + 1, reads=metrics.reads + 2)
            should_swap = arr[i] > arr[i + 1]
            if should_swap:
                arr[i], arr[i + 1] = arr[i + 1], arr[i]
                metrics = update_metrics(metrics, writes=metrics.writes + 2, swaps=metrics.swaps + 1)
                swapped_this_pass = True
            add_in_place_step(
                steps,
                arr,
                metrics,
                label="swap" if should_swap else "compare",
                status_text=f"backward pass: {start + 1}-{end + 1}",
                pair=(i, i + 1),
                before=before,
            )
        add_in_place_step(steps, arr, metrics, label="lock", status_text=f"locked low end: {start + 1}/{n}")
        start += 1
    add_in_place_step(steps, arr, metrics, label="sorted", status_text="sorted")
    assert arr == sorted(values)
    return steps


def odd_even_meter_steps(values: list[int]) -> list[Step]:
    arr, metrics, steps = in_place_start(values, "phase: odd")
    n = len(arr)
    sorted_pass = False
    pass_number = 0
    while not sorted_pass:
        sorted_pass = True
        pass_number += 1
        for phase_name, first in (("odd", 1), ("even", 0)):
            for i in range(first, n - 1, 2):
                before = tuple(arr)
                metrics = update_metrics(metrics, comparisons=metrics.comparisons + 1, reads=metrics.reads + 2)
                should_swap = arr[i] > arr[i + 1]
                if should_swap:
                    arr[i], arr[i + 1] = arr[i + 1], arr[i]
                    metrics = update_metrics(metrics, writes=metrics.writes + 2, swaps=metrics.swaps + 1)
                    sorted_pass = False
                add_in_place_step(
                    steps,
                    arr,
                    metrics,
                    label="swap" if should_swap else "compare",
                    status_text=f"{phase_name} phase: {pass_number}",
                    pair=(i, i + 1),
                    before=before,
                )
    add_in_place_step(steps, arr, metrics, label="sorted", status_text="sorted")
    assert arr == sorted(values)
    return steps


def merge_meter_steps(values: list[int]) -> list[Step]:
    arr = values[:]
    metrics = Metrics()
    steps: list[Step] = [
        Step(tuple(arr), None, None, None, None, "start", "unsorted input", metrics)
    ]

    def set_metrics(**updates: int) -> None:
        nonlocal metrics
        data = metrics.__dict__ | updates
        metrics = Metrics(**data)

    def add_step(
        *,
        label: str,
        status_text: str,
        pair: tuple[int, int] | None = None,
        compare_values: tuple[int, int] | None = None,
        target: int | None = None,
        active_range: tuple[int, int] | None = None,
        before: tuple[int, ...] | None = None,
        took_right: bool = False,
    ) -> None:
        steps.append(
            Step(
                tuple(arr),
                pair,
                compare_values,
                target,
                active_range,
                label,
                status_text,
                metrics,
                before=before,
                took_right=took_right,
            )
        )

    def sort_range(left: int, right: int) -> None:
        if right - left <= 1:
            return

        mid = (left + right) // 2
        sort_range(left, mid)
        sort_range(mid, right)

        run_size = right - left
        set_metrics(aux_memory=run_size, peak_aux_memory=max(metrics.peak_aux_memory, run_size))
        add_step(
            label="allocate",
            status_text=f"allocate temp buffer: {run_size}",
            active_range=(left, right),
        )

        left_values = arr[left:mid]
        right_values = arr[mid:right]
        i = 0
        j = 0
        target = left

        while i < len(left_values) and j < len(right_values):
            left_value = left_values[i]
            right_value = right_values[j]
            set_metrics(comparisons=metrics.comparisons + 1, reads=metrics.reads + 2)
            took_right = right_value < left_value
            add_step(
                label="compare",
                status_text="compare buffer fronts",
                pair=(left + i, mid + j),
                compare_values=(left_value, right_value),
                target=target,
                active_range=(left, right),
                took_right=took_right,
            )

            before = tuple(arr)
            if took_right:
                arr[target] = right_value
                j += 1
            else:
                arr[target] = left_value
                i += 1
            set_metrics(writes=metrics.writes + 1)
            add_step(
                label="write",
                status_text="write selected value",
                pair=(left + i - (0 if took_right else 1), mid + j - (1 if took_right else 0)),
                compare_values=(left_value, right_value),
                target=target,
                active_range=(left, right),
                before=before,
                took_right=took_right,
            )
            target += 1

        while i < len(left_values):
            before = tuple(arr)
            set_metrics(reads=metrics.reads + 1, writes=metrics.writes + 1)
            arr[target] = left_values[i]
            add_step(
                label="write",
                status_text="copy left remainder",
                target=target,
                active_range=(left, right),
                before=before,
            )
            i += 1
            target += 1

        while j < len(right_values):
            before = tuple(arr)
            set_metrics(reads=metrics.reads + 1, writes=metrics.writes + 1)
            arr[target] = right_values[j]
            add_step(
                label="write",
                status_text="copy right remainder",
                target=target,
                active_range=(left, right),
                before=before,
            )
            j += 1
            target += 1

        set_metrics(aux_memory=0)
        add_step(
            label="release",
            status_text=f"release temp buffer: {run_size}",
            active_range=(left, right),
        )

    sort_range(0, len(arr))
    add_step(label="sorted", status_text="sorted", active_range=(0, len(arr)))
    assert arr == sorted(values)
    return steps


STEP_BUILDERS = {
    "bubble": bubble_meter_steps,
    "insertion": insertion_meter_steps,
    "selection": selection_meter_steps,
    "merge": merge_meter_steps,
    "gnome": gnome_meter_steps,
    "cocktail": cocktail_meter_steps,
    "odd-even": odd_even_meter_steps,
}


def planned_frames(steps: list[Step], fps: int) -> list[FrameState]:
    timeline: list[FrameState] = []
    for _ in range(int(fps * 1.0)):
        timeline.append(FrameState(steps[0]))

    for step in steps[1:]:
        if step.label in {"write", "swap"}:
            for frame in range(5):
                timeline.append(FrameState(step, frame / 4, "swap" if frame == 0 else None))
            timeline.append(FrameState(step))
        elif step.label in {"allocate", "release"}:
            for frame in range(4):
                timeline.append(FrameState(step, audio_event="lock" if frame == 0 and step.label == "release" else None))
        elif step.label == "sorted":
            for frame in range(int(fps * 4.0)):
                timeline.append(FrameState(step, audio_event="sorted" if frame == 0 else None))
        else:
            timeline.append(FrameState(step))
    return timeline


def metric_maxes(final_metrics: Metrics) -> dict[str, int]:
    return {
        "comparisons": max(1, final_metrics.comparisons),
        "reads": max(1, final_metrics.reads),
        "writes": max(1, final_metrics.writes),
        "swaps": max(1, final_metrics.swaps),
        "aux memory": max(1, final_metrics.peak_aux_memory),
    }


def draw_meter(
    draw: ImageDraw.ImageDraw,
    *,
    x0: int,
    y0: int,
    width: int,
    height: int,
    label: str,
    value: int,
    max_value: int,
    color: tuple[int, int, int],
    label_font,
    value_font,
    peak_value: int | None = None,
) -> None:
    bubble.rounded_rect(draw, (x0, y0, x0 + width, y0 + height), 14, bubble.PANEL, bubble.GRID, 2)
    pad = 18
    label_y = y0 + 12
    draw.text((x0 + pad, label_y), label, font=label_font, fill=bubble.MUTED)
    value_text = str(value)
    draw.text((x0 + width - pad - bubble.text_width(draw, value_text, value_font), label_y - 2), value_text, font=value_font, fill=color)

    bar_x0 = x0 + pad
    bar_x1 = x0 + width - pad
    bar_y0 = y0 + int(height * 0.56)
    bar_y1 = y0 + int(height * 0.78)
    bubble.rounded_rect(draw, (bar_x0, bar_y0, bar_x1, bar_y1), 10, (13, 17, 25), bubble.GRID, 2)
    fill = max(0.0, min(1.0, value / max_value))
    inset = max(1, min(3, int((bar_y1 - bar_y0) / 3)))
    inner_y0 = bar_y0 + inset
    inner_y1 = bar_y1 - inset
    if fill > 0 and inner_y1 >= inner_y0:
        bubble.rounded_rect(
            draw,
            (bar_x0 + inset, inner_y0, bubble.lerp(bar_x0 + inset, bar_x1 - inset, fill), inner_y1),
            8,
            color,
        )

    if peak_value is not None and max_value > 0:
        peak_fill = max(0.0, min(1.0, peak_value / max_value))
        peak_x = bubble.lerp(bar_x0 + 3, bar_x1 - 3, peak_fill)
        draw.line((peak_x, bar_y0 - 5, peak_x, bar_y1 + 5), fill=bubble.TEXT, width=2)


def draw_frame(
    *,
    width: int,
    height: int,
    spec: AlgorithmSpec,
    step: Step,
    final_metrics: Metrics,
    maxes: dict[str, int],
    frame_number: int,
    total_frames: int,
    write_progress: float | None,
) -> Image.Image:
    image = bubble.gradient_background(width, height).copy()
    draw = ImageDraw.Draw(image)

    title_font = bubble.font(int(width * 0.072), bold=True)
    subtitle_font = bubble.font(int(width * 0.034))
    label_font = bubble.font(int(width * 0.029), bold=True)
    value_font = bubble.font(int(width * 0.034), bold=True)
    tiny_font = bubble.font(int(width * 0.022), bold=True)
    meter_label_font = bubble.font(int(width * 0.026), bold=True)
    meter_value_font = bubble.font(int(width * 0.034), bold=True)

    margin = int(width * 0.065)
    title_font = bubble.fit_font(draw, spec.title, int(width * 0.072), int(width * 0.9), bold=True, min_size=int(width * 0.052))
    bubble.draw_centered_text(draw, int(height * 0.045), spec.title, title_font, bubble.TEXT, width)
    bubble.draw_centered_text(draw, int(height * 0.096), spec.subtitle, subtitle_font, bubble.MUTED, width)

    badge_y = int(height * 0.145)
    badge_h = int(height * 0.052)
    bubble.rounded_rect(draw, (margin, badge_y, width - margin, badge_y + badge_h), 18, bubble.PANEL, bubble.GRID, 2)
    status = {
        "start": "Unsorted input",
        "allocate": "Allocate temp memory",
        "compare": "Compare buffered values",
        "new_min": "New minimum found",
        "swap": "Swap values",
        "write": "Write next value",
        "lock": "Position locked",
        "release": "Release temp memory",
        "sorted": "Sorted",
    }[step.label]
    status_color = (
        METER_COLORS["aux memory"]
        if step.label in {"allocate", "release"}
        else bubble.COMPARE
        if step.label in {"compare", "new_min"}
        else bubble.SWAP
        if step.label == "swap"
        else bubble.SORTED
        if step.label in {"lock", "sorted"}
        else METER_COLORS["writes"]
        if step.label == "write"
        else bubble.TEXT
    )
    draw.text((margin + 24, badge_y + int(badge_h * 0.22)), status, font=label_font, fill=status_color)
    draw.text(
        (width - margin - 24 - bubble.text_width(draw, step.status_text, label_font), badge_y + int(badge_h * 0.22)),
        step.status_text,
        font=label_font,
        fill=bubble.MUTED,
    )

    chart_top = int(height * 0.225)
    chart_bottom = int(height * 0.62)
    chart_left = int(width * 0.09)
    chart_right = int(width * 0.91)
    chart_height = chart_bottom - chart_top
    max_value = max(step.values)
    bar_gap = max(4, int(width * 0.006))
    bar_step = (chart_right - chart_left) / len(step.values)
    bar_width = max(10, int(bar_step - bar_gap))

    for grid_idx in range(5):
        y = chart_top + chart_height * grid_idx / 4
        draw.line((chart_left, y, chart_right, y), fill=bubble.GRID, width=2)

    if step.active_range is not None:
        range_left, range_right = step.active_range
        x0 = chart_left + bar_step * range_left
        x1 = chart_left + bar_step * range_right
        bubble.rounded_rect(draw, (x0, chart_top - 12, x1, chart_bottom + 10), 10, (18, 23, 33), bubble.GRID, 2)

    active = set(step.pair or ())
    if step.target is not None:
        active.add(step.target)

    previous_values = step.before if step.before is not None else step.values
    visuals: list[tuple[float, int, int]] = []
    if step.label == "swap" and write_progress is not None and step.before is not None and step.pair is not None:
        left_idx, right_idx = step.pair
        eased = bubble.ease(write_progress)
        left_x = chart_left + bar_step * left_idx + bar_step * 0.12
        right_x = chart_left + bar_step * right_idx + bar_step * 0.12
        for idx, value in enumerate(step.before):
            x = chart_left + bar_step * idx + bar_step * 0.12
            if idx == left_idx:
                x = bubble.lerp(left_x, right_x, eased)
            elif idx == right_idx:
                x = bubble.lerp(right_x, left_x, eased)
            visuals.append((x, value, idx))
    else:
        for idx, value in enumerate(step.values):
            x = chart_left + bar_step * idx + bar_step * 0.12
            visuals.append((x, value, idx))

    for x, value, idx in sorted(visuals, key=lambda item: item[0]):
        shown_value = value
        if idx == step.target and write_progress is not None and step.before is not None:
            shown_value = int(round(bubble.lerp(previous_values[idx], value, bubble.ease(write_progress))))
        bar_h = int((shown_value / max_value) * chart_height)
        y0 = chart_bottom - bar_h
        y1 = chart_bottom
        if step.label == "sorted":
            color = bubble.SORTED
        elif idx in active and step.label == "swap":
            color = bubble.SWAP
        elif idx == step.target and step.label == "write":
            color = METER_COLORS["writes"]
        elif idx in active:
            color = bubble.COMPARE
        elif step.active_range is not None and step.active_range[0] <= idx < step.active_range[1]:
            color = bubble.BAR_ALT
        else:
            color = bubble.BAR if idx % 2 == 0 else bubble.BAR_ALT
        bubble.rounded_rect(draw, (x, y0, x + bar_width, y1), 8, color)
        if idx in active or step.label == "sorted":
            value_text = str(value)
            draw.text((x + (bar_width - bubble.text_width(draw, value_text, tiny_font)) / 2, y0 - int(height * 0.022)), value_text, font=tiny_font, fill=bubble.TEXT)

    draw.line((chart_left, chart_bottom + int(height * 0.016), chart_right, chart_bottom + int(height * 0.016)), fill=bubble.GRID, width=3)

    compare_y = int(height * 0.648)
    compare_h = int(height * 0.046)
    compare_box_w = int(width * 0.24)
    compare_gap = int(width * 0.045)
    operator_w = int(width * 0.08)
    total_w = compare_box_w * 2 + compare_gap * 2 + operator_w
    left_x = int((width - total_w) / 2)
    op_x = left_x + compare_box_w + compare_gap
    right_x = op_x + operator_w + compare_gap
    compare_font = bubble.font(int(width * 0.045), bold=True)
    operator_font = bubble.font(int(width * 0.054), bold=True)

    if step.compare_values is not None:
        left_value, right_value = step.compare_values
        compare_color = bubble.SWAP if left_value > right_value else bubble.SORTED
    else:
        left_value = None
        right_value = None
        compare_color = bubble.MUTED
    for box_x, value in ((left_x, left_value), (right_x, right_value)):
        bubble.rounded_rect(draw, (box_x, compare_y, box_x + compare_box_w, compare_y + compare_h), 14, bubble.PANEL, compare_color, 3)
        if value is not None:
            text = str(value)
            draw.text((box_x + (compare_box_w - bubble.text_width(draw, text, compare_font)) / 2, compare_y + int(compare_h * 0.13)), text, font=compare_font, fill=compare_color)
    draw.text((op_x + (operator_w - bubble.text_width(draw, ">", operator_font)) / 2, compare_y + int(compare_h * 0.03)), ">", font=operator_font, fill=compare_color)

    meter_top = int(height * 0.706)
    meter_h = int(height * 0.043)
    meter_gap = int(height * 0.009)
    meter_w = width - margin * 2
    rows = [
        ("comparisons", step.metrics.comparisons, maxes["comparisons"], METER_COLORS["comparisons"], None),
        ("reads", step.metrics.reads, maxes["reads"], METER_COLORS["reads"], None),
        ("writes", step.metrics.writes, maxes["writes"], METER_COLORS["writes"], None),
        ("swaps", step.metrics.swaps, maxes["swaps"], METER_COLORS["swaps"], None),
        ("peak aux memory", step.metrics.peak_aux_memory, maxes["aux memory"], METER_COLORS["aux memory"], None),
    ]
    for row, (name, value, max_value_for_meter, color, peak_value) in enumerate(rows):
        draw_meter(
            draw,
            x0=margin,
            y0=meter_top + row * (meter_h + meter_gap),
            width=meter_w,
            height=meter_h,
            label=name,
            value=value,
            max_value=max_value_for_meter,
            color=color,
            label_font=meter_label_font,
            value_font=meter_value_font,
            peak_value=peak_value,
        )

    final_text = (
        f"final: {final_metrics.comparisons} comparisons, "
        f"{final_metrics.reads} reads, {final_metrics.writes} writes, "
        f"peak aux {final_metrics.peak_aux_memory}"
    )
    footer_font = bubble.font(int(width * 0.022))
    draw.text((margin, int(height * 0.966)), final_text, font=footer_font, fill=bubble.MUTED)
    return image


def output_path_for(slug: str, output_dir: Path) -> Path:
    return output_dir / f"{slug}_sort_operation_meters.mp4"


def thumbnail_path_for(slug: str, output_dir: Path) -> Path:
    return output_dir / f"{slug}_sort_operation_meters_thumbnail.png"


def render_one(args: argparse.Namespace, slug: str) -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required but was not found on PATH")

    spec = ALGORITHMS[slug]
    output = args.output if args.output and not args.all else output_path_for(slug, args.output_dir)
    thumbnail = args.thumbnail if args.thumbnail and not args.all else thumbnail_path_for(slug, args.output_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    values = bubble.seeded_values(args.bars, args.seed)
    steps = STEP_BUILDERS[slug](values)
    final_metrics = steps[-1].metrics
    maxes = metric_maxes(final_metrics)
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
                    final_metrics=final_metrics,
                    maxes=maxes,
                    frame_number=frame_number,
                    total_frames=len(timeline),
                    write_progress=frame_state.write_progress,
                )
                if not thumbnail_saved and frame_number >= min(len(timeline) - 1, args.fps * 4):
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
                final_metrics=final_metrics,
                maxes=maxes,
                frame_number=len(timeline) - 1,
                total_frames=len(timeline),
                write_progress=None,
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
    print(f"Comparisons: {final_metrics.comparisons}")
    print(f"Reads: {final_metrics.reads}")
    print(f"Writes: {final_metrics.writes}")
    print(f"Swaps: {final_metrics.swaps}")
    print(f"Peak aux memory: {final_metrics.peak_aux_memory}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", choices=sorted(ALGORITHMS), default="merge")
    parser.add_argument("--all", action="store_true", help="render all configured operation-meter shorts")
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


if __name__ == "__main__":
    parsed_args = parse_args()
    slugs = list(ALGORITHMS) if parsed_args.all else [parsed_args.algorithm]
    for algorithm_slug in slugs:
        render_one(parsed_args, algorithm_slug)
