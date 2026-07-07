#!/usr/bin/env python3
"""Render a Shorts-ready Merge Sort animation."""

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
DEFAULT_OUTPUT = ROOT / "artifacts" / "shorts" / "merge_sort.mp4"
DEFAULT_THUMBNAIL = ROOT / "artifacts" / "shorts" / "merge_sort_thumbnail.png"


@dataclass(frozen=True)
class MergeStep:
    values: tuple[int, ...]
    pair: tuple[int, int] | None
    compare_values: tuple[int, int] | None
    target: int | None
    active_range: tuple[int, int] | None
    sorted_span: int
    label: str
    comparisons: int
    moves: int
    inversions: int
    max_inversions: int
    took_right: bool = False
    before: tuple[int, ...] | None = None


@dataclass(frozen=True)
class FrameState:
    step: MergeStep
    write_progress: float | None = None
    audio_event: str | None = None


def merge_sort_steps(values: list[int]) -> list[MergeStep]:
    arr = values[:]
    comparisons = 0
    moves = 0
    sorted_span = 1
    max_inversions = len(arr) * (len(arr) - 1) // 2
    steps = [
        MergeStep(
            tuple(arr),
            None,
            None,
            None,
            None,
            sorted_span,
            "start",
            comparisons,
            moves,
            bubble.inversion_count(arr),
            max_inversions,
        )
    ]

    def add_step(
        *,
        label: str,
        pair: tuple[int, int] | None = None,
        compare_values: tuple[int, int] | None = None,
        target: int | None = None,
        active_range: tuple[int, int] | None = None,
        took_right: bool = False,
        before: tuple[int, ...] | None = None,
    ) -> None:
        steps.append(
            MergeStep(
                tuple(arr),
                pair,
                compare_values,
                target,
                active_range,
                sorted_span,
                label,
                comparisons,
                moves,
                bubble.inversion_count(arr),
                max_inversions,
                took_right=took_right,
                before=before,
            )
        )

    def sort_range(left: int, right: int) -> None:
        nonlocal comparisons, moves, sorted_span
        if right - left <= 1:
            return

        mid = (left + right) // 2
        sort_range(left, mid)
        sort_range(mid, right)

        before_merge = arr[:]
        left_values = before_merge[left:mid]
        right_values = before_merge[mid:right]
        i = 0
        j = 0
        target = left

        while i < len(left_values) and j < len(right_values):
            left_value = left_values[i]
            right_value = right_values[j]
            left_index = left + i
            right_index = mid + j
            comparisons += 1
            took_right = right_value < left_value
            add_step(
                label="compare",
                pair=(left_index, right_index),
                compare_values=(left_value, right_value),
                target=target,
                active_range=(left, right),
                took_right=took_right,
            )

            before_write = tuple(arr)
            if took_right:
                arr[target] = right_value
                j += 1
            else:
                arr[target] = left_value
                i += 1
            moves += 1
            add_step(
                label="write",
                pair=(left_index, right_index),
                compare_values=(left_value, right_value),
                target=target,
                active_range=(left, right),
                took_right=took_right,
                before=before_write,
            )
            target += 1

        while i < len(left_values):
            before_write = tuple(arr)
            arr[target] = left_values[i]
            moves += 1
            add_step(
                label="write",
                target=target,
                active_range=(left, right),
                before=before_write,
            )
            i += 1
            target += 1

        while j < len(right_values):
            before_write = tuple(arr)
            arr[target] = right_values[j]
            moves += 1
            add_step(
                label="write",
                target=target,
                active_range=(left, right),
                before=before_write,
            )
            j += 1
            target += 1

        sorted_span = max(sorted_span, right - left)
        add_step(label="lock", active_range=(left, right))

    sort_range(0, len(arr))
    sorted_span = len(arr)
    add_step(label="sorted", active_range=(0, len(arr)))
    assert arr == sorted(values)
    return steps


def draw_frame(
    *,
    width: int,
    height: int,
    step: MergeStep,
    frame_number: int,
    total_frames: int,
    total_moves: int,
    write_progress: float | None = None,
) -> Image.Image:
    image = bubble.gradient_background(width, height).copy()
    draw = ImageDraw.Draw(image)

    title_font = bubble.font(int(width * 0.085), bold=True)
    subtitle_font = bubble.font(int(width * 0.038))
    label_font = bubble.font(int(width * 0.042), bold=True)
    small_font = bubble.font(int(width * 0.029))
    tiny_font = bubble.font(int(width * 0.024), bold=True)

    margin = int(width * 0.065)
    bubble.draw_centered_text(draw, int(height * 0.055), "MERGE SORT", title_font, bubble.TEXT, width)
    bubble.draw_centered_text(
        draw,
        int(height * 0.115),
        "Split recursively. Merge sorted halves.",
        subtitle_font,
        bubble.MUTED,
        width,
    )

    badge_y = int(height * 0.17)
    badge_h = int(height * 0.052)
    bubble.rounded_rect(draw, (margin, badge_y, width - margin, badge_y + badge_h), 18, bubble.PANEL, bubble.GRID, 2)

    label = {
        "start": "Unsorted input",
        "compare": "Compare front values",
        "write": "Write next value",
        "lock": "Sorted run complete",
        "sorted": "Sorted",
    }[step.label]
    label_color = (
        bubble.SWAP
        if step.label == "write" and step.took_right
        else bubble.SORTED
        if step.label in {"lock", "sorted"}
        else bubble.COMPARE
        if step.label == "compare"
        else bubble.TEXT
    )
    span_text = f"sorted span: {step.sorted_span}/{len(step.values)}"
    span_x = width - margin - bubble.text_width(draw, span_text, small_font) - 24
    label_max_width = int(span_x - (margin + 24) - 28)
    fitted_label_font = bubble.fit_font(
        draw,
        label,
        int(width * 0.042),
        label_max_width,
        bold=True,
        min_size=int(width * 0.026),
    )
    draw.text((margin + 24, badge_y + int(badge_h * 0.22)), label, font=fitted_label_font, fill=label_color)
    draw.text((span_x, badge_y + int(badge_h * 0.31)), span_text, font=small_font, fill=bubble.MUTED)

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

    if step.active_range is not None:
        range_left, range_right = step.active_range
        x0 = chart_left + bar_step * range_left
        x1 = chart_left + bar_step * range_right
        bubble.rounded_rect(draw, (x0, chart_top - 14, x1, chart_bottom + 10), 10, (18, 23, 33), bubble.GRID, 2)

    active = set(step.pair or ())
    if step.target is not None:
        active.add(step.target)

    previous_values = step.before if step.before is not None else step.values
    for idx, value in enumerate(step.values):
        x = chart_left + bar_step * idx + bar_step * 0.12
        shown_value = value
        if idx == step.target and write_progress is not None and step.before is not None:
            shown_value = int(round(bubble.lerp(previous_values[idx], value, bubble.ease(write_progress))))
        bar_h = int((shown_value / max_value) * chart_height)
        y0 = chart_bottom - bar_h
        y1 = chart_bottom

        if step.label == "sorted":
            color = bubble.SORTED
        elif idx == step.target and step.label == "write":
            color = bubble.SWAP if step.took_right else bubble.SORTED
        elif idx in active:
            color = bubble.COMPARE
        elif step.active_range is not None and step.active_range[0] <= idx < step.active_range[1]:
            color = bubble.BAR_ALT
        else:
            color = bubble.BAR if idx % 2 == 0 else bubble.BAR_ALT

        bubble.rounded_rect(draw, (x, y0, x + bar_width, y1), 8, color)

        if idx in active or step.label == "sorted":
            value_text = str(value)
            draw.text(
                (x + (bar_width - bubble.text_width(draw, value_text, tiny_font)) / 2, y0 - int(height * 0.026)),
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

    if step.compare_values is not None:
        left_value, right_value = step.compare_values
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
        ("moves", step.moves, bubble.SWAP if step.label == "write" else bubble.BAR),
    )
    for stat_index, (name, value, color) in enumerate(stats):
        x0 = margin + stat_index * (stats_w + stats_gap)
        x1 = x0 + stats_w
        bubble.rounded_rect(draw, (x0, stats_y, x1, stats_y + stats_h), 16, bubble.PANEL, bubble.GRID, 2)
        value_text = str(value)
        draw.text((x0 + 22, stats_y + int(stats_h * 0.14)), value_text, font=label_font, fill=color)
        draw.text((x0 + 22, stats_y + int(stats_h * 0.58)), name, font=small_font, fill=bubble.MUTED)

    progress_fill = step.moves / max(1, total_moves)
    progress_fill = max(0.0, min(1.0, progress_fill))
    progress_label = "merge progress"
    progress_score = f"{progress_fill:.2f}"
    progress_label_y = int(height * 0.922)
    progress_bar_y = int(height * 0.95)
    progress_bar_h = int(height * 0.012)
    progress_bar_x0 = margin
    progress_bar_x1 = width - margin
    draw.text((progress_bar_x0, progress_label_y), progress_label, font=small_font, fill=bubble.MUTED)
    draw.text(
        (progress_bar_x1 - bubble.text_width(draw, progress_score, small_font), progress_label_y),
        progress_score,
        font=small_font,
        fill=bubble.SORTED,
    )
    bubble.rounded_rect(draw, (progress_bar_x0, progress_bar_y, progress_bar_x1, progress_bar_y + progress_bar_h), 12, (15, 18, 25), bubble.GRID, 2)
    inner_x0 = progress_bar_x0 + 4
    inner_x1 = progress_bar_x1 - 4
    inner_y0 = progress_bar_y + 4
    inner_y1 = progress_bar_y + progress_bar_h - 4
    if progress_fill > 0 and inner_y1 > inner_y0:
        bubble.rounded_rect(draw, (inner_x0, inner_y0, bubble.lerp(inner_x0, inner_x1, progress_fill), inner_y1), 8, bubble.SORTED)

    return image


def planned_frames(steps: list[MergeStep], fps: int) -> list[FrameState]:
    timeline: list[FrameState] = []
    for _ in range(int(fps * 1.0)):
        timeline.append(FrameState(steps[0]))

    for step in steps[1:]:
        if step.label == "write":
            for frame in range(5):
                timeline.append(FrameState(step, frame / 4, "swap" if frame == 0 else None))
            timeline.append(FrameState(step))
        elif step.label == "compare":
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


def render_video(args: argparse.Namespace) -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required but was not found on PATH")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    values = bubble.seeded_values(args.bars, args.seed)
    steps = merge_sort_steps(values)
    total_moves = steps[-1].moves
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
                    total_moves=total_moves,
                    write_progress=frame_state.write_progress,
                )
                if not thumbnail_saved and frame_number >= min(len(timeline) - 1, args.fps * 3):
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
                total_moves=total_moves,
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

    duration = len(timeline) / args.fps
    print(f"Rendered {args.output}")
    print(f"Rendered {args.thumbnail}")
    print(f"Frames: {len(timeline)} at {args.fps} fps ({duration:.1f}s)")
    print(f"Audio: {'on' if audio_enabled else 'off'}")
    print(f"Initial values: {values}")
    print(f"Comparisons: {steps[-1].comparisons}")
    print(f"Moves: {steps[-1].moves}")
    print(f"Final inversions: {steps[-1].inversions}/{steps[-1].max_inversions}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--thumbnail", type=Path, default=DEFAULT_THUMBNAIL)
    parser.add_argument("--width", type=int, default=1080)
    parser.add_argument("--height", type=int, default=1920)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--bars", type=int, default=24)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--no-audio", action="store_true", help="render without operation tones")
    args = parser.parse_args()

    if args.bars < 6:
        parser.error("--bars must be at least 6")
    if args.width <= 0 or args.height <= 0 or args.fps <= 0:
        parser.error("--width, --height, and --fps must be positive")

    return args


if __name__ == "__main__":
    render_video(parse_args())
