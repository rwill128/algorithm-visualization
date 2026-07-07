#!/usr/bin/env python3
"""Render a Shorts-ready Bubble Sort animation."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import math
import random
import shutil
import subprocess
import tempfile
import wave
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "artifacts" / "shorts" / "bubble_sort.mp4"
DEFAULT_THUMBNAIL = ROOT / "artifacts" / "shorts" / "bubble_sort_thumbnail.png"

FONT_REGULAR = Path("/System/Library/Fonts/Supplemental/Arial.ttf")
FONT_BOLD = Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf")

BG_TOP = (10, 12, 18)
BG_BOTTOM = (18, 20, 30)
TEXT = (245, 242, 232)
MUTED = (152, 164, 176)
BAR = (45, 208, 184)
BAR_ALT = (86, 176, 255)
COMPARE = (255, 202, 77)
SWAP = (255, 104, 91)
SORTED = (126, 220, 135)
PANEL = (23, 28, 39)
GRID = (40, 48, 62)
AUDIO_RATE = 48_000


@dataclass(frozen=True)
class Step:
    values: tuple[int, ...]
    pair: tuple[int, int] | None
    sorted_from: int
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


@lru_cache(maxsize=None)
def font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    path = FONT_BOLD if bold else FONT_REGULAR
    if path.exists():
        return ImageFont.truetype(str(path), size)
    return ImageFont.load_default(size=size)


def text_width(draw: ImageDraw.ImageDraw, text: str, typeface: ImageFont.ImageFont) -> int:
    left, _, right, _ = draw.textbbox((0, 0), text, font=typeface)
    return right - left


def inversion_count(values: list[int] | tuple[int, ...]) -> int:
    inversions = 0
    for i, left in enumerate(values):
        for right in values[i + 1 :]:
            if left > right:
                inversions += 1
    return inversions


def bubble_sort_steps(values: list[int]) -> list[Step]:
    arr = values[:]
    comparisons = 0
    moves = 0
    swaps = 0
    max_inversions = len(arr) * (len(arr) - 1) // 2
    steps: list[Step] = [
        Step(tuple(arr), None, len(arr), "start", comparisons, moves, swaps, inversion_count(arr), max_inversions),
    ]

    n = len(arr)
    for end in range(n - 1, 0, -1):
        for i in range(end):
            before = tuple(arr)
            comparisons += 1
            should_swap = arr[i] > arr[i + 1]
            if should_swap:
                arr[i], arr[i + 1] = arr[i + 1], arr[i]
                moves += 2
                swaps += 1
            steps.append(
                Step(
                    tuple(arr),
                    (i, i + 1),
                    end + 1,
                    "swap" if should_swap else "compare",
                    comparisons,
                    moves,
                    swaps,
                    inversion_count(arr),
                    max_inversions,
                    swapped=should_swap,
                    before=before,
                )
            )
        steps.append(Step(tuple(arr), None, end, "lock", comparisons, moves, swaps, inversion_count(arr), max_inversions))

    steps.append(Step(tuple(arr), None, 0, "sorted", comparisons, moves, swaps, inversion_count(arr), max_inversions))
    assert arr == sorted(values)
    return steps


def seeded_values(count: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    values = list(range(1, count + 1))
    rng.shuffle(values)
    return values


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def ease(t: float) -> float:
    return 0.5 - math.cos(t * math.pi) * 0.5


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
    w = text_width(draw, content, typeface)
    draw.text(((canvas_width - w) / 2, y), content, font=typeface, fill=fill)


def fit_font(
    draw: ImageDraw.ImageDraw,
    content: str,
    start_size: int,
    max_width: int,
    *,
    bold: bool = False,
    min_size: int = 24,
) -> ImageFont.ImageFont:
    size = start_size
    while size > min_size:
        candidate = font(size, bold=bold)
        if text_width(draw, content, candidate) <= max_width:
            return candidate
        size -= 2
    return font(min_size, bold=bold)


def bar_positions(
    values: tuple[int, ...],
    width: int,
    height: int,
    pair: tuple[int, int] | None,
    before: tuple[int, ...] | None,
    swap_progress: float | None,
) -> list[tuple[float, int]]:
    chart_left = int(width * 0.09)
    chart_right = int(width * 0.91)
    chart_width = chart_right - chart_left
    step = chart_width / len(values)
    positions = [(chart_left + step * i + step * 0.12, value) for i, value in enumerate(values)]

    if pair is None or before is None or swap_progress is None:
        return positions

    i, j = pair
    eased = ease(swap_progress)
    base_i = chart_left + step * i + step * 0.12
    base_j = chart_left + step * j + step * 0.12
    animated = [(chart_left + step * idx + step * 0.12, value) for idx, value in enumerate(before)]
    animated[i] = (lerp(base_i, base_j, eased), before[i])
    animated[j] = (lerp(base_j, base_i, eased), before[j])
    return animated


def draw_frame(
    *,
    width: int,
    height: int,
    step: Step,
    frame_number: int,
    total_frames: int,
    swap_progress: float | None = None,
) -> Image.Image:
    image = gradient_background(width, height).copy()
    draw = ImageDraw.Draw(image)

    title_font = font(int(width * 0.085), bold=True)
    subtitle_font = font(int(width * 0.038))
    label_font = font(int(width * 0.042), bold=True)
    small_font = font(int(width * 0.029))
    tiny_font = font(int(width * 0.024), bold=True)

    margin = int(width * 0.065)
    draw_centered_text(draw, int(height * 0.055), "BUBBLE SORT", title_font, TEXT, width)
    draw_centered_text(
        draw,
        int(height * 0.115),
        "Compare neighbors. Swap if left is larger.",
        subtitle_font,
        MUTED,
        width,
    )

    badge_y = int(height * 0.17)
    badge_h = int(height * 0.052)
    rounded_rect(draw, (margin, badge_y, width - margin, badge_y + badge_h), 18, PANEL, GRID, 2)

    label = {
        "start": "Unsorted input",
        "compare": "Compare neighbors",
        "swap": "Swap smaller left",
        "lock": "Largest value locked",
        "sorted": "Sorted",
    }[step.label]
    label_color = SWAP if step.label == "swap" else COMPARE if step.label == "compare" else SORTED if step.label == "sorted" else TEXT
    pass_count = sum(1 for idx, value in enumerate(step.values) if idx >= step.sorted_from)
    pass_text = f"sorted bars: {pass_count}/{len(step.values)}"
    pass_x = width - margin - text_width(draw, pass_text, small_font) - 24
    label_max_width = int(pass_x - (margin + 24) - 28)
    fitted_label_font = fit_font(
        draw,
        label,
        int(width * 0.042),
        label_max_width,
        bold=True,
        min_size=int(width * 0.026),
    )
    draw.text((margin + 24, badge_y + int(badge_h * 0.22)), label, font=fitted_label_font, fill=label_color)
    draw.text((pass_x, badge_y + int(badge_h * 0.31)), pass_text, font=small_font, fill=MUTED)

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
        draw.line((chart_left, y, chart_right, y), fill=GRID, width=2)

    tau_fill = 1.0
    if step.max_inversions > 0:
        tau_fill = 1.0 - (step.inversions / step.max_inversions)
    tau_fill = max(0.0, min(1.0, tau_fill))

    positions = bar_positions(step.values, width, height, step.pair, step.before, swap_progress)
    active = set(step.pair or ())

    for idx, (x, value) in sorted(enumerate(positions), key=lambda item: item[1][0]):
        bar_h = int((value / max_value) * chart_height)
        y0 = chart_bottom - bar_h
        y1 = chart_bottom

        if idx >= step.sorted_from or step.label == "sorted":
            color = SORTED
        elif idx in active and step.label == "swap":
            color = SWAP
        elif idx in active:
            color = COMPARE
        else:
            color = BAR if idx % 2 == 0 else BAR_ALT

        rounded_rect(draw, (x, y0, x + bar_width, y1), 8, color)

        if idx in active or idx >= step.sorted_from or step.label == "sorted":
            value_text = str(value)
            draw.text(
                (x + (bar_width - text_width(draw, value_text, tiny_font)) / 2, y0 - int(height * 0.026)),
                value_text,
                font=tiny_font,
                fill=TEXT,
            )

    axis_y = chart_bottom + int(height * 0.018)
    draw.line((chart_left, axis_y, chart_right, axis_y), fill=GRID, width=3)

    compare_y = int(height * 0.787)
    compare_h = int(height * 0.052)
    compare_box_w = int(width * 0.29)
    compare_gap = int(width * 0.055)
    operator_w = int(width * 0.1)
    compare_total_w = compare_box_w * 2 + compare_gap * 2 + operator_w
    compare_left_x = int((width - compare_total_w) / 2)
    compare_operator_x = compare_left_x + compare_box_w + compare_gap
    compare_right_x = compare_operator_x + operator_w + compare_gap
    compare_value_font = font(int(width * 0.052), bold=True)
    operator_font = font(int(width * 0.062), bold=True)

    if step.pair is not None and step.before is not None:
        left_idx, right_idx = step.pair
        left_value = step.before[left_idx]
        right_value = step.before[right_idx]
        compare_color = SWAP if step.swapped else SORTED
        operator_text = ">"
    else:
        left_value = None
        right_value = None
        compare_color = SORTED if step.label in {"lock", "sorted"} else MUTED
        operator_text = ">"

    for box_x, value in ((compare_left_x, left_value), (compare_right_x, right_value)):
        rounded_rect(draw, (box_x, compare_y, box_x + compare_box_w, compare_y + compare_h), 16, PANEL, compare_color, 3)
        if value is not None:
            value_text = str(value)
            draw.text(
                (
                    box_x + (compare_box_w - text_width(draw, value_text, compare_value_font)) / 2,
                    compare_y + int(compare_h * 0.13),
                ),
                value_text,
                font=compare_value_font,
                fill=compare_color,
            )

    operator_color = compare_color if left_value is not None else MUTED
    draw.text(
        (
            compare_operator_x + (operator_w - text_width(draw, operator_text, operator_font)) / 2,
            compare_y + int(compare_h * 0.05),
        ),
        operator_text,
        font=operator_font,
        fill=operator_color,
    )

    stats_y = int(height * 0.854)
    stats_h = int(height * 0.052)
    stats_gap = int(width * 0.025)
    stats_w = int((width - margin * 2 - stats_gap) / 2)
    stats = (
        ("comparisons", step.comparisons, COMPARE),
        ("moves", step.moves, SWAP if step.label == "swap" else BAR),
    )
    for stat_index, (name, value, color) in enumerate(stats):
        x0 = margin + stat_index * (stats_w + stats_gap)
        x1 = x0 + stats_w
        rounded_rect(draw, (x0, stats_y, x1, stats_y + stats_h), 16, PANEL, GRID, 2)
        value_text = str(value)
        draw.text((x0 + 22, stats_y + int(stats_h * 0.14)), value_text, font=label_font, fill=color)
        draw.text(
            (x0 + 22, stats_y + int(stats_h * 0.58)),
            name,
            font=small_font,
            fill=MUTED,
        )

    tau_label = "Kendall tau sortedness"
    tau_score = f"{tau_fill:.2f}"
    tau_label_y = int(height * 0.922)
    tau_bar_y = int(height * 0.95)
    tau_bar_h = int(height * 0.012)
    tau_bar_x0 = margin
    tau_bar_x1 = width - margin
    draw.text((tau_bar_x0, tau_label_y), tau_label, font=small_font, fill=MUTED)
    draw.text(
        (tau_bar_x1 - text_width(draw, tau_score, small_font), tau_label_y),
        tau_score,
        font=small_font,
        fill=SORTED,
    )
    rounded_rect(draw, (tau_bar_x0, tau_bar_y, tau_bar_x1, tau_bar_y + tau_bar_h), 12, (15, 18, 25), GRID, 2)
    tau_inner_x0 = tau_bar_x0 + 4
    tau_inner_x1 = tau_bar_x1 - 4
    tau_inner_y0 = tau_bar_y + 4
    tau_inner_y1 = tau_bar_y + tau_bar_h - 4
    if tau_fill > 0 and tau_inner_y1 > tau_inner_y0:
        rounded_rect(
            draw,
            (tau_inner_x0, tau_inner_y0, lerp(tau_inner_x0, tau_inner_x1, tau_fill), tau_inner_y1),
            8,
            SORTED,
        )

    return image


def planned_frames(steps: list[Step], fps: int) -> list[FrameState]:
    timeline: list[FrameState] = []

    intro_frames = int(fps * 1.0)
    for _ in range(intro_frames):
        timeline.append(FrameState(steps[0]))

    for step in steps[1:]:
        if step.swapped:
            for frame in range(5):
                timeline.append(FrameState(step, frame / 4, "swap" if frame == 0 else None))
            timeline.append(FrameState(step))
        elif step.label == "compare":
            timeline.append(FrameState(step))
        elif step.label == "lock":
            for frame in range(6):
                timeline.append(FrameState(step, audio_event="lock" if frame == 0 else None))
        elif step.label == "sorted":
            for frame in range(int(fps * 4.0)):
                timeline.append(FrameState(step, audio_event="sorted" if frame == 0 else None))
        else:
            timeline.append(FrameState(step))

    return timeline


def add_tone(
    samples: list[float],
    start_time: float,
    frequency: float,
    duration: float,
    amplitude: float,
    *,
    attack: float = 0.018,
    release: float = 0.09,
) -> None:
    start = int(start_time * AUDIO_RATE)
    count = int(duration * AUDIO_RATE)
    for index in range(count):
        sample_index = start + index
        if sample_index >= len(samples):
            break
        t = index / AUDIO_RATE
        if t < attack:
            envelope = t / attack
        elif t > duration - release:
            envelope = max(0.0, (duration - t) / release)
        else:
            envelope = 1.0
        samples[sample_index] += math.sin(2 * math.pi * frequency * t) * amplitude * envelope


def add_chord(
    samples: list[float],
    start_time: float,
    notes: tuple[float, ...],
    duration: float,
    amplitude: float,
    *,
    attack: float = 0.018,
    release: float = 0.09,
) -> None:
    per_note_amplitude = amplitude / max(1, len(notes))
    for frequency in notes:
        add_tone(
            samples,
            start_time,
            frequency,
            duration,
            per_note_amplitude,
            attack=attack,
            release=release,
        )


def generate_audio_track(timeline: list[FrameState], fps: int, output: Path) -> None:
    duration = len(timeline) / fps
    samples = [0.0] * int((duration + 0.1) * AUDIO_RATE)

    for frame_number, frame_state in enumerate(timeline):
        if frame_state.audio_event is None:
            continue
        start_time = frame_number / fps
        if frame_state.audio_event == "swap":
            add_chord(samples, start_time, (261.63, 329.63), 0.12, 0.24)  # C/E major third
        elif frame_state.audio_event == "lock":
            add_chord(samples, start_time, (261.63, 523.25), 0.2, 0.24)  # C octave
        elif frame_state.audio_event == "sorted":
            add_chord(
                samples,
                start_time,
                (174.61, 261.63, 349.23, 523.25, 698.46),
                1.9,
                0.5,
                attack=0.04,
                release=0.75,
            )  # F/C resolution stack, no third

    peak = max(0.01, max(abs(sample) for sample in samples))
    scale = min(1.0, 0.92 / peak)

    output.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(AUDIO_RATE)
        frames = bytearray()
        for sample in samples:
            value = int(max(-1.0, min(1.0, sample * scale)) * 32767)
            frames += value.to_bytes(2, "little", signed=True)
        wav.writeframes(frames)


def render_video(args: argparse.Namespace) -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required but was not found on PATH")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    values = seeded_values(args.bars, args.seed)
    steps = bubble_sort_steps(values)
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
                    swap_progress=frame_state.swap_progress,
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
            ).save(args.thumbnail)

        if audio_enabled:
            audio_output = Path(temp_dir) / "operations.wav"
            generate_audio_track(timeline, args.fps, audio_output)
            mux_command = [
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
            ]
            subprocess.run(mux_command, check=True)

    duration = len(timeline) / args.fps
    print(f"Rendered {args.output}")
    print(f"Rendered {args.thumbnail}")
    print(f"Frames: {len(timeline)} at {args.fps} fps ({duration:.1f}s)")
    print(f"Audio: {'on' if audio_enabled else 'off'}")
    print(f"Initial values: {values}")
    print(f"Comparisons: {steps[-1].comparisons}")
    print(f"Moves: {steps[-1].moves}")
    print(f"Swaps: {steps[-1].swaps}")
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
