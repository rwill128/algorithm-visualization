#!/usr/bin/env python3
"""Render Shorts-ready coding interview pattern walkthroughs."""

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


ROOT = Path(__file__).resolve().parents[1]
SHORTS_DIR = ROOT / "artifacts" / "shorts"
DEFAULT_OUTPUT = SHORTS_DIR / "two_sum_hash_map_pattern.mp4"
DEFAULT_THUMBNAIL = SHORTS_DIR / "two_sum_hash_map_pattern_thumbnail.png"

BLUE = (86, 176, 255)
GREEN = (126, 220, 135)
YELLOW = (255, 202, 77)
RED = (255, 104, 91)
PURPLE = (176, 134, 255)
DARK = (13, 17, 26)
PANEL_DARK = (18, 23, 34)
CARD = (27, 33, 46)
LINE = (47, 56, 74)


NUMS = (3, 8, 4, 14, 6, 1, 9)
TARGET = 13
ANSWER_PAIR = (2, 6)


@dataclass(frozen=True)
class FrameState:
    mode: str
    step: int = 0
    substep: int = 0
    progress: float = 0.0
    audio_event: str | None = None


@dataclass(frozen=True)
class HashStep:
    index: int
    value: int
    complement: int
    seen_before: tuple[tuple[int, int], ...]
    found_index: int | None


def two_sum_hash_steps() -> tuple[HashStep, ...]:
    seen: dict[int, int] = {}
    steps: list[HashStep] = []
    for index, value in enumerate(NUMS):
        complement = TARGET - value
        found_index = seen.get(complement)
        steps.append(HashStep(index, value, complement, tuple(sorted(seen.items())), found_index))
        if found_index is not None:
            break
        seen[value] = index
    return tuple(steps)


def timeline(hash_steps: tuple[HashStep, ...], fps: int) -> list[FrameState]:
    frames: list[FrameState] = []

    def hold(mode: str, seconds: float, *, step: int = 0, substep: int = 0, event: str | None = None) -> None:
        count = max(1, int(seconds * fps))
        for frame in range(count):
            frames.append(FrameState(mode, step, substep, frame / max(1, count - 1), event if frame == 0 else None))

    hold("intro", 2.4, event="lock")
    hold("problem", 3.0)

    naive_pairs = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 6))
    for pair_index, _pair in enumerate(naive_pairs):
        hold("naive", 0.78, step=pair_index, event="swap")
    hold("naive_summary", 2.2, step=len(naive_pairs) - 1)

    hold("insight", 3.6, event="lock")
    hold("code", 3.2)

    for step_index, _hash_step in enumerate(hash_steps):
        hold("hash_read", 1.25, step=step_index, event="swap")
        hold("hash_check", 1.25, step=step_index, event="swap")
        if step_index == len(hash_steps) - 1:
            hold("hash_found", 2.2, step=step_index, event="lock")
        else:
            hold("hash_store", 1.15, step=step_index, event="lock")

    hold("outro", 4.0, step=len(hash_steps) - 1, event="sorted")
    return frames


def draw_wrapped_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    xy: tuple[int, int],
    typeface,
    fill: tuple[int, int, int],
    max_width: int,
    line_gap: int = 8,
) -> int:
    x, y = xy
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if current and bubble.text_width(draw, candidate, typeface) > max_width:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    line_height = typeface.size + line_gap
    for offset, line in enumerate(lines):
        draw.text((x, y + offset * line_height), line, font=typeface, fill=fill)
    return y + len(lines) * line_height


def draw_header(draw: ImageDraw.ImageDraw, width: int, title: str, subtitle: str) -> None:
    title_font = bubble.fit_font(draw, title, 66, int(width * 0.9), bold=True, min_size=42)
    bubble.draw_centered_text(draw, 60, title, title_font, bubble.TEXT, width)
    subtitle_font = bubble.font(31)
    bubble.draw_centered_text(draw, 142, subtitle, subtitle_font, bubble.MUTED, width)


def draw_array(
    draw: ImageDraw.ImageDraw,
    *,
    width: int,
    y: int,
    highlight: tuple[int, ...] = (),
    active: tuple[int, ...] = (),
    found: tuple[int, ...] = (),
) -> dict[int, tuple[int, int, int, int]]:
    margin = 74
    gap = 12
    box_w = int((width - 2 * margin - gap * (len(NUMS) - 1)) / len(NUMS))
    box_h = 98
    label_font = bubble.font(42, bold=True)
    index_font = bubble.font(23)
    boxes: dict[int, tuple[int, int, int, int]] = {}
    for index, value in enumerate(NUMS):
        x0 = margin + index * (box_w + gap)
        y0 = y
        x1 = x0 + box_w
        y1 = y + box_h
        fill = CARD
        outline = LINE
        if index in highlight:
            fill = (54, 47, 29)
            outline = YELLOW
        if index in active:
            fill = (37, 48, 62)
            outline = BLUE
        if index in found:
            fill = (32, 58, 40)
            outline = GREEN
        bubble.rounded_rect(draw, (x0, y0, x1, y1), 12, fill, outline, 3)
        text = str(value)
        draw.text((x0 + (box_w - bubble.text_width(draw, text, label_font)) / 2, y0 + 22), text, font=label_font, fill=bubble.TEXT)
        idx = f"i={index}"
        draw.text((x0 + (box_w - bubble.text_width(draw, idx, index_font)) / 2, y1 + 10), idx, font=index_font, fill=bubble.MUTED)
        boxes[index] = (x0, y0, x1, y1)
    return boxes


def draw_target(draw: ImageDraw.ImageDraw, width: int, y: int) -> None:
    text = f"target = {TARGET}"
    typeface = bubble.font(40, bold=True)
    pill_w = bubble.text_width(draw, text, typeface) + 56
    x0 = int((width - pill_w) / 2)
    bubble.rounded_rect(draw, (x0, y, x0 + pill_w, y + 70), 18, (40, 33, 23), YELLOW, 2)
    draw.text((x0 + 28, y + 13), text, font=typeface, fill=YELLOW)


def draw_naive_panel(draw: ImageDraw.ImageDraw, width: int, y: int, pair_index: int) -> None:
    pairs = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 6))
    i, j = pairs[min(pair_index, len(pairs) - 1)]
    left = NUMS[i]
    right = NUMS[j]
    total = left + right
    good = total == TARGET
    margin = 72
    bubble.rounded_rect(draw, (margin, y, width - margin, y + 220), 18, PANEL_DARK, LINE, 2)
    label_font = bubble.font(29, bold=True)
    math_font = bubble.font(56, bold=True)
    draw.text((margin + 26, y + 24), "naive scan", font=label_font, fill=bubble.MUTED)
    expression = f"{left} + {right} = {total}"
    color = GREEN if good else RED
    draw.text((margin + 26, y + 76), expression, font=math_font, fill=color)
    verdict = "found pair" if good else "not target; keep trying"
    draw.text((margin + 26, y + 150), verdict, font=bubble.font(31), fill=color)
    count = "checks many pairs: O(n²)"
    draw.text((width - margin - bubble.text_width(draw, count, label_font) - 26, y + 30), count, font=label_font, fill=YELLOW)


def draw_hash_map(draw: ImageDraw.ImageDraw, x: int, y: int, w: int, h: int, entries: tuple[tuple[int, int], ...], highlight_value: int | None = None) -> None:
    bubble.rounded_rect(draw, (x, y, x + w, y + h), 18, PANEL_DARK, LINE, 2)
    title_font = bubble.font(30, bold=True)
    draw.text((x + 22, y + 18), "seen[value] = index", font=title_font, fill=PURPLE)
    item_font = bubble.font(31, bold=True)
    if not entries:
        draw.text((x + 22, y + 82), "empty", font=item_font, fill=bubble.MUTED)
        return
    visible = entries[:5]
    row_count = max(1, len(visible))
    item_h = 44
    row_gap = min(56, max(item_h + 6, int((h - 96) / row_count)))
    for row, (value, index) in enumerate(visible):
        yy = y + 76 + row * row_gap
        fill = (42, 32, 54) if value == highlight_value else CARD
        outline = PURPLE if value == highlight_value else LINE
        bubble.rounded_rect(draw, (x + 22, yy, x + w - 22, yy + 44), 12, fill, outline, 2)
        text = f"{value} -> {index}"
        draw.text((x + 42, yy + 6), text, font=item_font, fill=bubble.TEXT)


def draw_code_panel(draw: ImageDraw.ImageDraw, x: int, y: int, w: int, h: int, active_line: int | None = None) -> None:
    bubble.rounded_rect(draw, (x, y, x + w, y + h), 18, (14, 18, 28), LINE, 2)
    lines = [
        "seen = {}",
        "for i, x in enumerate(nums):",
        "    need = target - x",
        "    if need in seen:",
        "        return [seen[need], i]",
        "    seen[x] = i",
    ]
    code_font = bubble.font(24, bold=True)
    title_font = bubble.font(28, bold=True)
    draw.text((x + 22, y + 18), "pseudocode", font=title_font, fill=BLUE)
    for idx, line in enumerate(lines):
        yy = y + 66 + idx * 42
        if active_line == idx:
            bubble.rounded_rect(draw, (x + 14, yy - 6, x + w - 14, yy + 34), 10, (45, 43, 24), YELLOW, 2)
        color = GREEN if idx == 4 and active_line == idx else bubble.TEXT
        draw.text((x + 24, yy), line, font=code_font, fill=color)


def draw_hash_step(draw: ImageDraw.ImageDraw, width: int, y: int, state: FrameState, hash_steps: tuple[HashStep, ...]) -> None:
    step = hash_steps[min(state.step, len(hash_steps) - 1)]
    active = (step.index,)
    found = (step.index, step.found_index) if step.found_index is not None else ()
    draw_array(draw, width=width, y=y, active=active, found=found)

    margin = 72
    panel_y = y + 170
    available_w = width - 2 * margin - 24
    left_w = int(available_w * 0.58)
    right_w = width - 2 * margin - left_w - 24
    bubble.rounded_rect(draw, (margin, panel_y, margin + left_w, panel_y + 320), 18, PANEL_DARK, LINE, 2)
    title_font = bubble.font(31, bold=True)
    big_font = bubble.font(50, bold=True)
    small_font = bubble.font(28)
    draw.text((margin + 24, panel_y + 22), f"i = {step.index}, x = {step.value}", font=title_font, fill=BLUE)
    draw.text((margin + 24, panel_y + 82), f"need = {TARGET} - {step.value} = {step.complement}", font=big_font, fill=YELLOW)
    if state.mode == "hash_read":
        msg = "read the next number"
        color = BLUE
    elif state.mode == "hash_check":
        msg = f"ask: have we seen {step.complement}?"
        color = YELLOW
    elif state.mode == "hash_store":
        msg = f"not yet, so store {step.value} -> {step.index}"
        color = PURPLE
    else:
        msg = f"yes: {step.complement} was at index {step.found_index}"
        color = GREEN
    draw.text((margin + 24, panel_y + 166), msg, font=small_font, fill=color)
    if step.found_index is not None:
        answer = f"return [{step.found_index}, {step.index}]"
        draw.text((margin + 24, panel_y + 226), answer, font=big_font, fill=GREEN)

    highlight = step.complement if state.mode in {"hash_check", "hash_found"} else None
    entries = step.seen_before
    if state.mode == "hash_store":
        entries = tuple(sorted((*entries, (step.value, step.index))))
    draw_hash_map(draw, margin + left_w + 24, panel_y, right_w, 320, entries, highlight)

    active_line = {
        "hash_read": 1,
        "hash_check": 3,
        "hash_store": 5,
        "hash_found": 4,
    }.get(state.mode)
    draw_code_panel(draw, margin, panel_y + 356, width - 2 * margin, 340, active_line)


def draw_frame(
    width: int,
    height: int,
    state: FrameState,
    frame_number: int,
    total_frames: int,
    hash_steps: tuple[HashStep, ...],
) -> Image.Image:
    image = bubble.gradient_background(width, height).copy()
    draw = ImageDraw.Draw(image)
    margin = 72

    draw_header(draw, width, "TWO SUM: THE HASH MAP TRICK", "Pattern behind the LeetCode classic")
    draw_target(draw, width, 204)

    if state.mode == "intro":
        draw_array(draw, width=width, y=340)
        bubble.rounded_rect(draw, (margin, 620, width - margin, 1060), 22, PANEL_DARK, LINE, 2)
        headline = "Find two values that add to the target."
        draw_wrapped_text(draw, headline, (margin + 34, 660), bubble.font(56, bold=True), bubble.TEXT, width - 2 * margin - 68)
        draw_wrapped_text(
            draw,
            "The trick is to stop searching for a pair and start searching for the missing complement.",
            (margin + 34, 820),
            bubble.font(34),
            bubble.MUTED,
            width - 2 * margin - 68,
        )
    elif state.mode == "problem":
        draw_array(draw, width=width, y=340)
        bubble.rounded_rect(draw, (margin, 610, width - margin, 1040), 22, PANEL_DARK, LINE, 2)
        draw_wrapped_text(draw, "Input: nums and target", (margin + 34, 650), bubble.font(48, bold=True), bubble.TEXT, width - 2 * margin - 68)
        draw_wrapped_text(
            draw,
            "Output the two indices. Here the answer is nums[2] + nums[6] = 4 + 9 = 13.",
            (margin + 34, 760),
            bubble.font(35),
            bubble.MUTED,
            width - 2 * margin - 68,
        )
        draw_code_panel(draw, margin, 1125, width - 2 * margin, 340)
    elif state.mode in {"naive", "naive_summary"}:
        pairs = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 6))
        pair = pairs[min(state.step, len(pairs) - 1)]
        found = pair if NUMS[pair[0]] + NUMS[pair[1]] == TARGET else ()
        draw_array(draw, width=width, y=340, highlight=pair, found=found)
        draw_naive_panel(draw, width, 610, state.step)
        if state.mode == "naive_summary":
            bubble.rounded_rect(draw, (margin, 900, width - margin, 1205), 22, (34, 23, 27), RED, 2)
            draw_wrapped_text(draw, "Works, but it wastes comparisons.", (margin + 34, 940), bubble.font(48, bold=True), RED, width - 2 * margin - 68)
            draw_wrapped_text(
                draw,
                "Every value asks every later value: O(n²). The optimized version asks a hash map one question per value.",
                (margin + 34, 1040),
                bubble.font(34),
                bubble.TEXT,
                width - 2 * margin - 68,
            )
    elif state.mode == "insight":
        draw_array(draw, width=width, y=340, active=(2,))
        bubble.rounded_rect(draw, (margin, 610, width - margin, 1145), 22, PANEL_DARK, YELLOW, 2)
        draw_wrapped_text(draw, "Key question:", (margin + 34, 650), bubble.font(40, bold=True), YELLOW, width - 2 * margin - 68)
        draw_wrapped_text(draw, "If x is 4, what number would finish the pair?", (margin + 34, 720), bubble.font(52, bold=True), bubble.TEXT, width - 2 * margin - 68)
        draw_wrapped_text(draw, "need = target - x = 13 - 4 = 9", (margin + 34, 900), bubble.font(46, bold=True), GREEN, width - 2 * margin - 68)
        draw_wrapped_text(draw, "So we store numbers we have already passed, then check whether the needed number is already there.", (margin + 34, 1005), bubble.font(31), bubble.MUTED, width - 2 * margin - 68)
    elif state.mode == "code":
        draw_array(draw, width=width, y=310)
        draw_code_panel(draw, margin, 540, width - 2 * margin, 360)
        bubble.rounded_rect(draw, (margin, 980, width - margin, 1275), 22, PANEL_DARK, GREEN, 2)
        draw_wrapped_text(draw, "One pass. One lookup per value.", (margin + 34, 1020), bubble.font(50, bold=True), GREEN, width - 2 * margin - 68)
        draw_wrapped_text(draw, "The hash map turns pair search into complement lookup.", (margin + 34, 1130), bubble.font(34), bubble.MUTED, width - 2 * margin - 68)
    elif state.mode in {"hash_read", "hash_check", "hash_store", "hash_found"}:
        draw_hash_step(draw, width, 315, state, hash_steps)
    elif state.mode == "outro":
        final = hash_steps[-1]
        draw_array(draw, width=width, y=330, found=ANSWER_PAIR)
        bubble.rounded_rect(draw, (margin, 610, width - margin, 1110), 22, (21, 38, 28), GREEN, 3)
        draw_wrapped_text(draw, "Result: indices [2, 6]", (margin + 34, 655), bubble.font(58, bold=True), GREEN, width - 2 * margin - 68)
        draw_wrapped_text(draw, f"nums[2] + nums[6] = {NUMS[2]} + {NUMS[6]} = {TARGET}", (margin + 34, 780), bubble.font(45, bold=True), bubble.TEXT, width - 2 * margin - 68)
        draw_wrapped_text(draw, "Time: O(n)    Space: O(n)", (margin + 34, 900), bubble.font(48, bold=True), YELLOW, width - 2 * margin - 68)
        draw_wrapped_text(draw, "Pattern: when a pair must sum to a target, store what you have seen and search for what is missing.", (margin + 34, 1010), bubble.font(31), bubble.MUTED, width - 2 * margin - 68)
        draw_hash_map(draw, margin, 1200, width - 2 * margin, 380, final.seen_before, final.complement)

    footer_font = bubble.font(23)
    progress = f"{frame_number + 1}/{total_frames}"
    draw.text((margin, height - 72), "LeetCode pattern: complement lookup with a hash map", font=footer_font, fill=bubble.MUTED)
    draw.text((width - margin - bubble.text_width(draw, progress, footer_font), height - 72), progress, font=footer_font, fill=bubble.MUTED)
    return image


def render_video(args: argparse.Namespace) -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required but was not found on PATH")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    hash_steps = two_sum_hash_steps()
    frames = timeline(hash_steps, args.fps)
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
                image = draw_frame(args.width, args.height, frame_state, frame_number, len(frames), hash_steps)
                if not thumbnail_saved and frame_state.mode == "hash_check" and frame_state.step == 2:
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
            draw_frame(args.width, args.height, frames[-1], len(frames) - 1, len(frames), hash_steps).save(args.thumbnail)

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--thumbnail", type=Path, default=DEFAULT_THUMBNAIL)
    parser.add_argument("--width", type=int, default=1080)
    parser.add_argument("--height", type=int, default=1920)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--no-audio", action="store_true")
    args = parser.parse_args()
    if args.width <= 0 or args.height <= 0 or args.fps <= 0:
        parser.error("--width, --height, and --fps must be positive")
    return args


if __name__ == "__main__":
    render_video(parse_args())
