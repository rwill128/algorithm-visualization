#!/usr/bin/env python3
"""Render a Shorts-ready explainer for next-token logit attribution."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import dataclass
import math
from pathlib import Path
import shutil
import subprocess
import tempfile

from PIL import Image, ImageDraw

import render_bubble_sort_short as bubble
import render_llm_next_token_short as llm


ROOT = Path(__file__).resolve().parents[1]
SHORTS_DIR = ROOT / "artifacts" / "shorts"
DEFAULT_OUTPUT = SHORTS_DIR / "llm_logit_attribution_explained.mp4"
DEFAULT_THUMBNAIL = SHORTS_DIR / "llm_logit_attribution_explained_thumbnail.png"

TOKEN_RED = (255, 104, 91)
TOKEN_GREEN = (126, 220, 135)
TOKEN_YELLOW = (255, 202, 77)
TOKEN_BLUE = (86, 176, 255)
BAR_BG = (15, 18, 25)


@dataclass(frozen=True)
class AttributionExample:
    context_tokens: tuple[llm.ContextToken, ...]
    selected_text: str
    selected_probability: float
    selected_logit: float
    top_token: str
    top_score: float
    ranked_tokens: tuple[tuple[str, float], ...]


@dataclass(frozen=True)
class ExplainerFrame:
    scene: str
    progress: float
    audio_event: str | None = None


def blend(color_a: tuple[int, int, int], color_b: tuple[int, int, int], amount: float) -> tuple[int, int, int]:
    amount = max(0.0, min(1.0, amount))
    return tuple(int(bubble.lerp(color_a[channel], color_b[channel], amount)) for channel in range(3))


def attribution_color(score: float) -> tuple[int, int, int]:
    return blend(TOKEN_RED, TOKEN_GREEN, score)


def build_example(args: argparse.Namespace) -> AttributionExample:
    torch, tokenizer, model = llm.load_model(args.model)
    torch.manual_seed(args.seed)
    generated = tokenizer(args.prompt, return_tensors="pt").input_ids

    first_selected = None
    selected_id = None
    selected_probability = 0.0
    selected_logit_value = 0.0
    positive_scores: list[float] = []

    for step_index in range(2):
        with torch.no_grad():
            outputs = model(generated)
            logits = outputs.logits[0, -1] / args.temperature
            probabilities = torch.softmax(logits, dim=-1)
            sample_values, sample_indices = torch.topk(probabilities, min(args.sample_top_k, probabilities.shape[-1]))
            sample_values = sample_values / sample_values.sum()
            selected_position = torch.multinomial(sample_values, 1).item()
            sampled_id = int(sample_indices[selected_position].item())

        if step_index == 0:
            first_selected = sampled_id
            generated = torch.cat([generated, torch.tensor([[sampled_id]], dtype=generated.dtype)], dim=1)
            continue

        selected_id = sampled_id
        selected_probability = float(probabilities[selected_id].item())
        selected_logit_value = float(logits[selected_id].item())

        model.zero_grad(set_to_none=True)
        embeddings = model.get_input_embeddings()(generated).detach()
        embeddings.requires_grad_(True)
        attribution_outputs = model(inputs_embeds=embeddings)
        selected_logit = attribution_outputs.logits[0, -1, selected_id] / args.temperature
        selected_logit.backward()
        token_attribution = (embeddings.grad[0] * embeddings.detach()[0]).sum(dim=-1)
        positive_scores = torch.clamp(token_attribution, min=0.0).tolist()

    if first_selected is None or selected_id is None:
        raise RuntimeError("failed to build two-token attribution example")

    max_score = max(positive_scores, default=0.0)
    context_tokens = []
    ranked = []
    for token_id, score in zip(generated[0].tolist(), positive_scores):
        display = llm.display_token(tokenizer.decode([token_id]))
        normalized = 0.0 if max_score <= 0 else score / max_score
        context_tokens.append(llm.ContextToken(display, normalized))
        ranked.append((display, normalized))

    ranked.sort(key=lambda item: item[1], reverse=True)
    top_token, top_score = ranked[0]
    return AttributionExample(
        context_tokens=tuple(context_tokens),
        selected_text=llm.display_token(tokenizer.decode([selected_id])),
        selected_probability=selected_probability,
        selected_logit=selected_logit_value,
        top_token=top_token,
        top_score=top_score,
        ranked_tokens=tuple(ranked[:5]),
    )


def build_timeline(fps: int) -> list[ExplainerFrame]:
    timeline: list[ExplainerFrame] = []
    scene_durations = (
        ("setup", 4.5, "swap"),
        ("logit", 5.5, "lock"),
        ("gradient", 6.0, "swap"),
        ("multiply", 6.5, "lock"),
        ("sum", 5.5, "swap"),
        ("color", 7.5, "sorted"),
    )
    for scene, duration, event in scene_durations:
        frames = int(fps * duration)
        for frame in range(frames):
            timeline.append(ExplainerFrame(scene, frame / max(1, frames - 1), event if frame == 0 else None))
    return timeline


def text(draw: ImageDraw.ImageDraw, xy: tuple[float, float], content: str, size: int, fill, *, bold: bool = False) -> None:
    draw.text(xy, content, font=bubble.font(size, bold=bold), fill=fill)


def draw_chip(
    draw: ImageDraw.ImageDraw,
    x: float,
    y: float,
    label: str,
    color: tuple[int, int, int],
    width: int,
    *,
    outline: tuple[int, int, int] | None = None,
) -> tuple[float, float]:
    chip_font = bubble.font(int(width * 0.020), bold=True)
    max_text_width = int(width * 0.16)
    chip_label = label
    if bubble.text_width(draw, chip_label, chip_font) > max_text_width:
        while chip_label and bubble.text_width(draw, chip_label + "...", chip_font) > max_text_width:
            chip_label = chip_label[:-1]
        chip_label += "..."
    chip_w = max(int(width * 0.040), bubble.text_width(draw, chip_label, chip_font) + int(width * 0.025))
    chip_h = int(width * 0.039)
    bubble.rounded_rect(draw, (x, y, x + chip_w, y + chip_h), 10, color, outline)
    draw.text((x + int(width * 0.012), y + int(width * 0.008)), chip_label, font=chip_font, fill=(18, 22, 30))
    return chip_w, chip_h


def draw_context(draw: ImageDraw.ImageDraw, width: int, top: int, example: AttributionExample, reveal_color: float) -> None:
    margin = int(width * 0.065)
    x = margin + 24
    y = top
    max_x = width - margin - 24
    gap = int(width * 0.007)
    row_gap = int(width * 0.011)
    for token in example.context_tokens:
        color = attribution_color(token.importance * reveal_color)
        chip_w, chip_h = draw_chip(draw, x, y, token.text, color, width)
        if x + chip_w > max_x:
            x = margin + 24
            y += chip_h + row_gap
            chip_w, chip_h = draw_chip(draw, x, y, token.text, color, width)
        x += chip_w + gap
    token_color = blend(bubble.PANEL, TOKEN_YELLOW, reveal_color)
    if x > width - margin - int(width * 0.15):
        x = margin + 24
        y += int(width * 0.050)
    draw_chip(draw, x, y, example.selected_text, token_color, width, outline=TOKEN_YELLOW)


def draw_header(draw: ImageDraw.ImageDraw, width: int, height: int) -> None:
    title = "LOGIT ATTRIBUTION"
    title_font = bubble.fit_font(draw, title, int(width * 0.068), int(width * 0.9), bold=True, min_size=int(width * 0.046))
    subtitle_font = bubble.font(int(width * 0.030))
    draw.text(((width - bubble.text_width(draw, title, title_font)) / 2, int(height * 0.04)), title, font=title_font, fill=bubble.TEXT)
    bubble.draw_centered_text(draw, int(height * 0.092), "Which context tokens pushed this next token up?", subtitle_font, bubble.MUTED, width)


def draw_equation_panel(
    draw: ImageDraw.ImageDraw,
    width: int,
    height: int,
    title: str,
    lines: tuple[str, ...],
    accent: tuple[int, int, int],
) -> tuple[int, int, int, int]:
    margin = int(width * 0.065)
    top = int(height * 0.405)
    bottom = int(height * 0.735)
    bubble.rounded_rect(draw, (margin, top, width - margin, bottom), 18, bubble.PANEL, bubble.GRID, 2)
    text(draw, (margin + 24, top + 24), title, int(width * 0.031), accent, bold=True)
    line_font_size = int(width * 0.035)
    y = top + 84
    for line in lines:
        text(draw, (margin + 28, y), line, line_font_size, bubble.TEXT if line else bubble.MUTED, bold=True)
        y += int(height * 0.043)
    return margin, top, width - margin, bottom


def draw_setup(draw: ImageDraw.ImageDraw, width: int, height: int, example: AttributionExample, progress: float) -> None:
    draw_equation_panel(
        draw,
        width,
        height,
        "The prediction has already picked a target.",
        (
            f'next token = "{example.selected_text}"',
            f"probability = {example.selected_probability * 100:.1f}%",
            "Now ask: what pushed that token score upward?",
        ),
        TOKEN_YELLOW,
    )


def draw_logit(draw: ImageDraw.ImageDraw, width: int, height: int, example: AttributionExample, progress: float) -> None:
    draw_equation_panel(
        draw,
        width,
        height,
        "Start before softmax: the raw logit.",
        (
            "final hidden state",
            "dot output weights",
            f'logit("{example.selected_text}") = raw score',
            "higher logit -> higher probability",
        ),
        TOKEN_BLUE,
    )


def draw_gradient(draw: ImageDraw.ImageDraw, width: int, height: int, example: AttributionExample, progress: float) -> None:
    draw_equation_panel(
        draw,
        width,
        height,
        "Backprop asks a sensitivity question.",
        (
            f'd logit("{example.selected_text}")',
            "--------------------",
            "d previous embedding",
            "",
            "If this token changed, would the logit move?",
        ),
        TOKEN_GREEN,
    )


def draw_multiply(draw: ImageDraw.ImageDraw, width: int, height: int, example: AttributionExample, progress: float) -> None:
    margin, top, right, bottom = draw_equation_panel(
        draw,
        width,
        height,
        "Turn sensitivity into contribution.",
        (
            "gradient x embedding",
            "= per-dimension contribution",
            "sum dimensions -> one token score",
        ),
        TOKEN_YELLOW,
    )
    box_top = top + int(height * 0.195)
    columns = ("embedding", "gradient", "product")
    values = ("0.42", "1.8", "0.76")
    col_w = (right - margin - 72) / 3
    for index, (label, value) in enumerate(zip(columns, values)):
        x0 = margin + 28 + index * col_w
        bubble.rounded_rect(draw, (x0, box_top, x0 + col_w - 16, box_top + int(height * 0.07)), 14, (31, 37, 50), bubble.GRID, 2)
        text(draw, (x0 + 16, box_top + 12), label, int(width * 0.021), bubble.MUTED, bold=True)
        text(draw, (x0 + 16, box_top + 44), value, int(width * 0.032), TOKEN_GREEN if index == 2 else bubble.TEXT, bold=True)


def draw_sum(draw: ImageDraw.ImageDraw, width: int, height: int, example: AttributionExample, progress: float) -> None:
    draw_equation_panel(
        draw,
        width,
        height,
        "Rank tokens by positive logit push.",
        (
            "negative scores -> clipped to red",
            "largest positive score -> green",
            f'top contributor here: "{example.top_token}"',
        ),
        TOKEN_GREEN,
    )
    margin = int(width * 0.065)
    y = int(height * 0.62)
    max_bar_w = int(width * 0.45)
    for label, score in example.ranked_tokens[:3]:
        text(draw, (margin + 30, y), label, int(width * 0.023), bubble.TEXT, bold=True)
        bar_x = margin + int(width * 0.26)
        bubble.rounded_rect(draw, (bar_x, y + 6, bar_x + max_bar_w, y + 28), 8, BAR_BG, bubble.GRID, 1)
        bubble.rounded_rect(draw, (bar_x + 3, y + 9, bar_x + 3 + (max_bar_w - 6) * score, y + 25), 7, attribution_color(score), None)
        y += int(height * 0.035)


def draw_color(draw: ImageDraw.ImageDraw, width: int, height: int, example: AttributionExample, progress: float) -> None:
    draw_equation_panel(
        draw,
        width,
        height,
        "Final visualization.",
        (
            "red = little positive push on selected logit",
            "green = strong positive push",
            "This is not attention. It is output-score pressure.",
        ),
        TOKEN_YELLOW,
    )


def draw_frame(
    *,
    width: int,
    height: int,
    frame: ExplainerFrame,
    frame_number: int,
    total_frames: int,
    example: AttributionExample,
) -> Image.Image:
    image = bubble.gradient_background(width, height).copy()
    draw = ImageDraw.Draw(image)
    draw_header(draw, width, height)

    margin = int(width * 0.065)
    panel_top = int(height * 0.155)
    panel_bottom = int(height * 0.37)
    bubble.rounded_rect(draw, (margin, panel_top, width - margin, panel_bottom), 20, bubble.PANEL, bubble.GRID, 2)
    text(draw, (margin + 24, panel_top + 20), "context colored by logit attribution", int(width * 0.024), bubble.MUTED, bold=True)
    text(draw, (width - margin - int(width * 0.26), panel_top + 20), "red less | green more", int(width * 0.024), bubble.MUTED, bold=True)
    reveal_color = frame.progress if frame.scene == "color" else (0.22 if frame.scene in {"setup", "logit", "gradient"} else 0.65)
    draw_context(draw, width, panel_top + 64, example, reveal_color)

    if frame.scene == "setup":
        draw_setup(draw, width, height, example, frame.progress)
    elif frame.scene == "logit":
        draw_logit(draw, width, height, example, frame.progress)
    elif frame.scene == "gradient":
        draw_gradient(draw, width, height, example, frame.progress)
    elif frame.scene == "multiply":
        draw_multiply(draw, width, height, example, frame.progress)
    elif frame.scene == "sum":
        draw_sum(draw, width, height, example, frame.progress)
    else:
        draw_color(draw, width, height, example, frame.progress)

    progress = (frame_number + 1) / max(1, total_frames)
    progress_top = int(height * 0.93)
    text(draw, (margin, progress_top - 34), "explainer progress", int(width * 0.021), bubble.MUTED)
    text(draw, (width - margin - bubble.text_width(draw, f"{progress:.2f}", bubble.font(int(width * 0.021))), progress_top - 34), f"{progress:.2f}", int(width * 0.021), TOKEN_GREEN)
    bubble.rounded_rect(draw, (margin, progress_top, width - margin, progress_top + int(height * 0.016)), 12, BAR_BG, bubble.GRID, 2)
    bubble.rounded_rect(
        draw,
        (margin + 4, progress_top + 4, bubble.lerp(margin + 4, width - margin - 4, progress), progress_top + int(height * 0.016) - 4),
        8,
        TOKEN_GREEN,
        None,
    )
    footer = "selected logit -> gradient -> gradient x embedding -> token score"
    text(draw, (margin, int(height * 0.958)), footer, int(width * 0.019), bubble.MUTED)
    return image


def render_video(args: argparse.Namespace) -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required but was not found on PATH")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.thumbnail.parent.mkdir(parents=True, exist_ok=True)

    example = build_example(args)
    timeline = build_timeline(args.fps)
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
            for frame_number, state in enumerate(timeline):
                image = draw_frame(
                    width=args.width,
                    height=args.height,
                    frame=state,
                    frame_number=frame_number,
                    total_frames=len(timeline),
                    example=example,
                )
                if not thumbnail_saved and state.scene == "multiply" and state.progress > 0.5:
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
                frame=timeline[-1],
                frame_number=len(timeline) - 1,
                total_frames=len(timeline),
                example=example,
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

    print(f"Rendered {args.output}")
    print(f"Rendered {args.thumbnail}")
    print(f"Frames: {len(timeline)} at {args.fps} fps ({len(timeline) / args.fps:.1f}s)")
    print(f"Audio: {'on' if audio_enabled else 'off'}")
    print(f"Selected token: {example.selected_text}")
    print(f"Selected logit: {example.selected_logit:.3f}")
    print(f"Top contributor: {example.top_token} ({example.top_score:.2f})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--thumbnail", type=Path, default=DEFAULT_THUMBNAIL)
    parser.add_argument("--width", type=int, default=1080)
    parser.add_argument("--height", type=int, default=1920)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--prompt", default=llm.PROMPT)
    parser.add_argument("--sample-top-k", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.85)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--no-audio", action="store_true")
    args = parser.parse_args()
    if args.width <= 0 or args.height <= 0 or args.fps <= 0:
        parser.error("--width, --height, and --fps must be positive")
    if args.sample_top_k <= 0:
        parser.error("--sample-top-k must be positive")
    if args.temperature <= 0:
        parser.error("--temperature must be positive")
    return args


if __name__ == "__main__":
    render_video(parse_args())
