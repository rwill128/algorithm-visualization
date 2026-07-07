#!/usr/bin/env python3
"""Render a Shorts-ready next-token probability visualization for GPT-2."""

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


ROOT = Path(__file__).resolve().parents[1]
SHORTS_DIR = ROOT / "artifacts" / "shorts"
DEFAULT_OUTPUT = SHORTS_DIR / "llm_next_token_probabilities.mp4"
DEFAULT_THUMBNAIL = SHORTS_DIR / "llm_next_token_probabilities_thumbnail.png"
INFLUENCE_LABELS = {
    "ablation": "context influence by ablation",
    "attention": "context attention weight",
    "logit-attribution": "context logit attribution",
}

PROMPT = "In a quiet lab, the tiny robot learned to"

BG_HIGH_UNCERTAINTY = (255, 104, 91)
BG_LOW_UNCERTAINTY = (126, 220, 135)
TOKEN_BLUE = (86, 176, 255)
TOKEN_YELLOW = (255, 202, 77)
TOKEN_GREEN = (126, 220, 135)
TOKEN_RED = (255, 104, 91)
BAR_BG = (15, 18, 25)
READ_SECONDS = 0.48
REVEAL_SECONDS = 1.28
LOCK_SECONDS = 0.52
FINAL_HOLD_SECONDS = 2.6


@dataclass(frozen=True)
class Candidate:
    token_id: int
    text: str
    probability: float
    selected: bool


@dataclass(frozen=True)
class ContextToken:
    text: str
    importance: float


@dataclass(frozen=True)
class TokenStep:
    index: int
    context: str
    context_tokens: tuple[ContextToken, ...]
    selected_text: str
    selected_probability: float
    surprisal_bits: float
    token_perplexity: float
    entropy_bits: float
    effective_choices: float
    candidates: tuple[Candidate, ...]


@dataclass(frozen=True)
class FrameState:
    step: TokenStep
    reveal: float
    phase: str
    audio_event: str | None = None


def load_model(model_name: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="eager")
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    return torch, tokenizer, model


def display_token(text: str) -> str:
    if text == "":
        return "<empty>"
    if text.startswith(" "):
        text = text.lstrip(" ")
    if text == "":
        return "<space>"
    if text and set(text) == {"\n"}:
        return "line break"
    if text and set(text) == {"\t"}:
        return "tab"
    text = text.replace("\n", " line break ").replace("\t", " tab ")
    return text


def js_divergence_bits(torch, left, right) -> float:
    left = left.clamp_min(1e-20)
    right = right.clamp_min(1e-20)
    middle = 0.5 * (left + right)
    value = 0.5 * (left * (torch.log2(left) - torch.log2(middle))).sum()
    value += 0.5 * (right * (torch.log2(right) - torch.log2(middle))).sum()
    return float(value.item())


def build_token_steps(
    *,
    model_name: str,
    prompt: str,
    token_count: int,
    top_k: int,
    sample_top_k: int,
    temperature: float,
    seed: int,
    influence_method: str,
) -> list[TokenStep]:
    torch, tokenizer, model = load_model(model_name)
    torch.manual_seed(seed)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    generated = input_ids.clone()
    steps: list[TokenStep] = []

    for index in range(token_count):
        context = tokenizer.decode(generated[0])
        with torch.no_grad():
            outputs = model(generated, output_attentions=influence_method == "attention")
            logits = outputs.logits[0, -1] / temperature
            probabilities = torch.softmax(logits, dim=-1)
            sample_values, sample_indices = torch.topk(probabilities, min(sample_top_k, probabilities.shape[-1]))
            sample_values = sample_values / sample_values.sum()
            selected_position = torch.multinomial(sample_values, 1).item()
            selected_id = int(sample_indices[selected_position].item())

            top_values, top_indices = torch.topk(probabilities, top_k)
            top_ids = [int(value.item()) for value in top_indices]
            if selected_id not in top_ids:
                top_ids = top_ids[:-1] + [selected_id]

            selected_probability = float(probabilities[selected_id].item())
            entropy_bits = float(-(probabilities * torch.log2(probabilities.clamp_min(1e-20))).sum().item())
            if influence_method == "ablation":
                influence_scores: list[float] = []
                for token_position in range(generated.shape[1]):
                    ablated = torch.cat((generated[:, :token_position], generated[:, token_position + 1 :]), dim=1)
                    if ablated.shape[1] == 0:
                        influence_scores.append(0.0)
                        continue
                    ablated_logits = model(ablated).logits[0, -1] / temperature
                    ablated_probabilities = torch.softmax(ablated_logits, dim=-1)
                    influence_scores.append(js_divergence_bits(torch, probabilities, ablated_probabilities))
            elif influence_method == "attention":
                layer_attention = []
                for attention in outputs.attentions:
                    final_query_attention = attention[0, :, -1, :].mean(dim=0)
                    layer_attention.append(final_query_attention)
                influence_scores = torch.stack(layer_attention).mean(dim=0).tolist()
            else:
                influence_scores = []

        if influence_method == "logit-attribution":
            model.zero_grad(set_to_none=True)
            embeddings = model.get_input_embeddings()(generated).detach()
            embeddings.requires_grad_(True)
            attribution_outputs = model(inputs_embeds=embeddings)
            selected_logit = attribution_outputs.logits[0, -1, selected_id] / temperature
            selected_logit.backward()
            token_attribution = (embeddings.grad[0] * embeddings.detach()[0]).sum(dim=-1)
            influence_scores = torch.clamp(token_attribution, min=0.0).tolist()

        max_influence = max(influence_scores, default=0.0)
        context_tokens = []
        for token_id, score in zip(generated[0].tolist(), influence_scores):
            context_tokens.append(
                ContextToken(
                    display_token(tokenizer.decode([token_id])),
                    0.0 if max_influence <= 0 else score / max_influence,
                )
            )

        candidates = []
        for token_id in top_ids:
            text = tokenizer.decode([token_id])
            candidates.append(
                Candidate(
                    token_id,
                    display_token(text),
                    float(probabilities[token_id].item()),
                    token_id == selected_id,
                )
            )

        selected_text = tokenizer.decode([selected_id])
        surprisal = -math.log2(max(selected_probability, 1e-20))
        steps.append(
            TokenStep(
                index=index + 1,
                context=context,
                context_tokens=tuple(context_tokens),
                selected_text=selected_text,
                selected_probability=selected_probability,
                surprisal_bits=surprisal,
                token_perplexity=1.0 / max(selected_probability, 1e-20),
                entropy_bits=entropy_bits,
                effective_choices=2**entropy_bits,
                candidates=tuple(candidates),
            )
        )
        next_token = torch.tensor([[selected_id]], dtype=generated.dtype)
        generated = torch.cat([generated, next_token], dim=1)

    return steps


def build_timeline(steps: list[TokenStep], fps: int) -> list[FrameState]:
    timeline: list[FrameState] = []
    for step in steps:
        timeline.extend(FrameState(step, 0.0, "read") for _ in range(int(fps * READ_SECONDS)))
        reveal_frames = int(fps * REVEAL_SECONDS)
        for frame in range(reveal_frames):
            timeline.append(FrameState(step, frame / max(1, reveal_frames - 1), "choose", "swap" if frame == 0 else None))
        timeline.extend(FrameState(step, 1.0, "lock", "lock" if hold == 0 else None) for hold in range(int(fps * LOCK_SECONDS)))
    timeline.extend(FrameState(steps[-1], 1.0, "done", "sorted" if frame == 0 else None) for frame in range(int(fps * FINAL_HOLD_SECONDS)))
    return timeline


def blend(color_a: tuple[int, int, int], color_b: tuple[int, int, int], amount: float) -> tuple[int, int, int]:
    return tuple(int(bubble.lerp(color_a[channel], color_b[channel], amount)) for channel in range(3))


def wrap_text(draw: ImageDraw.ImageDraw, text: str, typeface, max_width: int, max_lines: int) -> list[str]:
    words = text.split(" ")
    lines: list[str] = []
    current = ""
    for word in words:
        trial = word if current == "" else f"{current} {word}"
        if bubble.text_width(draw, trial, typeface) <= max_width:
            current = trial
        else:
            if current:
                lines.append(current)
            current = word
        if len(lines) == max_lines:
            break
    if len(lines) < max_lines and current:
        lines.append(current)
    if len(lines) == max_lines and " ".join(lines) != text:
        while lines[-1] and bubble.text_width(draw, lines[-1] + "...", typeface) > max_width:
            lines[-1] = lines[-1][:-1]
        lines[-1] += "..."
    return lines


def uncertainty_color(step: TokenStep) -> tuple[int, int, int]:
    normalized = max(0.0, min(1.0, (step.entropy_bits - 4.5) / 6.5))
    return blend(BG_LOW_UNCERTAINTY, BG_HIGH_UNCERTAINTY, normalized)


def importance_color(importance: float) -> tuple[int, int, int]:
    amount = max(0.0, min(1.0, importance))
    return blend(TOKEN_RED, TOKEN_GREEN, amount)


def draw_header(draw: ImageDraw.ImageDraw, width: int, height: int, step: TokenStep) -> None:
    title = "LLM NEXT-TOKEN CHOICE"
    title_font = bubble.fit_font(draw, title, int(width * 0.066), int(width * 0.9), bold=True, min_size=int(width * 0.046))
    subtitle_font = bubble.font(int(width * 0.031))
    draw.text(((width - bubble.text_width(draw, title, title_font)) / 2, int(height * 0.04)), title, font=title_font, fill=bubble.TEXT)
    bubble.draw_centered_text(draw, int(height * 0.092), "The model scores thousands of possible next tokens.", subtitle_font, bubble.MUTED, width)

    margin = int(width * 0.065)
    badge_top = int(height * 0.13)
    badge_h = int(height * 0.052)
    bubble.rounded_rect(draw, (margin, badge_top, width - margin, badge_top + badge_h), 18, bubble.PANEL, bubble.GRID, 2)
    badge_font = bubble.font(int(width * 0.029), bold=True)
    small_font = bubble.font(int(width * 0.023))
    left = f"token {step.index}"
    right = "distribution before sampling"
    draw.text((margin + 22, badge_top + 20), left, font=badge_font, fill=TOKEN_YELLOW)
    draw.text((width - margin - 22 - bubble.text_width(draw, right, small_font), badge_top + 24), right, font=small_font, fill=bubble.MUTED)


def draw_context_panel(
    draw: ImageDraw.ImageDraw,
    width: int,
    height: int,
    step: TokenStep,
    reveal: float,
    influence_label: str,
) -> None:
    margin = int(width * 0.065)
    top = int(height * 0.205)
    bottom = int(height * 0.365)
    bubble.rounded_rect(draw, (margin, top, width - margin, bottom), 20, bubble.PANEL, bubble.GRID, 2)
    label_font = bubble.font(int(width * 0.024), bold=True)
    chip_font = bubble.font(int(width * 0.019), bold=True)

    draw.text((margin + 24, top + 18), influence_label, font=label_font, fill=bubble.MUTED)
    legend = "red less  |  green more"
    draw.text((width - margin - 24 - bubble.text_width(draw, legend, label_font), top + 18), legend, font=label_font, fill=bubble.MUTED)

    x = margin + 24
    y = top + 58
    max_x = width - margin - 24
    max_y = bottom - 20
    chip_h = int(height * 0.022)
    chip_gap = int(width * 0.007)
    row_gap = int(height * 0.006)

    def fit_chip_text(token_text: str) -> str:
        max_chip_text_w = int(width * 0.16)
        if bubble.text_width(draw, token_text, chip_font) <= max_chip_text_w:
            return token_text
        while token_text and bubble.text_width(draw, token_text + "...", chip_font) > max_chip_text_w:
            token_text = token_text[:-1]
        return token_text + "..."

    def draw_chip(token_text: str, color: tuple[int, int, int], outline: tuple[int, int, int] | None = None) -> bool:
        nonlocal x, y
        token_text = fit_chip_text(token_text)
        chip_w = max(int(width * 0.037), bubble.text_width(draw, token_text, chip_font) + int(width * 0.022))
        if x + chip_w > max_x:
            x = margin + 24
            y += chip_h + row_gap
        if y + chip_h > max_y:
            return False
        bubble.rounded_rect(draw, (x, y, x + chip_w, y + chip_h), 9, color, outline)
        draw.text((x + int(width * 0.011), y + int(height * 0.0048)), token_text, font=chip_font, fill=(18, 22, 30))
        x += chip_w + chip_gap
        return True

    for context_token in step.context_tokens:
        token_text = context_token.text
        color = importance_color(context_token.importance)
        if not draw_chip(token_text, color):
            break

    token_alpha = max(0.0, min(1.0, reveal))
    token_color = blend(bubble.PANEL, TOKEN_YELLOW, token_alpha)
    draw_chip(display_token(step.selected_text), token_color, TOKEN_YELLOW)


def draw_candidates(draw: ImageDraw.ImageDraw, width: int, height: int, step: TokenStep, reveal: float) -> None:
    margin = int(width * 0.065)
    top = int(height * 0.395)
    row_h = int(height * 0.042)
    gap = int(height * 0.01)
    max_prob = max(candidate.probability for candidate in step.candidates)
    label_font = bubble.font(int(width * 0.024), bold=True)
    token_font = bubble.font(int(width * 0.028), bold=True)
    prob_font = bubble.font(int(width * 0.022))

    draw.text((margin, top - 38), "possible next tokens", font=label_font, fill=bubble.MUTED)
    for index, candidate in enumerate(step.candidates):
        y0 = top + index * (row_h + gap)
        y1 = y0 + row_h
        selected_reveal = reveal if candidate.selected else 0.0
        border = TOKEN_YELLOW if candidate.selected else bubble.GRID
        fill = blend(bubble.PANEL, (38, 43, 56), 0.55)
        bubble.rounded_rect(draw, (margin, y0, width - margin, y1), 13, fill, border, 3 if candidate.selected else 1)

        token_text = candidate.text
        if bubble.text_width(draw, token_text, token_font) > int(width * 0.28):
            while token_text and bubble.text_width(draw, token_text + "...", token_font) > int(width * 0.28):
                token_text = token_text[:-1]
            token_text += "..."
        draw.text((margin + 18, y0 + 12), token_text, font=token_font, fill=TOKEN_YELLOW if candidate.selected else bubble.TEXT)

        bar_x0 = margin + int(width * 0.35)
        bar_x1 = width - margin - int(width * 0.13)
        bar_y0 = y0 + int(row_h * 0.34)
        bar_y1 = y0 + int(row_h * 0.68)
        bubble.rounded_rect(draw, (bar_x0, bar_y0, bar_x1, bar_y1), 8, BAR_BG, bubble.GRID, 1)
        fill_width = (bar_x1 - bar_x0 - 6) * (candidate.probability / max_prob)
        bar_color = TOKEN_YELLOW if candidate.selected else TOKEN_BLUE
        if candidate.selected:
            bar_color = blend(TOKEN_BLUE, TOKEN_YELLOW, 0.55 + 0.45 * selected_reveal)
        bubble.rounded_rect(draw, (bar_x0 + 3, bar_y0 + 3, bar_x0 + 3 + fill_width, bar_y1 - 3), 6, bar_color, None)

        prob_text = f"{candidate.probability * 100:.1f}%"
        draw.text((width - margin - 20 - bubble.text_width(draw, prob_text, prob_font), y0 + 17), prob_text, font=prob_font, fill=bubble.MUTED)


def metric_text(value: float) -> str:
    if value >= 1000:
        return f"{value / 1000:.1f}K"
    if value >= 100:
        return f"{value:.0f}"
    if value >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}"


def draw_metric_card(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    label: str,
    value: str,
    color: tuple[int, int, int],
    width: int,
) -> None:
    x0, y0, x1, y1 = box
    bubble.rounded_rect(draw, box, 16, bubble.PANEL, bubble.GRID, 2)
    value_font = bubble.font(int(width * 0.033), bold=True)
    label_font = bubble.font(int(width * 0.021))
    draw.text((x0 + 18, y0 + 13), value, font=value_font, fill=color)
    draw.text((x0 + 18, y1 - 34), label, font=label_font, fill=bubble.MUTED)


def draw_metrics(draw: ImageDraw.ImageDraw, width: int, height: int, step: TokenStep) -> None:
    margin = int(width * 0.065)
    top = int(height * 0.775)
    gap = int(width * 0.022)
    card_h = int(height * 0.058)
    card_w = int((width - margin * 2 - gap * 3) / 4)
    uncertainty = uncertainty_color(step)
    cards = (
        ("chosen prob", f"{step.selected_probability * 100:.1f}%", TOKEN_YELLOW),
        ("surprisal", f"{step.surprisal_bits:.2f} bits", TOKEN_BLUE),
        ("token ppl", metric_text(step.token_perplexity), TOKEN_RED if step.token_perplexity > 50 else TOKEN_GREEN),
        ("entropy", f"{step.entropy_bits:.2f} bits", uncertainty),
    )
    for index, (label, value, color) in enumerate(cards):
        x0 = margin + index * (card_w + gap)
        draw_metric_card(draw, (x0, top, x0 + card_w, top + card_h), label, value, color, width)

    panel_top = top + card_h + int(height * 0.018)
    panel_h = int(height * 0.07)
    bubble.rounded_rect(draw, (margin, panel_top, width - margin, panel_top + panel_h), 16, bubble.PANEL, bubble.GRID, 2)
    label_font = bubble.font(int(width * 0.028), bold=True)
    small_font = bubble.font(int(width * 0.024))
    draw.text((margin + 22, panel_top + 18), f"effective choices: about {metric_text(step.effective_choices)}", font=label_font, fill=uncertainty)
    draw.text((margin + 22, panel_top + 66), "high entropy = many plausible continuations", font=small_font, fill=bubble.MUTED)


def draw_frame(
    *,
    width: int,
    height: int,
    state: FrameState,
    frame_number: int,
    total_frames: int,
    total_tokens: int,
    influence_label: str,
) -> Image.Image:
    image = bubble.gradient_background(width, height).copy()
    draw = ImageDraw.Draw(image)
    step = state.step
    draw_header(draw, width, height, step)
    draw_context_panel(draw, width, height, step, state.reveal, influence_label)
    draw_candidates(draw, width, height, step, state.reveal)
    draw_metrics(draw, width, height, step)

    margin = int(width * 0.065)
    progress_top = int(height * 0.942)
    progress_h = int(height * 0.016)
    progress = step.index / max(1, total_tokens)
    small_font = bubble.font(int(width * 0.021))
    label = "generation progress"
    draw.text((margin, progress_top - 33), label, font=small_font, fill=bubble.MUTED)
    draw.text((width - margin - bubble.text_width(draw, f"{progress:.2f}", small_font), progress_top - 33), f"{progress:.2f}", font=small_font, fill=TOKEN_GREEN)
    bubble.rounded_rect(draw, (margin, progress_top, width - margin, progress_top + progress_h), 12, BAR_BG, bubble.GRID, 2)
    bubble.rounded_rect(
        draw,
        (margin + 4, progress_top + 4, bubble.lerp(margin + 4, width - margin - 4, progress), progress_top + progress_h - 4),
        8,
        TOKEN_GREEN,
        None,
    )
    footer_font = bubble.font(int(width * 0.019))
    footer = f"frame {frame_number + 1}/{total_frames} | logits -> softmax -> sample one token"
    draw.text((margin, int(height * 0.966)), footer, font=footer_font, fill=bubble.MUTED)
    return image

def render_video(args: argparse.Namespace) -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required but was not found on PATH")

    output = args.output
    thumbnail = args.thumbnail
    output.parent.mkdir(parents=True, exist_ok=True)
    thumbnail.parent.mkdir(parents=True, exist_ok=True)

    steps = build_token_steps(
        model_name=args.model,
        prompt=args.prompt,
        token_count=args.tokens,
        top_k=args.top_k,
        sample_top_k=args.sample_top_k,
        temperature=args.temperature,
        seed=args.seed,
        influence_method=args.influence_method,
    )
    timeline = build_timeline(steps, args.fps)
    audio_enabled = not args.no_audio
    temp_context = tempfile.TemporaryDirectory() if audio_enabled else nullcontext(None)

    # Patch progress denominator by wrapping draw_frame with the exact token count.
    def draw_exact_frame(frame_state: FrameState, frame_number: int) -> Image.Image:
        image = draw_frame(
            width=args.width,
            height=args.height,
            state=frame_state,
            frame_number=frame_number,
            total_frames=len(timeline),
            total_tokens=args.tokens,
            influence_label=INFLUENCE_LABELS[args.influence_method],
        )
        return image

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
                image = draw_exact_frame(frame_state, frame_number)
                if not thumbnail_saved and frame_number >= min(len(timeline) - 1, args.fps * 8):
                    image.save(thumbnail)
                    thumbnail_saved = True
                process.stdin.write(image.tobytes())
        finally:
            process.stdin.close()

        return_code = process.wait()
        if return_code != 0:
            raise RuntimeError(f"ffmpeg exited with status {return_code}")

        if not thumbnail_saved:
            draw_exact_frame(timeline[-1], len(timeline) - 1).save(thumbnail)

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

    final_text = steps[-1].context + steps[-1].selected_text
    print(f"Rendered {output}")
    print(f"Rendered {thumbnail}")
    print(f"Model: {args.model}")
    print(f"Frames: {len(timeline)} at {args.fps} fps ({len(timeline) / args.fps:.1f}s)")
    print(f"Audio: {'on' if audio_enabled else 'off'}")
    print(f"Prompt: {args.prompt}")
    print(f"Context coloring: {args.influence_method}")
    print(f"Generated: {final_text}")
    for step in steps:
        print(
            f"{step.index:02d} {display_token(step.selected_text)!r} "
            f"p={step.selected_probability:.4f} entropy={step.entropy_bits:.2f} "
            f"surprisal={step.surprisal_bits:.2f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--thumbnail", type=Path, default=DEFAULT_THUMBNAIL)
    parser.add_argument("--width", type=int, default=1080)
    parser.add_argument("--height", type=int, default=1920)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--tokens", type=int, default=14)
    parser.add_argument("--top-k", type=int, default=7)
    parser.add_argument("--sample-top-k", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.85)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--influence-method", choices=sorted(INFLUENCE_LABELS), default="ablation")
    parser.add_argument("--no-audio", action="store_true")
    args = parser.parse_args()
    if args.width <= 0 or args.height <= 0 or args.fps <= 0:
        parser.error("--width, --height, and --fps must be positive")
    if args.tokens <= 0 or args.top_k <= 0 or args.sample_top_k <= 0:
        parser.error("--tokens, --top-k, and --sample-top-k must be positive")
    if args.temperature <= 0:
        parser.error("--temperature must be positive")
    return args


if __name__ == "__main__":
    render_video(parse_args())
