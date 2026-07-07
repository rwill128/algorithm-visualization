#!/usr/bin/env python3
"""Render a Shorts-ready Q-learning gridworld visualization."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
import random
import shutil
import subprocess
import tempfile

from PIL import Image, ImageDraw

import render_bubble_sort_short as bubble


ROOT = Path(__file__).resolve().parents[1]
SHORTS_DIR = ROOT / "artifacts" / "shorts"
DEFAULT_OUTPUT = SHORTS_DIR / "q_learning_gridworld.mp4"
DEFAULT_THUMBNAIL = SHORTS_DIR / "q_learning_gridworld_thumbnail.png"

ROWS = 9
COLS = 9
START = (7, 1)
GOAL = (1, 7)
Coord = tuple[int, int]

WALLS: frozenset[Coord] = frozenset(
    {
        (1, 3),
        (2, 1),
        (2, 2),
        (2, 3),
        (3, 3),
        (3, 5),
        (3, 6),
        (4, 3),
        (4, 5),
        (5, 1),
        (5, 2),
        (5, 3),
        (5, 5),
        (6, 5),
        (7, 3),
        (7, 4),
        (7, 5),
    }
)
PITS: frozenset[Coord] = frozenset({(4, 7), (6, 7)})
OPEN_CELLS: tuple[Coord, ...] = tuple((row, col) for row in range(ROWS) for col in range(COLS) if (row, col) not in WALLS)

ACTIONS: tuple[tuple[str, tuple[int, int]], ...] = (
    ("U", (-1, 0)),
    ("R", (0, 1)),
    ("D", (1, 0)),
    ("L", (0, -1)),
)
ACTION_INDEX = {name: index for index, (name, _) in enumerate(ACTIONS)}

STEP_REWARD = -0.035
BUMP_REWARD = -0.18
GOAL_REWARD = 1.0
PIT_REWARD = -1.0

VALUE_LOW = (255, 104, 91)
VALUE_HIGH = (126, 220, 135)
VALUE_NEUTRAL = (30, 37, 51)
WALL = (66, 74, 91)
PIT = (126, 43, 52)
START_COLOR = (86, 176, 255)
GOAL_COLOR = (126, 220, 135)
AGENT = (255, 202, 77)
TRAIL = (54, 121, 190)
GRID_EDGE = (49, 58, 76)


@dataclass(frozen=True)
class Rollout:
    path: tuple[Coord, ...]
    actions: tuple[int, ...]
    rewards: tuple[float, ...]
    total_reward: float
    done: bool
    outcome: str


@dataclass(frozen=True)
class Checkpoint:
    label: str
    episodes: int
    q_values: dict[Coord, tuple[float, float, float, float]]
    rollout: Rollout
    recent_success: float
    updates: int


@dataclass(frozen=True)
class FrameState:
    checkpoint: Checkpoint
    step_index: int
    pulse: float = 0.0
    audio_event: str | None = None


def make_q_table() -> dict[Coord, list[float]]:
    return {cell: [0.0, 0.0, 0.0, 0.0] for cell in OPEN_CELLS}


def freeze_q_table(q_values: dict[Coord, list[float]]) -> dict[Coord, tuple[float, float, float, float]]:
    return {cell: tuple(values) for cell, values in q_values.items()}


def step_environment(cell: Coord, action_index: int) -> tuple[Coord, float, bool]:
    if cell == GOAL or cell in PITS:
        return cell, 0.0, True
    row_delta, col_delta = ACTIONS[action_index][1]
    next_cell = (cell[0] + row_delta, cell[1] + col_delta)
    if not (0 <= next_cell[0] < ROWS and 0 <= next_cell[1] < COLS) or next_cell in WALLS:
        return cell, BUMP_REWARD, False
    if next_cell == GOAL:
        return next_cell, GOAL_REWARD, True
    if next_cell in PITS:
        return next_cell, PIT_REWARD, True
    return next_cell, STEP_REWARD, False


def greedy_action(q_values: dict[Coord, tuple[float, float, float, float]] | dict[Coord, list[float]], cell: Coord, rng: random.Random | None = None) -> int:
    values = q_values[cell]
    best_value = max(values)
    candidates = [index for index, value in enumerate(values) if abs(value - best_value) < 1e-10]
    if rng is not None:
        return rng.choice(candidates)
    return candidates[0]


def run_rollout(
    q_values: dict[Coord, tuple[float, float, float, float]],
    *,
    seed: int,
    random_policy: bool,
    max_steps: int,
) -> Rollout:
    rng = random.Random(seed)
    cell = START
    path = [cell]
    actions: list[int] = []
    rewards: list[float] = []
    total_reward = 0.0
    done = False

    for _ in range(max_steps):
        action = rng.randrange(len(ACTIONS)) if random_policy else greedy_action(q_values, cell)
        next_cell, reward, done = step_environment(cell, action)
        actions.append(action)
        rewards.append(reward)
        total_reward += reward
        path.append(next_cell)
        cell = next_cell
        if done:
            break

    if path[-1] == GOAL:
        outcome = "goal reached"
    elif path[-1] in PITS:
        outcome = "penalty cell"
    else:
        outcome = "episode timed out"

    return Rollout(tuple(path), tuple(actions), tuple(rewards), total_reward, done, outcome)


def train_checkpoints(seed: int) -> list[Checkpoint]:
    q_values = make_q_table()
    rng = random.Random(seed)
    checkpoints = {0: freeze_q_table(q_values)}
    success_history: list[int] = []
    success_at_episode = {0: 0.0}
    updates_at_episode = {0: 0}
    updates = 0

    for episode in range(1, 501):
        cell = START
        epsilon = max(0.05, 0.8 * (0.99**episode))
        reached_goal = False

        for _ in range(70):
            if rng.random() < epsilon:
                action = rng.randrange(len(ACTIONS))
            else:
                action = greedy_action(q_values, cell, rng)
            next_cell, reward, done = step_environment(cell, action)
            future = 0.0 if done else max(q_values[next_cell])
            target = reward + 0.94 * future
            q_values[cell][action] += 0.18 * (target - q_values[cell][action])
            updates += 1
            cell = next_cell
            if done:
                reached_goal = cell == GOAL
                break

        success_history.append(1 if reached_goal else 0)
        if episode in {20, 100, 500}:
            checkpoints[episode] = freeze_q_table(q_values)
            window = success_history[-50:]
            success_at_episode[episode] = sum(window) / max(1, len(window))
            updates_at_episode[episode] = updates

    specs = (
        ("UNTRAINED POLICY", 0, 62, True, 45),
        ("AFTER 20 EPISODES", 20, 119, False, 42),
        ("AFTER 100 EPISODES", 100, 199, False, 60),
        ("AFTER 500 EPISODES", 500, 599, False, 60),
    )
    result: list[Checkpoint] = []
    for label, episodes, rollout_seed, random_policy, max_steps in specs:
        q_table = checkpoints[episodes]
        result.append(
            Checkpoint(
                label,
                episodes,
                q_table,
                run_rollout(q_table, seed=rollout_seed, random_policy=random_policy, max_steps=max_steps),
                success_at_episode.get(episodes, 0.0),
                updates_at_episode.get(episodes, 0),
            )
        )
    return result


def build_timeline(checkpoints: list[Checkpoint], fps: int) -> list[FrameState]:
    timeline: list[FrameState] = []
    for checkpoint_index, checkpoint in enumerate(checkpoints):
        hold_frames = int(fps * (1.15 if checkpoint_index == 0 else 0.9))
        for frame in range(hold_frames):
            timeline.append(FrameState(checkpoint, 0, frame / max(1, hold_frames - 1), "lock" if frame == 0 and checkpoint_index else None))

        frames_per_step = 12
        for step_index in range(1, len(checkpoint.rollout.path)):
            for frame in range(frames_per_step):
                event = "swap" if frame == 0 and step_index % 3 == 0 else None
                timeline.append(FrameState(checkpoint, step_index, frame / (frames_per_step - 1), event))

        terminal_hold = int(fps * (2.6 if checkpoint_index == len(checkpoints) - 1 else 1.2))
        for frame in range(terminal_hold):
            if frame == 0:
                event = "sorted" if checkpoint.rollout.path[-1] == GOAL and checkpoint_index == len(checkpoints) - 1 else "lock"
            else:
                event = None
            timeline.append(FrameState(checkpoint, len(checkpoint.rollout.path) - 1, frame / max(1, terminal_hold - 1), event))
    return timeline


def blend(color_a: tuple[int, int, int], color_b: tuple[int, int, int], amount: float) -> tuple[int, int, int]:
    return tuple(int(bubble.lerp(color_a[channel], color_b[channel], amount)) for channel in range(3))


def value_for_cell(checkpoint: Checkpoint, cell: Coord) -> float:
    if cell == GOAL:
        return GOAL_REWARD
    if cell in PITS:
        return PIT_REWARD
    if cell in WALLS:
        return 0.0
    return max(checkpoint.q_values[cell])


def color_for_value(value: float) -> tuple[int, int, int]:
    clipped = max(-1.0, min(1.0, value))
    if clipped >= 0:
        return blend(VALUE_NEUTRAL, VALUE_HIGH, clipped)
    return blend(VALUE_NEUTRAL, VALUE_LOW, -clipped)


def grid_geometry(width: int, height: int) -> tuple[int, int, int, int]:
    cell = min(int(width * 0.086), int(height * 0.052))
    grid_width = cell * COLS
    left = (width - grid_width) // 2
    top = int(height * 0.22)
    return left, top, cell, max(3, int(cell * 0.08))


def draw_legend(draw: ImageDraw.ImageDraw, width: int, y: int) -> None:
    font = bubble.font(int(width * 0.021), bold=True)
    items = (("low value", VALUE_LOW), ("neutral", VALUE_NEUTRAL), ("high value", VALUE_HIGH), ("wall", WALL), ("agent", AGENT))
    x = int(width * 0.07)
    swatch = int(width * 0.022)
    for label, color in items:
        bubble.rounded_rect(draw, (x, y, x + swatch, y + swatch), 5, color, None)
        draw.text((x + swatch + 8, y - 2), label, font=font, fill=bubble.MUTED)
        x += swatch + 10 + bubble.text_width(draw, label, font) + int(width * 0.018)


def draw_agent(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], pulse: float) -> None:
    x0, y0, x1, y1 = box
    cx = (x0 + x1) / 2
    cy = (y0 + y1) / 2
    radius = (x1 - x0) * (0.27 + 0.04 * pulse)
    draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=AGENT, outline=(255, 238, 168), width=3)


def draw_grid(draw: ImageDraw.ImageDraw, checkpoint: Checkpoint, step_index: int, pulse: float, width: int, height: int) -> None:
    left, top, cell, gap = grid_geometry(width, height)
    grid_width = cell * COLS
    grid_height = cell * ROWS
    bubble.rounded_rect(draw, (left - 16, top - 16, left + grid_width + 16, top + grid_height + 16), 22, bubble.PANEL, GRID_EDGE, 3)

    current_path = checkpoint.rollout.path[: step_index + 1]
    visited = set(current_path[:-1])
    agent_cell = current_path[-1]
    policy_font = bubble.font(int(width * 0.021), bold=True)
    label_font = bubble.font(int(width * 0.031), bold=True)

    for row in range(ROWS):
        for col in range(COLS):
            cell_coord = (row, col)
            x0 = left + col * cell + gap
            y0 = top + row * cell + gap
            x1 = left + (col + 1) * cell - gap
            y1 = top + (row + 1) * cell - gap

            if cell_coord in WALLS:
                color = WALL
            elif cell_coord in PITS:
                color = blend(PIT, VALUE_LOW, 0.35)
            else:
                color = color_for_value(value_for_cell(checkpoint, cell_coord))
                if cell_coord in visited:
                    color = blend(color, TRAIL, 0.35)
                if cell_coord == START:
                    color = blend(color, START_COLOR, 0.48)
                if cell_coord == GOAL:
                    color = blend(color, GOAL_COLOR, 0.5)

            bubble.rounded_rect(draw, (x0, y0, x1, y1), max(5, int(cell * 0.14)), color, None)

            if cell_coord == START:
                draw.text((x0 + 10, y0 + 5), "S", font=label_font, fill=bubble.TEXT)
            elif cell_coord == GOAL:
                draw.text((x0 + 10, y0 + 5), "G", font=label_font, fill=bubble.TEXT)
            elif cell_coord in PITS:
                draw.text((x0 + 14, y0 + 5), "!", font=label_font, fill=bubble.TEXT)
            elif checkpoint.episodes > 0 and cell_coord not in WALLS:
                value = value_for_cell(checkpoint, cell_coord)
                if abs(value) > 0.015:
                    action_name = ACTIONS[greedy_action(checkpoint.q_values, cell_coord)][0]
                    draw.text((x1 - 20, y1 - 26), action_name, font=policy_font, fill=blend(bubble.TEXT, bubble.MUTED, 0.25))

    if agent_cell not in WALLS:
        row, col = agent_cell
        box = (
            left + col * cell + gap,
            top + row * cell + gap,
            left + (col + 1) * cell - gap,
            top + (row + 1) * cell - gap,
        )
        draw_agent(draw, box, pulse)


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
    value_font = bubble.font(int(width * 0.034), bold=True)
    label_font = bubble.font(int(width * 0.022))
    draw.text((x0 + 18, y0 + 14), value, font=value_font, fill=color)
    draw.text((x0 + 18, y1 - 36), label, font=label_font, fill=bubble.MUTED)


def draw_metrics(draw: ImageDraw.ImageDraw, checkpoint: Checkpoint, step_index: int, width: int, height: int) -> None:
    margin = int(width * 0.065)
    top = int(height * 0.765)
    gap = int(width * 0.022)
    card_height = int(height * 0.058)
    card_width = int((width - margin * 2 - gap * 3) / 4)
    rollout = checkpoint.rollout
    current_reward = sum(rollout.rewards[:step_index])
    success_text = "random" if checkpoint.episodes == 0 else f"{checkpoint.recent_success * 100:.0f}%"
    cards = (
        ("episodes trained", str(checkpoint.episodes), AGENT),
        ("success recent", success_text, GOAL_COLOR),
        ("steps this episode", f"{min(step_index, len(rollout.actions))}/{len(rollout.actions)}", bubble.BAR_ALT),
        ("return so far", f"{current_reward:+.2f}", VALUE_HIGH if current_reward >= 0 else VALUE_LOW),
    )
    for index, (label, value, color) in enumerate(cards):
        x0 = margin + index * (card_width + gap)
        draw_metric_card(draw, (x0, top, x0 + card_width, top + card_height), label, value, color, width)

    panel_top = top + card_height + int(height * 0.016)
    panel_height = int(height * 0.08)
    bubble.rounded_rect(draw, (margin, panel_top, width - margin, panel_top + panel_height), 16, bubble.PANEL, bubble.GRID, 2)
    left_font = bubble.font(int(width * 0.029), bold=True)
    right_font = bubble.font(int(width * 0.026))
    outcome_color = GOAL_COLOR if rollout.path[min(step_index, len(rollout.path) - 1)] == GOAL else VALUE_LOW if rollout.path[min(step_index, len(rollout.path) - 1)] in PITS else bubble.MUTED
    status = rollout.outcome if step_index == len(rollout.path) - 1 else "policy is choosing actions from current values"
    draw.text((margin + 22, panel_top + 18), status, font=left_font, fill=outcome_color)
    rule = "cell color = V(s) = best expected future reward"
    draw.text((margin + 22, panel_top + 76), rule, font=right_font, fill=bubble.MUTED)

    bar_top = int(height * 0.932)
    bar_height = int(height * 0.018)
    progress = min(1.0, step_index / max(1, len(rollout.path) - 1))
    small = bubble.font(int(width * 0.022))
    draw.text((margin, bar_top - 34), "episode progress", font=small, fill=bubble.MUTED)
    draw.text((width - margin - bubble.text_width(draw, f"{progress:.2f}", small), bar_top - 34), f"{progress:.2f}", font=small, fill=GOAL_COLOR)
    bubble.rounded_rect(draw, (margin, bar_top, width - margin, bar_top + bar_height), 12, (15, 18, 25), bubble.GRID, 2)
    bubble.rounded_rect(
        draw,
        (margin + 4, bar_top + 4, bubble.lerp(margin + 4, width - margin - 4, progress), bar_top + bar_height - 4),
        8,
        GOAL_COLOR,
        None,
    )


def draw_header(draw: ImageDraw.ImageDraw, checkpoint: Checkpoint, width: int, height: int) -> None:
    margin = int(width * 0.065)
    title = "Q-LEARNING GRIDWORLD"
    title_font = bubble.fit_font(draw, title, int(width * 0.071), int(width * 0.9), bold=True, min_size=int(width * 0.048))
    subtitle_font = bubble.font(int(width * 0.032))
    draw.text(((width - bubble.text_width(draw, title, title_font)) / 2, int(height * 0.044)), title, font=title_font, fill=bubble.TEXT)
    bubble.draw_centered_text(draw, int(height * 0.096), "A policy learns which cells are worth moving toward.", subtitle_font, bubble.MUTED, width)

    badge_y = int(height * 0.137)
    badge_h = int(height * 0.058)
    bubble.rounded_rect(draw, (margin, badge_y, width - margin, badge_y + badge_h), 18, bubble.PANEL, bubble.GRID, 2)
    label_font = bubble.font(int(width * 0.032), bold=True)
    small_font = bubble.font(int(width * 0.024))
    draw.text((margin + 24, badge_y + 20), checkpoint.label, font=label_font, fill=AGENT if checkpoint.episodes == 0 else GOAL_COLOR)
    right = "random actions" if checkpoint.episodes == 0 else "greedy policy from learned Q-values"
    draw.text((width - margin - 24 - bubble.text_width(draw, right, small_font), badge_y + 27), right, font=small_font, fill=bubble.MUTED)
    draw_legend(draw, width, int(height * 0.202))


def draw_frame(
    *,
    width: int,
    height: int,
    state: FrameState,
    frame_number: int,
    total_frames: int,
) -> Image.Image:
    image = bubble.gradient_background(width, height).copy()
    draw = ImageDraw.Draw(image)
    draw_header(draw, state.checkpoint, width, height)
    draw_grid(draw, state.checkpoint, state.step_index, state.pulse, width, height)
    draw_metrics(draw, state.checkpoint, state.step_index, width, height)
    footer_font = bubble.font(int(width * 0.02))
    footer = f"frame {frame_number + 1}/{total_frames} | tabular Q-learning updates values from reward plus future value"
    draw.text((int(width * 0.065), int(height * 0.965)), footer, font=footer_font, fill=bubble.MUTED)
    return image


def render_video(args: argparse.Namespace) -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required but was not found on PATH")

    output = args.output
    thumbnail = args.thumbnail
    output.parent.mkdir(parents=True, exist_ok=True)
    thumbnail.parent.mkdir(parents=True, exist_ok=True)

    checkpoints = train_checkpoints(args.seed)
    timeline = build_timeline(checkpoints, args.fps)
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
            for frame_number, state in enumerate(timeline):
                image = draw_frame(width=args.width, height=args.height, state=state, frame_number=frame_number, total_frames=len(timeline))
                if not thumbnail_saved and frame_number >= min(len(timeline) - 1, args.fps * 12):
                    image.save(thumbnail)
                    thumbnail_saved = True
                process.stdin.write(image.tobytes())
        finally:
            process.stdin.close()

        return_code = process.wait()
        if return_code != 0:
            raise RuntimeError(f"ffmpeg exited with status {return_code}")

        if not thumbnail_saved:
            draw_frame(width=args.width, height=args.height, state=timeline[-1], frame_number=len(timeline) - 1, total_frames=len(timeline)).save(thumbnail)

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

    print(f"Rendered {output}")
    print(f"Rendered {thumbnail}")
    print(f"Frames: {len(timeline)} at {args.fps} fps ({len(timeline) / args.fps:.1f}s)")
    print(f"Audio: {'on' if audio_enabled else 'off'}")
    for checkpoint in checkpoints:
        rollout = checkpoint.rollout
        print(
            f"{checkpoint.label}: steps={len(rollout.actions)} outcome={rollout.outcome} "
            f"return={rollout.total_reward:+.2f} recent_success={checkpoint.recent_success:.2f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--thumbnail", type=Path, default=DEFAULT_THUMBNAIL)
    parser.add_argument("--width", type=int, default=1080)
    parser.add_argument("--height", type=int, default=1920)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument("--no-audio", action="store_true")
    args = parser.parse_args()
    if args.width <= 0 or args.height <= 0 or args.fps <= 0:
        parser.error("--width, --height, and --fps must be positive")
    return args


if __name__ == "__main__":
    render_video(parse_args())
