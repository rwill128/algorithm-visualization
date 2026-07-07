# Algorithm Visualizations

Shorts-first algorithm visualization pipeline.

## First video

The first render is a vertical Bubble Sort visualization:

- Format: 1080x1920
- Frame rate: 60 fps
- Audio: C/E major third on swaps, C octave when a value locks into place, F/C resolution stack when the list is sorted
- Sortedness: bottom `Kendall tau sortedness` bar fills by normalized Kendall tau progress
- Output: `artifacts/shorts/bubble_sort.mp4`
- Thumbnail: `artifacts/shorts/bubble_sort_thumbnail.png`

The matching Merge Sort render uses the same 24-element vertical format:

- Audio: C/E major third on writes, C octave when a sorted run completes, F/C resolution stack when the list is sorted
- Progress: bottom `merge progress` bar fills by completed array writes
- Output: `artifacts/shorts/merge_sort.mp4`
- Thumbnail: `artifacts/shorts/merge_sort_thumbnail.png`

The elementary-sort short renderer uses the same bar, comparison, operation-count,
Kendall tau, and audio language for the rest of the configured elementary sorts:

- Insertion Sort: `artifacts/shorts/insertion_sort.mp4`
- Selection Sort: `artifacts/shorts/selection_sort.mp4`
- Gnome Sort: `artifacts/shorts/gnome_sort.mp4`
- Cocktail Sort: `artifacts/shorts/cocktail_sort.mp4`
- Odd-Even Sort: `artifacts/shorts/odd-even_sort.mp4`

The operation-meter renderer shows a sorting animation with separate live meters
for comparisons, reads, writes, swaps, and peak auxiliary memory:

- Bubble Sort: `artifacts/shorts/bubble_sort_operation_meters.mp4`
- Insertion Sort: `artifacts/shorts/insertion_sort_operation_meters.mp4`
- Selection Sort: `artifacts/shorts/selection_sort_operation_meters.mp4`
- Merge Sort: `artifacts/shorts/merge_sort_operation_meters.mp4`
- Gnome Sort: `artifacts/shorts/gnome_sort_operation_meters.mp4`
- Cocktail Sort: `artifacts/shorts/cocktail_sort_operation_meters.mp4`
- Odd-Even Sort: `artifacts/shorts/odd-even_sort_operation_meters.mp4`

## Connect Four videos

The Connect Four evaluator renderer shows a board position, checks legal drops,
scores four-cell windows, and short-circuits immediately when a candidate move
creates a terminal win:

- Basic Evaluator: `artifacts/shorts/connect_four_evaluator.mp4`

The Connect Four minimax renderer uses that same evaluator only at leaf
positions. A depth-limited alpha-beta search compares root moves, counts searched
nodes/leaves/prunes, and shows how lookahead can reject the one-ply evaluator's
favorite move:

- Alpha-Beta Minimax: `artifacts/shorts/connect_four_minimax.mp4`

## Pathfinding videos

The pathfinding renderer shows grid search with obstacles. It highlights the
frontier, explored cells, current cell, final reconstructed path, and live search
counters:

- A* Pathfinding: `artifacts/shorts/a_star_pathfinding.mp4`
- Breadth-First Search: `artifacts/shorts/breadth_first_search.mp4`
- Depth-First Search: `artifacts/shorts/depth_first_search.mp4`
- Dijkstra Pathfinding: `artifacts/shorts/dijkstra_pathfinding.mp4`
- Greedy Best-First Search: `artifacts/shorts/greedy_best_first.mp4`

The A* math walkthrough renderer slows the search down to 10 expansions,
showing the `f = g + h` arithmetic for each neighbor and the resulting frontier
priority queue:

- A* Math Walkthrough: `artifacts/shorts/a_star_math_walkthrough.mp4`

The Greedy Best-First math walkthrough uses the same slow neighbor-by-neighbor
style, but scores the frontier with the Manhattan heuristic `h` only:

- Greedy Best-First Math Walkthrough: `artifacts/shorts/greedy_best_first_math_walkthrough.mp4`

The Dijkstra math walkthrough scores the frontier with exact cumulative path
cost `g`, including a few higher-cost terrain cells so the queue behavior is
visible:

- Dijkstra Math Walkthrough: `artifacts/shorts/dijkstra_math_walkthrough.mp4`

The pathfinding distribution renderer flashes several random wall worlds before
plotting random solvable grid samples by grid area and search cost. The world
generator uses long connected barriers, sparse segment walls, randomized
start/goal orientation, and a solvability check so the samples include more
maze-like bad cases. Expanded cells are the default plotted unit; edge checks
can be selected with `--metric edge-checks`:

- A* Expanded-Cell Distribution: `artifacts/shorts/a_star_distribution_expanded.mp4`
- Breadth-First Expanded-Cell Distribution: `artifacts/shorts/breadth_first_distribution_expanded.mp4`
- Depth-First Expanded-Cell Distribution: `artifacts/shorts/depth_first_distribution_expanded.mp4`
- Dijkstra Expanded-Cell Distribution: `artifacts/shorts/dijkstra_distribution_expanded.mp4`
- Greedy Best-First Expanded-Cell Distribution: `artifacts/shorts/greedy_best_first_distribution_expanded.mp4`

The combined pathfinding comparison renderer sequences each algorithm's random
world examples, then plots its cost distribution on a shared graph while prior
distributions remain greyed out.

- Pathfinding Algorithm Comparison Distribution: `artifacts/shorts/pathfinding_algorithm_comparison_distribution.mp4`

## Chess engine videos

The bitboard renderer introduces Stockfish-style board representation with a
64-bit pawn mask, then shows pawn pushes as a north shift filtered by empty
squares:

- Bitboard Pawn Shift: `artifacts/shorts/bitboard_pawn_shift.mp4`

## Reinforcement learning videos

The gridworld renderer shows tabular Q-learning checkpoints. Each checkpoint
colors cells by the learned value estimate `V(s)`, then plays one episode using
the current policy so the viewer can compare untrained behavior against later
training:

- Q-Learning Gridworld: `artifacts/shorts/q_learning_gridworld.mp4`

## LLM videos

The next-token renderer runs a local GPT-2 model, records the probability
distribution at each generation step, and visualizes the sampled token against
the other high-probability candidates. It also shows chosen-token probability,
surprisal, token-level perplexity, entropy, and effective choices:

- LLM Next-Token Choice: `artifacts/shorts/llm_next_token_probabilities.mp4`
- LLM Next-Token Attention: `artifacts/shorts/llm_next_token_attention.mp4`
- LLM Next-Token Logit Attribution: `artifacts/shorts/llm_next_token_logit_attribution.mp4`
- Logit Attribution Explained: `artifacts/shorts/llm_logit_attribution_explained.mp4`

## Operation-cloud videos

The operation-cloud renderer creates vertical sorting-cost charts:

- Format: 1080x1920
- Frame rate: 60 fps
- X axis: elements in list, from 2 to 100 by default
- Y axis: operations, with all operation types summed into one bucket
- Best/worst curves: algorithm-specific operation envelopes
- Samples: random list sizes with uniformly shuffled permutations, plotted over time
- Default output: `artifacts/shorts/bubble_sort_operation_cloud.mp4`

The combined comparison renderer animates each algorithm sequentially on one shared
axis. When an algorithm finishes, its best/worst curves and point cloud fade to
grey and remain on-screen while the next algorithm draws.

- Default output: `artifacts/shorts/sorting_algorithm_comparison_cloud.mp4`

Currently configured algorithms:

- Bubble Sort: `artifacts/shorts/bubble_sort_operation_cloud.mp4`
- Insertion Sort: `artifacts/shorts/insertion_sort_operation_cloud.mp4`
- Selection Sort: `artifacts/shorts/selection_sort_operation_cloud.mp4`
- Merge Sort: `artifacts/shorts/merge_sort_operation_cloud.mp4`
- Gnome Sort: `artifacts/shorts/gnome_sort_operation_cloud.mp4`
- Cocktail Shaker Sort: `artifacts/shorts/cocktail_sort_operation_cloud.mp4`
- Odd-Even Sort: `artifacts/shorts/odd-even_sort_operation_cloud.mp4`

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Curated YouTube upload queue

Reviewed Shorts live in `data/youtube_upload_queue.json`. The scheduled uploader
only uploads items where both `enabled` and `approved_for_upload` are true, and
it records the resulting YouTube URL back into that queue. Title and description
hashtags are also maintained in the queue and appended automatically at upload
time.

Dry-run the next scheduled upload:

```bash
.venv/bin/python scripts/youtube_upload_queue.py upload-next --limit 1 --dry-run
```

Upload one approved item:

```bash
.venv/bin/python scripts/youtube_upload_queue.py upload-next --limit 1 --approved --allow-public
```

List queue state:

```bash
.venv/bin/python scripts/youtube_upload_queue.py list
```

## Render

```bash
.venv/bin/python scripts/render_bubble_sort_short.py
.venv/bin/python scripts/render_merge_sort_short.py
.venv/bin/python scripts/render_elementary_sort_shorts.py --all
.venv/bin/python scripts/render_operation_meter_short.py --all
.venv/bin/python scripts/render_connect_four_evaluator_short.py
.venv/bin/python scripts/render_connect_four_minimax_short.py
.venv/bin/python scripts/render_pathfinding_short.py
.venv/bin/python scripts/render_pathfinding_short.py --all
.venv/bin/python scripts/render_astar_math_walkthrough.py
.venv/bin/python scripts/render_greedy_best_first_math_walkthrough.py
.venv/bin/python scripts/render_dijkstra_math_walkthrough.py
.venv/bin/python scripts/render_bitboard_pawn_shift.py
.venv/bin/python scripts/render_gridworld_rl_short.py
.venv/bin/python scripts/render_llm_next_token_short.py
.venv/bin/python scripts/render_pathfinding_distribution.py
.venv/bin/python scripts/render_bubble_sort_operation_cloud.py
.venv/bin/python scripts/render_bubble_sort_operation_cloud.py --all
.venv/bin/python scripts/render_sort_comparison_cloud.py
```

Useful options:

```bash
.venv/bin/python scripts/render_bubble_sort_short.py --seed 12 --bars 24
.venv/bin/python scripts/render_bubble_sort_short.py --width 720 --height 1280 --fps 30
.venv/bin/python scripts/render_bubble_sort_short.py --no-audio
.venv/bin/python scripts/render_merge_sort_short.py --seed 12 --bars 24
.venv/bin/python scripts/render_merge_sort_short.py --no-audio
.venv/bin/python scripts/render_elementary_sort_shorts.py --algorithm selection
.venv/bin/python scripts/render_elementary_sort_shorts.py --algorithm gnome --no-audio
.venv/bin/python scripts/render_operation_meter_short.py --algorithm bubble
.venv/bin/python scripts/render_operation_meter_short.py --bars 32 --seed 12
.venv/bin/python scripts/render_operation_meter_short.py --no-audio
.venv/bin/python scripts/render_connect_four_minimax_short.py --depth 5
.venv/bin/python scripts/render_pathfinding_short.py --algorithm bfs
.venv/bin/python scripts/render_pathfinding_short.py --no-audio
.venv/bin/python scripts/render_astar_math_walkthrough.py --expansions 12
.venv/bin/python scripts/render_greedy_best_first_math_walkthrough.py --expansions 12
.venv/bin/python scripts/render_dijkstra_math_walkthrough.py --expansions 12
.venv/bin/python scripts/render_pathfinding_distribution.py --algorithm bfs
.venv/bin/python scripts/render_pathfinding_distribution.py --metric edge-checks
.venv/bin/python scripts/render_bubble_sort_operation_cloud.py --algorithm merge
.venv/bin/python scripts/render_bubble_sort_operation_cloud.py --samples 700 --seed 19
.venv/bin/python scripts/render_bubble_sort_operation_cloud.py --max-n 500
.venv/bin/python scripts/render_bubble_sort_operation_cloud.py --y-scale log
.venv/bin/python scripts/render_bubble_sort_operation_cloud.py --sample-mode inversion-uniform
.venv/bin/python scripts/render_sort_comparison_cloud.py --seconds-per-algorithm 5 --samples 700
```

The renderer pipes raw frames directly to `ffmpeg`, so it does not leave thousands of frame images in the repo.
