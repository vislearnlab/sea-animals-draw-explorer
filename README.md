# Sea Animals Drawing Explorer

An interactive web page for exploring **551 children's and adults' drawings** of
six sea animals (crab, octopus, seahorse, shark, turtle, whale), collected at the
Birch Aquarium for the [`sea-animals-draw`](https://github.com/vislearnlab/sea-animals-draw)
study. Built in the style of the
[`drawing-explorer`](https://github.com/vislearnlab/drawing-explorer) page.

## What it shows

- Drawings laid out by a 2-D **t-SNE** of each drawing's **within-subject
  confusion vector** — a softmax over how similar (in CLIP space) the drawing is
  to the *same participant's* drawings of the other five categories. Drawings
  that get confused with the same other category land near each other; targets
  also separate into rough regions.
- **Color by:**
  - **Similarity to adult prototype** (`draw_cossim_adults`) — the paper's
    recognizability measure (how adult-like / canonical the drawing is). Adults
    have no score (they *define* the prototype) and are drawn faint.
  - **Confusion strength** — how strongly the drawing resembles another category.
  - **Producer age** or **target category**.
- **Filters:** target category (+ biological-group quick chips), age range,
  minimum recognizability, children-only.
- **Hover** a point to see the drawing + its scores; **click** to pin it.
- The footer stat bar updates with the count / child–adult split / mean
  recognizability / mean confusion of the visible subset.

## Run locally

```bash
python3 -m http.server 8000
# open http://127.0.0.1:8000/
```

(`index.html` is at the repo root; drawing PNGs live in `drawings/`.)

## Files

- `index.html` — the self-contained explorer (no dependencies, no build step).
- `points.json` — t-SNE layout + per-drawing scores consumed by the page.
- `drawings/` — the 551 drawing PNGs (150×150, flattened onto white).
- `build_data.py` — regenerates `drawings/` + `points.json` (see below).

## Rebuilding the data

The drawing PNGs are **not** stored in the source `sea-animals-draw` repo — they
live in the lab's `kiddraw.birch_run_v1` MongoDB collection. `build_data.py`
pulls the `finalImage` PNGs, joins them to `sea-animals-draw/data/tidy/drawings.csv`,
and recomputes the t-SNE layout. It needs `numpy`, `pillow`, `pymongo`, and
`scikit-learn`.

```bash
# point at your sea-animals-draw checkout if it isn't a sibling dir
export SEA_TIDY_CSV=/path/to/sea-animals-draw/data/tidy/drawings.csv
# Mongo connection string (do NOT commit this)
export SEA_MONGO_URI='mongodb://USER:PASS@HOST:27017/?authSource=admin'
python3 build_data.py
```

Credentials can instead go in an `auth.txt` file (first line = connection
string); it is git-ignored.

Only clean drawings are included (`draw_exists == 1`, `draw_interference == 0`)
that also have a complete within-subject confusion vector — 431 children + 120
adults = 551 drawings.

## Data & paper

Drawings and metrics come from:

> *What do drawing and speaking capture about how children express their
> knowledge of the world?* — children (ages 2–12) drew and described six sea
> animal categories at the Birch Aquarium.
> See [`sea-animals-draw`](https://github.com/vislearnlab/sea-animals-draw).
