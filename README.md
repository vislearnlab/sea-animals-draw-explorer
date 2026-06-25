# Sea Animals Explorer

An interactive web page for exploring children's and adults' **drawings** *and*
**descriptions** of six sea animals (crab, octopus, seahorse, shark, turtle,
whale), collected at the Birch Aquarium for the
[`sea-animals-draw`](https://github.com/vislearnlab/sea-animals-draw) study.
Built in the style of the
[`drawing-explorer`](https://github.com/vislearnlab/drawing-explorer) page.

- **602 drawings** (88 children + 20 adults)
- **538 descriptions** (79 children + 20 adults)

Toggle between the two modalities with the segmented control at the top.

## What it shows

Each item is laid out by a 2-D **t-SNE of its 6-way CLIP zero-shot probability
vector** — the same model the study's preprocessing uses (OpenCLIP **ViT-B-32**,
`pretrained="openai"`). Confident classifications pull out to the corners of the
simplex; confusable ones sit in the middle.

**Color by:**

- **CLIP (zero-shot)** — the layer this repo adds on top of the source data:
  - *CLIP recognizability* = probability of the intended category (default)
  - *CLIP correct / incorrect*, *CLIP log-odds*, *CLIP guess* (which category)
- **Similarity to adult prototype** (`draw_cossim_adults` /
  `utterance_full_cossim_adults`) — the study's recognizability measure. Adults
  have no score (they *define* the prototype) and are drawn faint.
- **Within-subject confusion** — how strongly the item resembles the *same
  participant's* other-category items.
- **Producer age** or **target category**.

**Filters:** target category (+ biological-group chips), age range, children-only,
CLIP-correct-only. **Hover** for the drawing (or the transcript, for
descriptions) + scores; **click** to pin. The footer updates with the visible
count, child/adult split, % CLIP-correct, and mean target probability.

## How the CLIP classification is computed

The source repo stores CLIP **image embeddings** and within-subject / to-adult
*similarities*, but **not** a zero-shot category classification — so this repo
computes it (`clip_lib.py`):

- **Drawings:** each PNG is content-cropped and encoded with CLIP, then scored
  against the six text anchors `"a drawing of a {category}"`; softmax over the
  cosine similarities (CLIP's logit scale) gives the 6-way probability vector.
- **Descriptions:** each masked transcript is encoded with the CLIP text encoder
  (chunked and averaged past the 77-token limit) and scored against `"a
  {category}"` anchors the same way.

CLIP top-1 accuracy: **48% on drawings, 44% on descriptions** (chance = 17%);
adults score higher than children in both, as expected.

## Run locally

```bash
python3 -m http.server 8000
# open http://127.0.0.1:8000/
```

(`index.html` is at the repo root; drawing PNGs live in `drawings/`.)

## Files

- `index.html` — the self-contained explorer (no dependencies, no build step).
- `points.json` — t-SNE layout + per-item CLIP / similarity scores for both modalities.
- `drawings/` — the 602 drawing PNGs (150×150, flattened onto white).
- `clip_lib.py` — loads OpenCLIP ViT-B-32 and does the image/text zero-shot scoring.
- `build_data.py` — regenerates `drawings/` + `points.json` (see below).

## Rebuilding the data

Needs `numpy`, `pillow`, `pymongo`, `scikit-learn`, and `open_clip_torch`
(`pip install open_clip_torch ftfy`). The drawing PNGs are pulled from the lab's
`kiddraw.birch_run_v1` MongoDB collection (PNGs already on disk are reused, so a
rebuild only hits Mongo for missing images); the descriptions come straight from
`sea-animals-draw/data/tidy/utterances.csv`.

```bash
# point at your sea-animals-draw checkout if it isn't a sibling dir
export SEA_TIDY_DIR=/path/to/sea-animals-draw/data/tidy
# Mongo connection string (do NOT commit this) — only needed for missing images
export SEA_MONGO_URI='mongodb://USER:PASS@HOST:27017/?authSource=admin'
python3 build_data.py
```

Credentials can instead go in a git-ignored `auth.txt` (first line = connection
string). Only clean items are included (`draw_exists`/`utterance_exists == 1` and
the matching `_interference == 0`). Where a participant didn't produce all six
categories, the missing within-subject confusion slots are simply absent (shown
as "drew/described only this category").

## Data & paper

> *What do drawing and speaking capture about how children express their
> knowledge of the world?* — children (ages 2–12) drew and described six sea
> animal categories at the Birch Aquarium.
> See [`sea-animals-draw`](https://github.com/vislearnlab/sea-animals-draw).
