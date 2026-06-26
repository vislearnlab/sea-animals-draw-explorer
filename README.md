# Sea Animals Explorer

An interactive web page for exploring children's and adults' **drawings** *and*
**descriptions** of six sea animals (crab, octopus, seahorse, shark, turtle,
whale), collected at the Birch Aquarium for the
[`sea-animals-draw`](https://github.com/vislearnlab/sea-animals-draw) study.
Built in the style of the
[`drawing-explorer`](https://github.com/vislearnlab/drawing-explorer) page.

- **602 drawings** (88 children + 20 adults)
- **538 descriptions** (79 children + 20 adults)
- **527** child × category items have *both* a drawing and a description.

The two modalities are shown **side by side** as linked t-SNEs: hover a point in
one panel and its cross-modal match (the same child's drawing/description of the
same animal) lights up in the other, with a paired card showing both together —
so you can explore how a child's drawing and description of the same concept
correspond.

**Drawing ↔ description agreement.** For the 527 child × category items that have
both modalities, the page scores how much CLIP's classification of the drawing
agrees with its classification of the description (cosine of the two 6-way
probability vectors, plus a same-top-guess flag). Color by it, use the **age
bins** (3–5 / 6–7 / 8+) to filter both panels at once, and read the visible
mean off the footer — **consistent with the prediction that cross-modal alignment
increases with age** (Pearson r ≈ +0.27 across 411 child pairs; bin means rise
0.67 → 0.76 → 0.79, and ~0.61 at age 3 → ~0.81 by age 11). A small **drawing-vs-
description scatter** in the right column plots the two modalities against each
other for whichever metric/age group is currently shown, with its own *r*.

## What it shows

Each panel lays items out by a 2-D **t-SNE of the raw CLIP embeddings** (OpenCLIP
**ViT-B-32**, `pretrained="openai"`) — so position reflects overall visual /
semantic similarity and the layout is **fixed regardless of which metric you
color by** (the classification numbers below are color layers on top of this
fixed map, not the map itself). Color, category / age filters, children-only,
CLIP-correct-only, and a "both modalities only" filter apply to both panels at
once.

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

The source repo stores CLIP **embeddings** and within-subject / to-adult
*similarities*, but **not** a category classification — so this repo computes one
(`clip_lib.py`, OpenCLIP **ViT-B-32**, `pretrained="openai"`). Each item is
classified against the six **adult prototypes** of its own modality — the mean
adult embedding per category — exactly the reference the study uses for
recognizability:

- **Drawings:** each PNG is content-cropped and encoded with the CLIP image
  encoder; cosine similarity to the six adult *drawing* prototypes → probabilities.
- **Descriptions:** each masked transcript is encoded with the CLIP text encoder
  (chunked and averaged past the 77-token limit); cosine similarity to the six
  adult *description* prototypes → probabilities.

Probabilities come from a self-calibrating (z-scored) softmax over the six
similarities, so they're graded rather than saturated. Adult items leave their
own embedding out of their category prototype.

Using adult prototypes instead of bare text labels (`"a crab"`) matters a lot for
descriptions: e.g. *"it has claws and it pinches people"* classifies as **crab**,
where label-anchored CLIP wrongly called it a shark. CLIP top-1 accuracy:
**49% on children's drawings, 67% on children's descriptions** (chance = 17%;
adults 78% / 80%).

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
