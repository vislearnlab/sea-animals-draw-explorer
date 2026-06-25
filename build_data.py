#!/usr/bin/env python3
"""
Build drawings/ + points.json for the Sea Animals Drawing Explorer.

Pulls the finalImage PNGs for the six sea-animal categories out of the
`kiddraw.birch_run_v1` MongoDB collection, joins them to the per-drawing
metrics in sea-animals-draw/data/tidy/drawings.csv, and lays the drawings out
with a 2-D t-SNE of the within-subject confusion vectors.

Two measures drive the page (both from the tidy CSV / the paper):

  * recognizability  = `draw_cossim_adults`  — cosine similarity of the drawing
    to the *adult prototype* embedding for its category. Higher = more
    canonical / adult-like. This is the paper's headline developmental measure.
    Adults have no value here (they DEFINE the prototype) -> shown faint.

  * confusion vector = `draw_probability_{c}` — within-subject softmax over the
    cosine similarity of this drawing to the SAME participant's drawings of the
    other five categories. The target category's own slot is blank by
    construction (you can't confuse a drawing with itself), so the layout uses
    a 6-d vector with the target slot = 0 and the other five = these softmax
    probabilities. argmax over the five = "most confusable with".

Mongo credentials are read from env SEA_MONGO_URI, or auth.txt (first line =
full connection string). Nothing secret is committed.

    SEA_MONGO_URI='mongodb://user:pass@host:27017/?authSource=admin' python3 build_data.py

Output:  drawings/<category>_<participant>.png (150x150) + points.json
"""
import os, io, csv, json, base64

import numpy as np
from PIL import Image
from pymongo import MongoClient
from sklearn.manifold import TSNE

# ---------------------------------------------------------------- config
HERE = os.path.dirname(os.path.abspath(__file__))
TIDY_CSV = os.environ.get(
    "SEA_TIDY_CSV",
    os.path.join(HERE, "..", "sea-animals-draw", "data", "tidy", "drawings.csv"),
)
DB_NAME, COLL = "kiddraw", "birch_run_v1"
DRAW_DIR = os.path.join(HERE, "drawings")

CATEGORIES = ["crab", "octopus", "seahorse", "shark", "turtle", "whale"]
CAT_IDX = {c: i for i, c in enumerate(CATEGORIES)}
MONGO_CAT = {  # how each category is stored in Mongo (with article)
    "crab": "a crab", "octopus": "an octopus", "seahorse": "a seahorse",
    "shark": "a shark", "turtle": "a turtle", "whale": "a whale",
}
GROUPS = {  # quick-filter chips by rough biological grouping (CATEGORIES indices)
    "Invertebrates": [CAT_IDX["crab"], CAT_IDX["octopus"]],
    "Fish": [CAT_IDX["seahorse"], CAT_IDX["shark"]],
    "Reptile": [CAT_IDX["turtle"]],
    "Mammal": [CAT_IDX["whale"]],
}


def mongo_uri():
    if os.environ.get("SEA_MONGO_URI"):
        return os.environ["SEA_MONGO_URI"]
    auth = os.path.join(HERE, "auth.txt")
    if os.path.exists(auth):
        with open(auth) as f:
            return f.readline().strip()
    raise SystemExit("Set SEA_MONGO_URI or create auth.txt with the connection string.")


def fnum(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def main():
    os.makedirs(DRAW_DIR, exist_ok=True)

    # ---- load tidy metadata, keep clean drawings with a full confusion vec --
    rows = []
    with open(TIDY_CSV) as f:
        for r in csv.DictReader(f):
            if not (r["draw_exists"] == "1" and r["draw_interference"] == "0"
                    and r["target_category"] in CAT_IDX):
                continue
            off = [fnum(r[f"draw_probability_{c}"])
                   for c in CATEGORIES if c != r["target_category"]]
            if any(v is None for v in off):
                continue  # incomplete confusion vector -> can't lay out
            rows.append(r)
    print(f"{len(rows)} clean drawings with full confusion vectors")

    # ---- pull latest finalImage per (participant, category) from Mongo ----
    col = MongoClient(mongo_uri(), serverSelectionTimeoutMS=10000)[DB_NAME][COLL]
    inv = {v: k for k, v in MONGO_CAT.items()}
    best = {}
    for d in col.find(
        {"dataType": "finalImage",
         "category": {"$in": list(MONGO_CAT.values())},
         "participantID": {"$regex": "^(AD|BD)"}},
        {"participantID": 1, "category": 1, "imgData": 1, "endTrialTime": 1},
    ):
        key = (d["participantID"], inv[d["category"]])
        prev = best.get(key)
        if prev is None or (d.get("endTrialTime") or 0) >= (prev.get("endTrialTime") or 0):
            best[key] = d
    print(f"{len(best)} unique (participant, category) finalImages in Mongo")

    # ---- assemble points, save PNGs ---------------------------------------
    P = dict(file=[], cat=[], age=[], is_adult=[], recog=[], conf=[],
             conf_with=[], strokes=[], duration=[], intensity=[])
    layout = []  # 6-d confusion vectors for t-SNE
    missing = 0
    for r in rows:
        pid, cat = r["participant_id"], r["target_category"]
        doc = best.get((pid, cat))
        if doc is None:
            missing += 1
            continue
        raw = doc["imgData"]
        if raw.startswith("data:"):
            raw = raw.split(",", 1)[1]
        img = Image.open(io.BytesIO(base64.b64decode(raw))).convert("RGBA")
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))  # white paper
        bg.alpha_composite(img)
        fname = f"{cat}_{pid}.png"
        bg.convert("RGB").save(os.path.join(DRAW_DIR, fname), "PNG")

        vec = [0.0] * len(CATEGORIES)
        for c in CATEGORIES:
            if c != cat:
                vec[CAT_IDX[c]] = fnum(r[f"draw_probability_{c}"])
        layout.append(vec)
        # most-confusable-with = argmax over the off-target slots
        ci = max((i for i in range(len(CATEGORIES)) if i != CAT_IDX[cat]),
                 key=lambda i: vec[i])

        P["file"].append(fname)
        P["cat"].append(CAT_IDX[cat])
        P["age"].append(round(fnum(r["age_yrs"]) or 0, 1))
        P["is_adult"].append(1 if r["is_adult"] == "True" else 0)
        rc = fnum(r["draw_cossim_adults"])
        P["recog"].append(round(rc, 4) if rc is not None else None)
        P["conf"].append(round(vec[ci], 4))
        P["conf_with"].append(ci)
        P["strokes"].append(fnum(r["num_strokes"]))
        P["duration"].append(round(fnum(r["draw_duration"]) or 0, 1))
        P["intensity"].append(round(fnum(r["mean_intensity"]) or 0, 4))
    n = len(P["file"])
    print(f"{n} points assembled ({missing} skipped: no matching image)")

    # ---- t-SNE layout on the confusion vectors ----------------------------
    X = np.array(layout, float)
    Xz = (X - X.mean(0)) / (X.std(0) + 1e-9)
    perp = max(5, min(40, (n - 1) // 3))
    print(f"t-SNE on {n}x{X.shape[1]} (perplexity={perp}) ...")
    emb = TSNE(n_components=2, perplexity=perp, init="pca",
               learning_rate=200.0, random_state=0).fit_transform(Xz)
    lo, hi = emb.min(0), emb.max(0)
    emb = 40 + (emb - lo) / (hi - lo + 1e-9) * 920  # fit into a 40..960 world
    P["x"] = [round(float(v), 2) for v in emb[:, 0]]
    P["y"] = [round(float(v), 2) for v in emb[:, 1]]

    out = dict(draw_dir="drawings", categories=CATEGORIES, groups=GROUPS, n=n, **P)
    with open(os.path.join(HERE, "points.json"), "w") as f:
        json.dump(out, f)
    n_adult = sum(P["is_adult"])
    print(f"wrote points.json: {n} points ({n - n_adult} children, {n_adult} adults), "
          f"{len(os.listdir(DRAW_DIR))} PNGs in drawings/")


if __name__ == "__main__":
    main()
