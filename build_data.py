#!/usr/bin/env python3
"""
Build drawings/ + points.json for the Sea Animals Drawing & Description Explorer.

Two modalities, laid out side by side:

  * DRAWINGS    — the finalImage PNGs from kiddraw.birch_run_v1 (MongoDB),
                  joined to data/tidy/drawings.csv.
  * DESCRIPTIONS — the masked transcripts from data/tidy/utterances.csv
                  (no images; the card shows the text).

For each item we compute THREE families of scores:

  1. CLIP zero-shot classification  [NEW — not stored in the source repo]
     The lab's exact model (OpenCLIP ViT-B-32, pretrained=openai) classifies
     each drawing against "a drawing of a {cat}" and each description against
     "a {cat}". Gives a 6-way softmax -> clip_tp (target prob), clip_guess,
     clip_correct, clip_logodds. The 2-D t-SNE layout uses this CLIP vector.

  2. Recognizability = cosine similarity to the adult prototype
     (draw_cossim_adults / utterance_full_cossim_adults). Adults have none.

  3. Within-subject confusion (draw/utterance_probability_{cat}) — softmax over
     similarity to the SAME participant's other-category items.

Mongo creds: env SEA_MONGO_URI or auth.txt (git-ignored). Drawings already on
disk are reused, so a rebuild needs Mongo only for missing images.

    SEA_MONGO_URI='mongodb://user:pass@host:27017/?authSource=admin' python3 build_data.py
"""
import os, io, csv, json, base64

import numpy as np
from PIL import Image
from sklearn.manifold import TSNE

import clip_lib
from clip_lib import CATEGORIES

CAT_IDX = {c: i for i, c in enumerate(CATEGORIES)}
HERE = os.path.dirname(os.path.abspath(__file__))
TIDY_DIR = os.environ.get(
    "SEA_TIDY_DIR", os.path.join(HERE, "..", "sea-animals-draw", "data", "tidy"))
DRAW_CSV = os.path.join(TIDY_DIR, "drawings.csv")
TALK_CSV = os.path.join(TIDY_DIR, "utterances.csv")
DRAW_DIR = os.path.join(HERE, "drawings")
DB_NAME, COLL = "kiddraw", "birch_run_v1"
MONGO_CAT = {"crab": "a crab", "octopus": "an octopus", "seahorse": "a seahorse",
             "shark": "a shark", "turtle": "a turtle", "whale": "a whale"}
GROUPS = {"Invertebrates": [CAT_IDX["crab"], CAT_IDX["octopus"]],
          "Fish": [CAT_IDX["seahorse"], CAT_IDX["shark"]],
          "Reptile": [CAT_IDX["turtle"]], "Mammal": [CAT_IDX["whale"]]}


def fnum(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def clean_rows(path, exist_col, intf_col):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            if (r[exist_col] == "1" and r[intf_col] == "0"
                    and r["target_category"] in CAT_IDX):
                rows.append(r)
    return rows


def confusion(r, prefix):
    """6-d within-subject vector (target slot 0) + (conf, conf_with) or (None,None)."""
    vec = [0.0] * len(CATEGORIES)
    has = False
    cat = r["target_category"]
    for c in CATEGORIES:
        if c != cat:
            v = fnum(r[f"{prefix}_probability_{c}"])
            if v is not None:
                vec[CAT_IDX[c]] = v
                has = True
    if not has:
        return vec, None, None
    ci = max((i for i in range(len(CATEGORIES)) if i != CAT_IDX[cat]),
             key=lambda i: vec[i])
    return vec, round(vec[ci], 4), ci


def tsne_layout(vectors):
    X = np.asarray(vectors, float)
    Xz = (X - X.mean(0)) / (X.std(0) + 1e-9)
    n = len(X)
    perp = max(5, min(40, (n - 1) // 3))
    print(f"  t-SNE on {n}x{X.shape[1]} (perplexity={perp}) ...")
    emb = TSNE(n_components=2, perplexity=perp, init="pca",
               learning_rate=200.0, random_state=0).fit_transform(Xz)
    lo, hi = emb.min(0), emb.max(0)
    emb = 40 + (emb - lo) / (hi - lo + 1e-9) * 920
    return [round(float(v), 2) for v in emb[:, 0]], [round(float(v), 2) for v in emb[:, 1]]


def clip_scores(prob, cat_i):
    tp = float(prob[cat_i])
    gi = int(prob.argmax())
    return dict(clip_tp=round(tp, 4), clip_guess=gi,
                clip_correct=1 if gi == cat_i else 0,
                clip_logodds=round(float(np.log(tp / (1 - tp + 1e-9) + 1e-9)), 3),
                clip_max=round(float(prob.max()), 4))


def prototype_classify(embs, cats_arr, isad):
    """Classify each item against the six ADULT prototypes (mean adult embedding
    per category) of the same modality — the study-aligned approach. For adult
    items, leave-one-out their own embedding from their category prototype.
    Returns a list of 6-way (z-softmax) probability vectors."""
    E = np.asarray(embs)
    K, d = len(CATEGORIES), E.shape[1]
    psum = np.zeros((K, d)); pn = np.zeros(K)
    for c in range(K):
        sel = [i for i in range(len(E)) if isad[i] and cats_arr[i] == c]
        if sel:
            psum[c] = E[sel].sum(0); pn[c] = len(sel)
    out = []
    for i in range(len(E)):
        protos = np.empty((K, d))
        for c in range(K):
            s, n = psum[c].copy(), pn[c]
            if isad[i] and cats_arr[i] == c:          # leave-one-out for adults
                s, n = s - E[i], n - 1
            v = s / max(n, 1.0)
            protos[c] = v / (np.linalg.norm(v) + 1e-9)
        out.append(clip_lib.prototype_probs(E[i], protos))
    return out


# ---------------------------------------------------------------- drawings
def build_drawings():
    rows = clean_rows(DRAW_CSV, "draw_exists", "draw_interference")
    nk = len({r["participant_id"] for r in rows if r["is_child"] == "True"})
    print(f"DRAWINGS: {len(rows)} clean ({nk} children + adults)")
    os.makedirs(DRAW_DIR, exist_ok=True)
    coll = None  # lazy Mongo only if an image is missing

    kept, embs, skipped = [], [], 0
    for j, r in enumerate(rows):
        pid, cat = r["participant_id"], r["target_category"]
        path = os.path.join(DRAW_DIR, f"{cat}_{pid}.png")
        if not os.path.exists(path):
            if coll is None:
                coll = mongo_coll()
            doc = latest_image(coll, pid, cat)
            if doc is None:
                skipped += 1
                continue
            save_png(doc, path)
        embs.append(clip_lib.image_embedding(path)); kept.append(r)
        if (j + 1) % 100 == 0:
            print(f"    {j + 1}/{len(rows)} drawings encoded")
    cats_arr = [CAT_IDX[r["target_category"]] for r in kept]
    isad = [r["is_adult"] == "True" for r in kept]
    probs = prototype_classify(embs, cats_arr, isad)

    P = {k: [] for k in ("pid", "file", "cat", "age", "is_adult", "recog", "conf",
                         "conf_with", "strokes", "duration", "intensity", "clip_probs",
                         "clip_tp", "clip_guess", "clip_correct", "clip_logodds", "clip_max")}
    for r, prob in zip(kept, probs):
        pid, cat, ci = r["participant_id"], r["target_category"], CAT_IDX[r["target_category"]]
        P["clip_probs"].append([round(float(x), 4) for x in prob])
        vec, conf, conf_with = confusion(r, "draw")
        rc = fnum(r["draw_cossim_adults"])
        P["pid"].append(pid)
        P["file"].append(f"{cat}_{pid}.png")
        P["cat"].append(ci)
        P["age"].append(round(fnum(r["age_yrs"]) or 0, 1))
        P["is_adult"].append(1 if r["is_adult"] == "True" else 0)
        P["recog"].append(round(rc, 4) if rc is not None else None)
        P["conf"].append(conf)
        P["conf_with"].append(conf_with)
        P["strokes"].append(fnum(r["num_strokes"]))
        P["duration"].append(round(fnum(r["draw_duration"]) or 0, 1))
        P["intensity"].append(round(fnum(r["mean_intensity"]) or 0, 4))
        for k, v in clip_scores(prob, ci).items():
            P[k].append(v)
    P["x"], P["y"] = tsne_layout(embs)   # layout from the raw CLIP embeddings (stable)
    P["n"] = len(P["file"])
    print(f"  -> {P['n']} drawings ({skipped} skipped: no image)")
    return P


# ------------------------------------------------------------ descriptions
def build_descriptions():
    rows = clean_rows(TALK_CSV, "utterance_exists", "utterance_interference")
    nk = len({r["participant_id"] for r in rows if r["is_child"] == "True"})
    print(f"DESCRIPTIONS: {len(rows)} clean ({nk} children + adults)")
    kept, embs = [], []
    for j, r in enumerate(rows):
        text = (r["utterance_full_masked_filtered"] or "").strip()
        if not text:
            continue
        embs.append(clip_lib.text_embedding(text)); kept.append((r, text))
        if (j + 1) % 100 == 0:
            print(f"    {j + 1}/{len(rows)} descriptions encoded")
    cats_arr = [CAT_IDX[r["target_category"]] for r, _ in kept]
    isad = [r["is_adult"] == "True" for r, _ in kept]
    probs = prototype_classify(embs, cats_arr, isad)

    P = {k: [] for k in ("pid", "text", "cat", "age", "is_adult", "recog", "conf",
                         "conf_with", "nwords", "clip_probs",
                         "clip_tp", "clip_guess", "clip_correct", "clip_logodds", "clip_max")}
    for (r, text), prob in zip(kept, probs):
        ci = CAT_IDX[r["target_category"]]
        P["clip_probs"].append([round(float(x), 4) for x in prob])
        vec, conf, conf_with = confusion(r, "utterance")
        rc = fnum(r["utterance_full_cossim_adults"])
        P["pid"].append(r["participant_id"])
        P["text"].append(text)
        P["cat"].append(ci)
        P["age"].append(round(fnum(r["age_yrs"]) or 0, 1))
        P["is_adult"].append(1 if r["is_adult"] == "True" else 0)
        P["recog"].append(round(rc, 4) if rc is not None else None)
        P["conf"].append(conf)
        P["conf_with"].append(conf_with)
        P["nwords"].append(len(text.split()))
        for k, v in clip_scores(prob, ci).items():
            P[k].append(v)
    P["x"], P["y"] = tsne_layout(embs)   # layout from the raw CLIP embeddings (stable)
    P["n"] = len(P["text"])
    print(f"  -> {P['n']} descriptions")
    return P


# ------------------------------------------------------------------- mongo
def mongo_coll():
    from pymongo import MongoClient
    uri = os.environ.get("SEA_MONGO_URI")
    if not uri and os.path.exists(os.path.join(HERE, "auth.txt")):
        with open(os.path.join(HERE, "auth.txt")) as f:
            uri = f.readline().strip()
    if not uri:
        raise SystemExit("Need an image but no SEA_MONGO_URI / auth.txt set.")
    return MongoClient(uri, serverSelectionTimeoutMS=10000)[DB_NAME][COLL]


def latest_image(coll, pid, cat):
    inv = {v: k for k, v in MONGO_CAT.items()}
    best = None
    for d in coll.find({"dataType": "finalImage", "participantID": pid,
                        "category": MONGO_CAT[cat]},
                       {"imgData": 1, "endTrialTime": 1}):
        if best is None or (d.get("endTrialTime") or 0) >= (best.get("endTrialTime") or 0):
            best = d
    return best


def save_png(doc, path):
    raw = doc["imgData"]
    if raw.startswith("data:"):
        raw = raw.split(",", 1)[1]
    img = Image.open(io.BytesIO(base64.b64decode(raw))).convert("RGBA")
    bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
    bg.alpha_composite(img)
    bg.convert("RGB").save(path, "PNG")


def cross_modal_agreement(draw, talk):
    """Per (participant, category) pair: how much CLIP's classification of the
    drawing agrees with its classification of the description.
      agree     = cosine similarity of the two 6-way CLIP probability vectors
      agree_top = 1 if both have the same CLIP top-1 guess, else 0
    Stored (same value) on both modalities; None where the pair is missing."""
    def key_index(P):
        return {(P["pid"][i], P["cat"][i]): i for i in range(P["n"])}
    dk, tk = key_index(draw), key_index(talk)
    for P in (draw, talk):
        P["agree"] = [None] * P["n"]
        P["agree_top"] = [None] * P["n"]
    n_pair = 0
    for key, i in dk.items():
        j = tk.get(key)
        if j is None:
            continue
        a = np.asarray(draw["clip_probs"][i]); b = np.asarray(talk["clip_probs"][j])
        cos = float(a @ b / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9))
        top = 1 if draw["clip_guess"][i] == talk["clip_guess"][j] else 0
        draw["agree"][i] = talk["agree"][j] = round(cos, 4)
        draw["agree_top"][i] = talk["agree_top"][j] = top
        n_pair += 1
    agree_vals = [v for v in draw["agree"] if v is not None]
    top_vals = [v for v in draw["agree_top"] if v is not None]
    print(f"cross-modal: {n_pair} paired items · mean CLIP-prob agreement "
          f"{sum(agree_vals)/len(agree_vals):.2f} · same top-guess "
          f"{100*sum(top_vals)/len(top_vals):.0f}%")


def main():
    draw = build_drawings()
    talk = build_descriptions()
    cross_modal_agreement(draw, talk)
    out = dict(categories=CATEGORIES, groups=GROUPS, draw_dir="drawings",
               drawings=draw, descriptions=talk)
    with open(os.path.join(HERE, "points.json"), "w") as f:
        json.dump(out, f)
    print(f"wrote points.json: {draw['n']} drawings + {talk['n']} descriptions")


if __name__ == "__main__":
    main()
