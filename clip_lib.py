"""Shared CLIP (OpenCLIP ViT-B-32, pretrained=openai) helpers — the exact model
the sea-animals-draw preprocessing pipeline uses. Loaded once, reused by
build_data.py for both image and text zero-shot classification."""
import os
# avoid the conda MKL / llvm-OpenMP double-runtime crash; must precede torch
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_THREADING_LAYER", "GNU")
import math
import numpy as np
import torch
import open_clip
from PIL import Image

torch.set_num_threads(1)

CATEGORIES = ["crab", "octopus", "seahorse", "shark", "turtle", "whale"]
# text anchors for zero-shot classification
DRAW_ANCHORS = [f"a drawing of {a}" for a in
                ["a crab", "an octopus", "a seahorse", "a shark", "a turtle", "a whale"]]
TEXT_ANCHORS = ["a crab", "an octopus", "a seahorse", "a shark", "a turtle", "a whale"]

_M = {}


def _load():
    if _M:
        return _M
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # the "-quickgelu" variant matches the OpenAI pretrained weights exactly
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32-quickgelu", pretrained="openai")
    model = model.to(device).eval()
    tok = open_clip.get_tokenizer("ViT-B-32")
    _M.update(model=model, preprocess=preprocess, tok=tok, device=device,
              logit_scale=float(model.logit_scale.exp().item()))
    return _M


def _encode_text(texts):
    m = _load()
    with torch.no_grad():
        t = m["tok"](texts).to(m["device"])
        f = m["model"].encode_text(t)
        f = f / f.norm(dim=-1, keepdim=True)
    return f


def anchor_feats(which):
    return _encode_text(DRAW_ANCHORS if which == "draw" else TEXT_ANCHORS)


def classify_image(path, anchors, content_crop=True):
    """Return a length-6 softmax probability vector for one drawing PNG."""
    m = _load()
    img = Image.open(path).convert("RGB")
    if content_crop:
        gray = np.asarray(img.convert("L"))
        ys, xs = np.where(gray < 250)  # non-white ink
        if len(xs) and len(ys):
            pad = 6
            l, r = max(xs.min() - pad, 0), min(xs.max() + pad, img.width)
            t, b = max(ys.min() - pad, 0), min(ys.max() + pad, img.height)
            img = img.crop((l, t, r, b))
    with torch.no_grad():
        x = m["preprocess"](img).unsqueeze(0).to(m["device"])
        f = m["model"].encode_image(x)
        f = f / f.norm(dim=-1, keepdim=True)
        logits = m["logit_scale"] * f @ anchors.T
        p = logits.softmax(dim=-1).cpu().numpy()[0]
    return p


def _chunk(text, tok, max_len=75):
    """Split into <=max_len-token chunks (CLIP's 77-token limit), evenly."""
    ids = tok([text])  # open_clip tokenizer pads to context; recover raw length
    # fall back to a simple word-based chunk: ~50 words ~ <75 tokens
    words = text.split()
    if len(words) <= 50:
        return [text]
    n = math.ceil(len(words) / 50)
    size = math.ceil(len(words) / n)
    return [" ".join(words[i:i + size]) for i in range(0, len(words), size)]


def classify_text(text, anchors):
    """Return a length-6 softmax probability vector for one utterance."""
    m = _load()
    chunks = _chunk(text, m["tok"])
    feats = _encode_text(chunks)              # (k, d), already L2-normalized
    f = feats.mean(dim=0, keepdim=True)
    f = f / f.norm(dim=-1, keepdim=True)
    with torch.no_grad():
        logits = m["logit_scale"] * f @ anchors.T
        p = logits.softmax(dim=-1).cpu().numpy()[0]
    return p
