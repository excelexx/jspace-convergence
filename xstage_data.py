"""Data acquisition for the text-vision experiment (section 4.2).

The 1,024 English WIT image-caption pairs everything cross-modal runs on
(image decodes, caption 20-200 chars), fixed once via streaming
shuffle(seed=0, buffer_size=2048); never resampled. Images are saved lossless
PNG under cache/images/eval/ for xstage1_vision.py; eval_manifest.json pins
the frozen set (caption, caption hash, image hash over raw RGB bytes, and the
source URL, since the images themselves are not shipped) and is read by
xstage1_text.py for the captions.
Idempotent: skips outputs that already exist."""
import hashlib
import json
import os

N_EVAL = 1024
CAP_MIN, CAP_MAX = 20, 200
SHUFFLE_SEED, SHUFFLE_BUFFER = 0, 2048

os.makedirs("cache/images/eval", exist_ok=True)


def sha(s):
    return hashlib.sha256(s).hexdigest()


def english_caption(row):
    """Best English caption per priority: reference desc > alt text."""
    wf = row["wit_features"]
    langs = wf["language"]
    for field in ["caption_reference_description", "caption_alt_text_description"]:
        for i, lang in enumerate(langs):
            if lang != "en":
                continue
            cap = wf[field][i]
            if cap:
                cap = " ".join(cap.split())
                if CAP_MIN <= len(cap) <= CAP_MAX:
                    return cap
    return None


def fetch_wit():
    if os.path.exists("eval_manifest.json"):
        print("eval_manifest.json exists, skipping (idempotent)")
        return

    from datasets import load_dataset

    ds = load_dataset("wikimedia/wit_base", split="train", streaming=True)
    ds = ds.shuffle(seed=SHUFFLE_SEED, buffer_size=SHUFFLE_BUFFER)

    eval_rows = []
    seen_cap_hashes, seen_img_hashes = set(), set()
    for row in ds:
        cap = english_caption(row)
        if cap is None:
            continue
        cap_h = sha(cap.encode("utf-8"))
        if cap_h in seen_cap_hashes:
            continue
        try:
            img = row["image"].convert("RGB")
        except Exception:
            continue                              # image must decode
        img_h = sha(img.tobytes() + f"{img.size}".encode())
        if img_h in seen_img_hashes:
            continue
        idx = len(eval_rows)
        img.save(f"cache/images/eval/{idx:04d}.png")
        eval_rows.append({"image_hash": img_h, "caption": cap,
                          "caption_hash": cap_h, "image_url": row["image_url"]})
        seen_cap_hashes.add(cap_h)
        seen_img_hashes.add(img_h)
        if idx % 128 == 0:
            print(f"  eval {idx}/{N_EVAL}", flush=True)
        if len(eval_rows) == N_EVAL:
            break
    assert len(eval_rows) == N_EVAL, f"only {len(eval_rows)} eval pairs found"

    with open("eval_manifest.json", "w", encoding="utf-8") as f:
        json.dump({"n": N_EVAL, "pairs": eval_rows}, f, indent=2)
    print("wrote eval_manifest.json")


if __name__ == "__main__":
    fetch_wit()
