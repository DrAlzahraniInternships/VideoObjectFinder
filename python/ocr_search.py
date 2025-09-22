import argparse, json
from pathlib import Path
from PIL import Image
import numpy as np
import easyocr

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("frames_dir")
    ap.add_argument("query")
    ap.add_argument("out_json")
    ap.add_argument("--fps", default="0.75")
    ap.add_argument("--step", default="2")        # analyze every Nth frame
    ap.add_argument("--long_edge", default="800")
    args = ap.parse_args()

    tokens = [t.strip().lower() for t in args.query.split(",") if t.strip()]
    if not tokens:
        Path(args.out_json).write_text("[]", encoding="utf-8"); return

    fps = float(args.fps)
    step = int(args.step)
    L    = int(args.long_edge)

    reader = easyocr.Reader(['en'], gpu=False)

    paths = sorted(Path(args.frames_dir).glob("frame_*.jpg"))
    hits = []

    for idx, p in enumerate(paths):
        if idx % step != 0:
            continue
        im = Image.open(p).convert("RGB")
        w, h = im.size
        scale = L / max(w, h)
        if scale < 1.0:
            im = im.resize((int(w*scale), int(h*scale)))
        arr = np.array(im)           # EASYOCR expects numpy / path / url

        try:
            results = reader.readtext(arr)   # [(bbox, text, conf), ...]
        except Exception:
            continue

        found = False
        for (_, text, conf) in results:
            s = (text or "").lower()
            if any(tok in s for tok in tokens):
                found = True
                break
        if found:
            t = idx / fps
            hits.append({"t": float(t)})

    Path(args.out_json).write_text(json.dumps(hits, ensure_ascii=False), encoding="utf-8")
    print(f"OCR hits: {len(hits)}")

if __name__ == "__main__":
    main()
