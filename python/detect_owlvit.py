import argparse, json
from pathlib import Path
from PIL import Image
import numpy as np
import torch
from transformers import OwlViTProcessor, OwlViTForObjectDetection

def load_images(paths, long_edge=768):
    imgs = []
    for p in paths:
        im = Image.open(p).convert("RGB")
        w, h = im.size
        scale = long_edge / max(w, h)
        if scale < 1.0:
            im = im.resize((int(w*scale), int(h*scale)))
        imgs.append(im)
    return imgs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("frames_dir")
    ap.add_argument("query")
    ap.add_argument("out_json")
    ap.add_argument("--fps", default="0.75")
    ap.add_argument("--box_threshold", default="0.15")
    ap.add_argument("--batch", default="1")
    ap.add_argument("--long_edge", default="768")
    args = ap.parse_args()

    texts = [t.strip() for t in args.query.split(",") if t.strip()]
    if not texts:
        Path(args.out_json).write_text("[]", encoding="utf-8"); return

    fps = float(args.fps)
    th  = float(args.box_threshold)
    bs  = int(args.batch)
    L   = int(args.long_edge)

    device = "cpu"
    torch.set_grad_enabled(False)

    processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch16")
    model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch16")
    model.to(device).eval()

    frame_paths = sorted(Path(args.frames_dir).glob("frame_*.jpg"))
    out = []

    for i in range(0, len(frame_paths), bs):
        chunk = frame_paths[i:i+bs]
        images = load_images(chunk, long_edge=L)
        inputs = processor(text=texts, images=images, return_tensors="pt")
        with torch.inference_mode():
            outputs = model(**inputs)

        target_sizes = torch.tensor([im.size[::-1] for im in images])  # (h,w)
        processed = processor.post_process_object_detection(
            outputs=outputs, threshold=th, target_sizes=target_sizes
        )

        for j, det in enumerate(processed):
            if det["scores"].numel() > 0:
                frame_index = i + j
                t = frame_index / fps
                out.append({"t": float(t)})

    Path(args.out_json).write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(out)} hits")

if __name__ == "__main__":
    main()
