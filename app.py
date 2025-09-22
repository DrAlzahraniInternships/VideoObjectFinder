import os, json, subprocess, uuid, shutil, logging
from pathlib import Path
from flask import Flask, request, render_template, redirect, url_for, send_from_directory

APP_DIR = Path(__file__).parent.resolve()
JOBS_DIR = (APP_DIR / "jobs").resolve()
PY_DIR = (APP_DIR / "python").resolve()
JOBS_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)

# ------------ utilities ------------
def run(cmd, cwd=None):
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"[RUN ERROR] {' '.join(cmd)}\n{p.stderr.strip()}")
    return p.stdout

def mmss(seconds: float) -> str:
    s = int(round(float(seconds)))
    return f"{s//60:02d}:{s%60:02d}"

# very small synonym helper to improve open-vocab matches
SYN = {
    "person": ["person", "human", "man", "woman", "people"],
    "glasses": ["glasses", "spectacles"],
    "spectacles": ["spectacles", "glasses"],
    "phone": ["phone", "cell phone", "mobile phone", "smartphone"],
    "laptop": ["laptop", "notebook", "macbook"],
    "money": ["money", "cash", "banknote", "dollar bill", "currency note"],
    "newspaper": ["newspaper", "paper", "magazine"],
    "white hair": ["white hair", "gray hair", "grey hair"],
    "red hair": ["red hair", "ginger hair", "auburn hair"],
    "code": ["code", "coding"],
}

# default processing knobs (safe on CPU)
FPS = "0.75"        # frame sampling when extracting
BOX_TH = "0.15"     # lower → more recall
BATCH = "1"         # keep 1 to avoid rare text batching issue
LONG_EDGE = "768"   # up-scale a bit for small objects
OCR_STEP = "2"      # analyze every Nth frame for OCR
OCR_EDGE = "800"    # OCR resize

# ------------ routes ------------
@app.get("/")
def index():
    return render_template("index.html")

@app.post("/upload")
def upload():
    f = request.files.get("video")
    if not f:
        return "No file uploaded", 400

    job_id = uuid.uuid4().hex[:20]
    job_dir = (JOBS_DIR / job_id)
    job_dir.mkdir(parents=True, exist_ok=True)

    tmp_path = job_dir / (f.filename or "upload.mp4")
    f.save(str(tmp_path))

    input_mp4 = job_dir / "input.mp4"
    # fast remux; if codec is exotic you could transcode
    run(["ffmpeg", "-y", "-i", str(tmp_path), "-c", "copy", str(input_mp4)])

    # duration gate (≤ 5 minutes)
    dur = run([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(input_mp4)
    ]).strip()
    try:
        duration = float(dur)
    except Exception:
        shutil.rmtree(job_dir, ignore_errors=True)
        return "Could not read duration", 400
    if duration > 300:
        shutil.rmtree(job_dir, ignore_errors=True)
        return "Cannot upload a video as it is more than 5 minutes", 413

    frames_dir = job_dir / "frames"
    run(["python", str(PY_DIR / "extract_frames.py"), str(input_mp4), str(frames_dir), FPS])
    return redirect(url_for("job_page", job_id=job_id))

@app.get("/job/<job_id>")
def job_page(job_id):
    job_dir = (JOBS_DIR / job_id)
    if not job_dir.exists():
        return "Unknown job", 404
    return render_template("search.html", job_id=job_id, video_url=url_for("video", job_id=job_id, filename="input.mp4"))

@app.get("/video/<job_id>/<path:filename>")
def video(job_id, filename):
    return send_from_directory(str(JOBS_DIR / job_id), filename, conditional=True)

@app.post("/search/<job_id>")
def search(job_id):
    job_dir = (JOBS_DIR / job_id)
    frames_dir = job_dir / "frames"
    if not frames_dir.exists():
        return "Results not ready", 404

    query = (request.form.get("q") or "").strip()
    q_lower = query.lower()
    expanded = SYN.get(q_lower, [query])
    text_for_model = ", ".join([t for t in expanded if t])

    # output jsons
    owl_json = job_dir / f"matches_owl_{q_lower.encode().hex()}.json"
    ocr_json = job_dir / f"matches_ocr_{q_lower.encode().hex()}.json"

    # --- OWL-ViT ---
    owl_times = []
    try:
        run([
            "python", str(PY_DIR / "detect_owlvit.py"),
            str(frames_dir), text_for_model, str(owl_json),
            "--fps", FPS, "--box_threshold", BOX_TH, "--batch", BATCH, "--long_edge", LONG_EDGE
        ])
        if owl_json.exists():
            raw = json.loads(owl_json.read_text(encoding="utf-8") or "[]")
            owl_times = [float(x["t"]) for x in raw]
    except Exception as e:
        app.logger.warning(f"OWL-ViT failed: {e}")

    # --- OCR (for visible text like captions, signs) ---
    ocr_times = []
    try:
        run([
            "python", str(PY_DIR / "ocr_search.py"),
            str(frames_dir), query, str(ocr_json),
            "--fps", FPS, "--step", OCR_STEP, "--long_edge", OCR_EDGE
        ])
        if ocr_json.exists():
            raw = json.loads(ocr_json.read_text(encoding="utf-8") or "[]")
            ocr_times = [float(x["t"]) for x in raw]
    except Exception as e:
        app.logger.warning(f"OCR failed: {e}")

    # merge, uniq, sort
    all_times = sorted({*owl_times, *ocr_times})
    items = [{"index": i+1, "t": t, "mmss": mmss(t)} for i, t in enumerate(all_times)]

    return render_template(
        "search.html",
        job_id=job_id,
        video_url=url_for("video", job_id=job_id, filename="input.mp4"),
        query=query,
        count=len(items),
        items=items
    )

if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "3000")), debug=False)
