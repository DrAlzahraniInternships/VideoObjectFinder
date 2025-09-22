import sys, subprocess
from pathlib import Path

video_path, out_dir, fps = sys.argv[1], sys.argv[2], sys.argv[3]
out = Path(out_dir)
out.mkdir(parents=True, exist_ok=True)

subprocess.run([
    "ffmpeg", "-y", "-i", video_path,
    "-vf", f"fps={fps},scale=-2:480",
    str(out / "frame_%06d.jpg")
], check=True)

print("OK")
