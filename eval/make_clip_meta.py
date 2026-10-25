"""Run perception on a rendered dish clip and write the board metadata + web copies.

    python -m eval.make_clip_meta data/real_dishes/clip_pasta.mp4 pasta
"""
import json
import subprocess
import sys
from pathlib import Path

from tempo.physics import nominal_boil_s
from tempo.vision import watch_video

src, dish = Path(sys.argv[1]), sys.argv[2]
stem = sys.argv[3] if len(sys.argv) > 3 else src.stem
gt = json.loads(src.with_suffix(".json").read_text())
sc = gt["scene"]
w = watch_video(str(src), prior_s=nominal_boil_s(sc["volume_ml"], sc["burner_kw"]))
st = {e["stage"]: e["t"] for e in w.events if e["kind"] == "stage"}
warns = [e["t"] for e in w.events if e["kind"] == "foam_warning"]
meta = {"burner": "auto", "dish": dish, "clip": f"/clips/{stem}.mp4", "boil_s": gt["t_boil"],
        "boil_detect_s": float(st.get(3, gt["t_boil"])), "stages": {"1": st.get(1), "2": st.get(2)},
        "add_s": sc["add_at"], "foam_warn_s": warns[0] if warns else None, "heat_down_s": sc.get("heat_down_at"),
        "source": gt.get("footage", "simulated footage") + ", labelled in frame"}
print(dish, "truth boil", gt["t_boil"], "stages truth", gt["t_stage1"], gt["t_stage2"], "| perceived", st, "foam", warns[:2])
out = Path("samples")
(out / f"{stem}.json").write_text(json.dumps(meta, indent=1))
subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(src), "-vf", "scale=960:-2", "-c:v", "libx264", "-crf", "26",
                "-preset", "medium", "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(out / f"{stem}.mp4")], check=True)
subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(src), "-vf", "scale=960:-2", "-c:v", "libvpx-vp9", "-crf", "38",
                "-b:v", "0", "-row-mt", "1", "-deadline", "good", "-cpu-used", "4", "-an", str(out / f"{stem}.webm")], check=True)
