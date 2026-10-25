"""Sanity check of the perception features on short real (stock) clips.

No ground-truth timeline exists for stock clips, so this only asks: does the absolute
rolling-boil rule (the non-baseline part of stage 3) fire on boiling footage and stay
quiet on still water? ROI = one tap per camera at install time (given below), and the
automatic pot finder is reported separately.
"""
import json, sys
import cv2, numpy as np
from tempo.vision import PotWatcher, StageConfig, find_pot

ROI = {  # (cx, cy, a, b, angle) in 1280-wide coordinates — the install-time tap
    "p1": (520, 333, 253, 133, 0), "p2": (624, 384, 384, 208, 0), "p3": (752, 448, 368, 176, 0),
    "p4": (656, 384, 352, 176, 0), "p5": (672, 336, 320, 192, 0), "p6": (430, 300, 330, 180, 0),
    "p8": (640, 400, 520, 400, 0), "p9": (650, 360, 560, 330, 0), "p10": (640, 1300, 560, 700, 0)}
cfg = StageConfig()
out = {}
for name, roi in ROI.items():
    cap = cv2.VideoCapture(f"data/stock/{name}.mp4")
    fps = cap.get(cv2.CAP_PROP_FPS)
    s = 300 / roi[2]                       # normalise: pot semi-major axis = 300 px
    ok, f = cap.read()
    W = int(round(1280 * s)); H = int(round(f.shape[0] * 1280 / f.shape[1] * s))
    r = tuple(int(round(v * s)) for v in roi[:4]) + (roi[4],)
    w = PotWatcher(fps=fps, roi=r, cfg=cfg)
    auto = find_pot(cv2.resize(f, (1280, int(f.shape[0] * 1280 / f.shape[1]))))
    k = 0
    while ok:
        w.update(cv2.resize(f, (W, H)), k / fps)
        ok, f = cap.read(); k += 1
    rows = [{"t": x["t"], "flow": round(x["flow"], 2), "motion_area": round(x["motion_area"], 3),
             "foam": round(x["foam"], 3),
             "boil": bool(x["flow"] >= cfg.boil_flow_min and x["motion_area"] >= cfg.boil_motion_area)} for x in w.series]
    out[name] = {"auto_roi": auto, "rows": rows}
    print(name, " ".join(f"{x['t']:.0f}:{x['flow']:.1f}/{x['motion_area']:.2f}{'B' if x['boil'] else '.'}" for x in rows))
json.dump(out, open("eval/real_check.json", "w"), default=float, indent=1)
