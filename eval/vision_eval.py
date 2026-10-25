"""Pot perception + boil prediction on held-out clips (leave-one-out calibration).

    python -m eval.vision_eval data/heldout [--h264]

1. Perception runs once per clip: pot localisation, stage onsets, boil
   detection, foam warnings.
2. Stage temperatures are calibrated leave-one-out: for each test clip, the
   fraction of heat-up time at which each stage appears is learned from the
   *other* clips only (this is all a real kitchen needs: a few clips that run
   to a boil).
3. Time-to-boil error is measured 120 / 60 / 30 s before the true boil and
   compared with the recipe-card / nameplate estimate.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from tempo.physics import BOIL_C, TAP_C, BoilEstimator, nominal_boil_s
from tempo.vision import watch_video


def h264(src: Path) -> Path:
    dst = src.with_name(src.stem + "_h264.mp4")
    if not dst.exists():
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(src), "-c:v", "libx264", "-crf", "28",
                        "-preset", "veryfast", "-pix_fmt", "yuv420p", str(dst)], check=True)
    return dst


def perceive(folder: Path, use_h264: bool) -> list[dict]:
    cache = folder / f"_perception_{'h264' if use_h264 else 'raw'}.json"
    if "--reuse" in sys.argv and cache.exists():
        return [r | {"stages": {int(k): v for k, v in r["stages"].items()}} for r in json.loads(cache.read_text())]
    rows = []
    for js in sorted(j for j in folder.glob("*.json") if not j.name.startswith("_")):
        gt = json.loads(js.read_text())
        sc = gt["scene"]
        video = js.with_suffix(".mp4")
        if use_h264:
            video = h264(video)
        prior = nominal_boil_s(sc["volume_ml"], sc["burner_kw"])
        w = watch_video(str(video), prior_s=prior)
        st = {e["stage"]: e["t"] for e in w.events if e["kind"] == "stage"}
        foam_full = (sc["add_at"] + sc["foam_after_add"] + 0.85 * 28) if sc.get("foam_after_add") else None
        warns = [e["t"] for e in w.events if e["kind"] == "foam_warning"]
        rows.append({"clip": js.stem, "t_boil": gt["t_boil"], "prior_s": prior, "stages": st,
                     "pot_err_px": float(np.hypot(w.roi[0] - gt["pot"]["cx"], w.roi[1] - gt["pot"]["cy"])),
                     "foam_full": foam_full, "foam_warns": warns,
                     "t_end": w.series[-1]["t"] if w.series else 0})
        print(js.stem, "boil", gt["t_boil"], "stages", st, "foam", foam_full, warns, flush=True)
    cache.write_text(json.dumps(rows))
    return rows


def eta_at(row: dict, now: float, stage_temp: dict) -> float:
    est = BoilEstimator(0.0, row["prior_s"], curvature=1.0)
    est.stage_temp.update(stage_temp)
    for s in (1, 2, 3):
        if s in row["stages"] and row["stages"][s] <= now:
            est.observe(s, row["stages"][s])
    return est.eta(now)["t_boil"]


def calibrate(train: list[dict]) -> dict:
    out = {}
    for s in (1, 2):
        fr = [r["stages"][s] / r["t_boil"] for r in train if s in r["stages"]]
        if fr:
            out[s] = TAP_C + float(np.median(fr)) * (BOIL_C - TAP_C)
    return out


def score(rows: list[dict]) -> dict:
    res = {"per_clip": []}
    errs = {h: {"vision": [], "nameplate": []} for h in (120, 60, 30)}
    for k, r in enumerate(rows):
        cal = calibrate(rows[:k] + rows[k + 1:])
        rec = {"clip": r["clip"], "calibrated_stage_temp": {s: round(v, 1) for s, v in cal.items()}}
        for h in (120, 60, 30):
            now = r["t_boil"] - h
            if now < 10:
                continue
            ev, ep = eta_at(r, now, cal) - r["t_boil"], r["prior_s"] - r["t_boil"]
            errs[h]["vision"].append(abs(ev)); errs[h]["nameplate"].append(abs(ep))
            rec[f"err_{h}s"] = {"vision": round(ev, 1), "nameplate": round(ep, 1)}
        d = r["stages"].get(3)
        rec["boil_detect_delay_s"] = None if d is None else round(d - r["t_boil"], 1)
        rec["pot_err_px"] = round(r["pot_err_px"], 1)
        if r["foam_full"]:
            early = [w for w in r["foam_warns"] if w <= r["foam_full"]]
            rec["foam_lead_s"] = round(r["foam_full"] - early[0], 1) if early else None
        else:
            rec["foam_false_alarms"] = len(r["foam_warns"])
        res["per_clip"].append(rec)
    delays = [c["boil_detect_delay_s"] for c in res["per_clip"] if c["boil_detect_delay_s"] is not None]
    leads = [c["foam_lead_s"] for c in res["per_clip"] if c.get("foam_lead_s") is not None]
    res["summary"] = {
        "clips": len(rows),
        **{f"mae_{h}s_before_boil": {k: round(float(np.mean(v)), 1) for k, v in errs[h].items()} | {"n": len(errs[h]["vision"])}
           for h in (120, 60, 30)},
        "boil_detected": f"{len(delays)}/{len(rows)}",
        "boil_detect_delay_s": {"mean": round(float(np.mean(delays)), 1), "max_abs": round(float(np.max(np.abs(delays))), 1)} if delays else None,
        "foam_warned_in_time": f"{len(leads)}/{sum(1 for r in rows if r['foam_full'])}",
        "foam_mean_lead_s": round(float(np.mean(leads)), 1) if leads else None,
        "foam_false_alarms": sum(c.get("foam_false_alarms", 0) for c in res["per_clip"]),
        "pot_center_err_px_max": round(max(c["pot_err_px"] for c in res["per_clip"]), 1),
    }
    return res


if __name__ == "__main__":
    folder = Path(sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else "data/heldout")
    out = {}
    label = "realistic" if "--realistic" in sys.argv else None
    for mode in (["raw", "h264"] if "--both" in sys.argv else (["h264"] if "--h264" in sys.argv else ["raw"])):
        rows = perceive(folder, mode == "h264")
        mode = label or mode
        out[mode] = score(rows)
        print(mode, json.dumps(out[mode]["summary"], indent=1))
    res_path = Path("eval/vision_results.json")
    prev = json.loads(res_path.read_text()) if res_path.exists() else {"summary": {}, "detail": {}}
    for m, v in out.items():
        prev["summary"][m], prev["detail"][m] = v["summary"], v
    res_path.write_text(json.dumps(prev, indent=1))
