"""Tempo web service: station screens, policy comparison, and a pot-vision endpoint.

    uvicorn app.main:app --port 8000
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
import threading
import time
import uuid
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response

from tempo import cloud
from tempo.kitchen import simulate
from tempo.physics import nominal_boil_s
from tempo.replay import build
from tempo.vision import watch_video

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("tempo.app")

ROOT = Path(__file__).resolve().parent.parent
CLIPS = ROOT / "samples"
MAX_UPLOAD_MB = int(os.getenv("TEMPO_MAX_UPLOAD_MB", "80"))
app = FastAPI(title="Tempo", version="1.0.0")
_jobs: dict[str, dict] = {}
_sem = threading.Semaphore(int(os.getenv("TEMPO_MAX_JOBS", "2")))


DEMO_SEED = 140        # a service where ramen, kimchi stew, egg soup and pasta all hit the burners early


def _demo_videos() -> list[dict]:
    """One camera clip per boiling dish (samples/demo_pot.json = ramen, samples/clip_*.json = the others)."""
    files = [CLIPS / "demo_pot.json"] + sorted(CLIPS.glob("clip_*.json"))
    return [json.loads(p.read_text()) for p in files if p.exists()]


@app.get("/", response_class=HTMLResponse)
def index():
    return (ROOT / "app" / "static" / "index.html").read_text(encoding="utf-8")


@app.get("/healthz")
def healthz():
    import cv2
    return {"ok": True, "opencv": cv2.__version__, "cloud": cloud.mode()}


@lru_cache(maxsize=16)
def _replay(seed: int, with_video: bool) -> bytes:
    v = _demo_videos() if with_video else None
    rep = build(seed=seed, n_tickets=6, mean_gap_s=240, video=v)
    return json.dumps(rep, separators=(",", ":"), default=float).encode()


@app.get("/api/replay")
def replay(seed: int = DEMO_SEED, video: bool = True):
    if not 0 <= seed < 10_000:
        raise HTTPException(422)
    return Response(_replay(seed, video), media_type="application/json")


@lru_cache(maxsize=64)
def _compare(seed: int) -> dict:
    out = {}
    for p in ("fifo", "plan_fixed", "tempo"):
        m, _ = simulate(seed, p, n_tickets=6, mean_gap_s=240)
        out[p] = {k: round(v, 1) if isinstance(v, float) else v for k, v in m.items()}
    return out


@app.get("/api/compare")
def compare(seed: int = DEMO_SEED):
    return _compare(seed)


@app.get("/api/eval")
def eval_results():
    out = {}
    for name in ("kitchen_results.json", "vision_results.json", "real_results.json"):
        p = ROOT / "eval" / name
        if p.exists():
            d = json.loads(p.read_text())
            out[name.split("_")[0]] = d.get("summary", d)
    return out


@app.get("/clips/{name}")
def clip(name: str):
    p = CLIPS / Path(name).name
    if not p.exists() or p.suffix not in {".mp4", ".webm", ".jpg"}:
        raise HTTPException(404)
    return FileResponse(p)


# ---------------------------------------------------------------- vision API
def _run_vision(job: str, path: Path, volume_ml: float, burner_kw: float):
    j = _jobs[job]
    with _sem:
        j["status"] = "running"
        t0 = time.time()
        try:
            w = watch_video(str(path), prior_s=nominal_boil_s(volume_ml, burner_kw))
            j["result"] = {"roi": w.roi, "events": w.events,
                           "series": [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()
                                       if k in ("t", "A_s", "stage", "foam", "flow", "micro")} |
                                      ({"eta": round(r["eta"]["t_boil"], 1), "pm": round(r["eta"]["pm"], 1),
                                        "basis": r["eta"]["basis"]} if "eta" in r else {})
                                      for r in w.series]}
            j["status"] = "done"
            cloud.put_json(f"vision/{job}.json", j["result"])
        except Exception as e:
            log.exception("vision job failed")
            j["status"], j["error"] = "error", str(e)
        finally:
            j["seconds"] = round(time.time() - t0, 1)
            shutil.rmtree(path.parent, ignore_errors=True)
            log.info(json.dumps({"event": "vision_job", "job": job, "status": j["status"], "s": j["seconds"]}))


@app.post("/api/vision")
async def vision(file: UploadFile = File(...), volume_ml: float = Form(550), burner_kw: float = Form(2.4)):
    if not (100 <= volume_ml <= 10000 and 0.5 <= burner_kw <= 10):
        raise HTTPException(422, "volume_ml or burner_kw out of range")
    ext = Path(file.filename or "").suffix.lower()
    if ext not in {".mp4", ".mov", ".webm", ".avi", ".mkv"}:
        raise HTTPException(415, "video file expected")
    d = Path(tempfile.mkdtemp(prefix="tempo_"))
    p, size = d / f"in{ext}", 0
    with p.open("wb") as fh:
        while chunk := await file.read(1 << 20):
            size += len(chunk)
            if size > MAX_UPLOAD_MB << 20:
                shutil.rmtree(d, ignore_errors=True)
                raise HTTPException(413, f"max {MAX_UPLOAD_MB} MB")
            fh.write(chunk)
    job = uuid.uuid4().hex[:10]
    _jobs[job] = {"id": job, "status": "queued"}
    threading.Thread(target=_run_vision, args=(job, p, volume_ml, burner_kw), daemon=True).start()
    return {"id": job}


@app.get("/api/vision/{job}")
def vision_job(job: str):
    j = _jobs.get(job)
    if not j:
        raise HTTPException(404)
    return JSONResponse(j)


# ---------------------------------------------------------------- live station feeds
# One shared service clock; every station screen / LED ring at its own spot in
# the kitchen asks "what should *I* show now?" and gets only its own slice.
STATIONS = ["prep", "B1", "B2", "B3", "B4", "pass", "line"]
LIVE = {"seed": DEMO_SEED, "speed": 10.0, "start": time.time()}
LED = {  # colour language shared by screens and LED rings
    "empty": ((40, 50, 55), "solid"), "loading": ((242, 165, 65), "solid"), "heating": ((242, 165, 65), "fill"),
    "boiling": ((228, 87, 46), "solid"), "cooking": ((46, 196, 182), "fill"), "done": ((138, 201, 38), "blink"),
}


def _frame_now() -> tuple[dict, dict]:
    rep = json.loads(_replay(LIVE["seed"], True))
    idx = int((time.time() - LIVE["start"]) * LIVE["speed"] / rep["meta"]["step"])
    idx = max(0, min(idx, len(rep["frames"]) - 1))
    return rep["frames"][idx], rep


@app.post("/api/live/start")
def live_start(seed: int = DEMO_SEED, speed: float = 10.0, at: float = 0.0):
    """Restart the shared service clock (at = jump to this service second)."""
    if not (0 <= seed < 10_000 and 0.5 <= speed <= 60 and 0 <= at < 4 * 3600):
        raise HTTPException(422)
    LIVE.update(seed=seed, speed=speed, start=time.time() - at / speed)
    return LIVE


@app.get("/api/live/{station}")
def live(station: str, format: str = "screen"):
    if station == "frame":
        return live_frame()
    if station not in STATIONS or station == "line":
        raise HTTPException(404, f"stations: {STATIONS}")
    F, rep = _frame_now()
    base = {"station": station, "t": F["t"], "speed": LIVE["speed"]}
    if station.startswith("B"):
        card = next(b for b in F["burners"] if b["name"] == station)
        if format == "led":   # tiny payload for an ESP32 + LED ring at the burner
            rgb, pattern = LED.get(card["phase"], LED["empty"])
            if card.get("level") == "warn":
                rgb, pattern = (255, 210, 63), "blink"
            elif card.get("blink"):
                pattern = "blink"
            return {"rgb": rgb, "pattern": pattern, "fill": card.get("progress", 0.0)}
        return base | {"card": card}
    if station == "prep":
        return base | {"now": F["prep_now"], "next": F["prep_next"]}
    return base | {"pass": F["pass"]}


@app.get("/station/{station}", response_class=HTMLResponse)
def station_page(station: str):
    if station not in STATIONS:
        raise HTTPException(404)
    page = "line.html" if station == "line" else "station.html"
    return (ROOT / "app" / "static" / page).read_text(encoding="utf-8")


@app.get("/api/live/frame")
def live_frame():
    """Whole-kitchen state for the dashboard, on the same clock as every station."""
    F, rep = _frame_now()
    ev = [e for e in rep["events"] if e["t"] <= F["t"] and e["kind"] in ("order", "served", "auto_heat_down", "replan", "stage")
          and (e["kind"] != "replan" or e["reason"].startswith("camera"))][-14:]
    return {"frame": F, "events": ev, "video": rep["meta"]["video"], "videos": rep["meta"]["videos"], "speed": LIVE["speed"]}


@app.get("/board", response_class=HTMLResponse)
def board():
    return (ROOT / "app" / "static" / "index.html").read_text(encoding="utf-8")


@app.get("/sim", response_class=HTMLResponse)
def sim():
    """Kitchen twin: the simulated kitchen and every station screen, all on the same clock."""
    return (ROOT / "app" / "static" / "sim.html").read_text(encoding="utf-8")


@app.get("/leds", response_class=HTMLResponse)
def leds():
    return (ROOT / "app" / "static" / "leds.html").read_text(encoding="utf-8")


@app.get("/api/qr")
def qr(path: str, request: Request):
    """QR code that opens a station screen on any phone/tablet on the same network."""
    if not (path.startswith("/station/") or path in ("/board", "/leds")):
        raise HTTPException(422)
    import io
    import qrcode
    img = qrcode.make(str(request.base_url).rstrip("/") + path, box_size=8, border=2)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return Response(buf.getvalue(), media_type="image/png")
