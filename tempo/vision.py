"""OpenCV 5 perception for pots seen from above.

PotWatcher turns a video of one burner into:
  - per-second surface features (micro-motion, bubble births, optical-flow
    turbulence, foam coverage),
  - monotone stage events: shimmer (1), simmer (2), rolling boil (3),
  - a foam-surge forecast (seconds until the surface is covered),
which the BoilEstimator (physics.py) converts into a time-to-boil prediction.

All thresholds are relative to the pot's own first seconds (cold, still
water), so the same code works across pots, lighting and cameras.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from .physics import BoilEstimator


def find_pot(frame: np.ndarray) -> Optional[tuple]:
    """Locate the pot opening.

    Straight down: a Hough circle, returned as (cx, cy, r).
    At an angle (a phone on a stand): an ellipse fitted to edge contours,
    returned as (cx, cy, ax, ay, angle_deg). The inner region is shrunk so
    the mask stays on the liquid, not the wall or rim.
    """
    g = cv2.medianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 5)
    H, W = g.shape
    edges = cv2.Canny(cv2.GaussianBlur(g, (5, 5), 0), 40, 110)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8))
    cnts, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    best, best_score = None, -1.0
    for c in cnts:
        if len(c) < 120:
            continue
        (ex, ey), (a1, a2), ang = cv2.fitEllipse(c)
        lo, hi = min(a1, a2), max(a1, a2)
        if not (0.25 * min(H, W) < lo and hi < 1.1 * max(H, W) and lo / hi > 0.5):
            continue
        if not (0.15 * W < ex < 0.85 * W and 0.15 * H < ey < 0.85 * H):
            continue
        m = np.zeros_like(g)
        cv2.ellipse(m, ((ex, ey), (a1 * 0.8, a2 * 0.8), ang), 255, -1)
        inside = float(cv2.mean(g, mask=m)[0])
        score = inside * (lo * hi) ** 0.25          # bright interior (liquid / pot), reasonably large
        if score > best_score:
            best, best_score = (ex, ey, a1, a2, ang), score
    if best is not None:
        ex, ey, a1, a2, ang = best
        return int(ex), int(ey), int(a1 / 2 * 0.74), int(a2 / 2 * 0.74), float(ang)
    circles = cv2.HoughCircles(g, cv2.HOUGH_GRADIENT, dp=1.5, minDist=H // 2, param1=110, param2=45,
                               minRadius=int(min(H, W) * 0.18), maxRadius=int(min(H, W) * 0.55))
    if circles is None:
        return None
    cx, cy, r = max(np.round(circles[0]).astype(int), key=lambda c: c[2])
    return int(cx), int(cy), int(r * 0.88)


def roi_mask(shape, roi) -> np.ndarray:
    m = np.zeros(shape[:2], np.uint8)
    if len(roi) == 3:
        cv2.circle(m, (roi[0], roi[1]), roi[2], 255, -1)
    else:
        cv2.ellipse(m, (roi[0], roi[1]), (roi[2], roi[3]), roi[4], 0, 360, 255, -1)
    return m


@dataclass
class StageConfig:
    """Thresholds on the activity index A (log-ratios vs. the cold baseline).

    A = log2(micro/micro0) + log2(flow/flow0) + log2((births+5)/(births0+5))
    Levels were set on synthetic footage and must be re-calibrated on real
    clips (eval/calibrate.py) - only clips that run to a boil are needed.
    """
    baseline_s: float = 8.0
    persist_s: int = 3
    smooth_s: int = 5
    a_stage1: float = 3.2           # ~72 C: shimmer + first bubbles
    a_stage2: float = 7.0           # ~90 C: bubbles rising, simmer
    boil_flow_ratio: float = 20.0   # rolling boil: surface turbulence
    boil_motion_area: float = 0.04
    boil_flow_min: float = 1.5      # px, absolute floor after removing camera motion
    foam_white_s: int = 60
    foam_white_v: int = 205
    stage_temp: Optional[dict] = None   # from calibration (eval/calibrate.py); None = physics defaults
    curvature: Optional[float] = None


def load_config(path) -> StageConfig:
    import json
    d = json.loads(open(path).read())
    if "stage_temp" in d:
        d["stage_temp"] = {int(k): v for k, v in d["stage_temp"].items()}
    return StageConfig(**d)


@dataclass
class PotWatcher:
    fps: float
    roi: tuple
    cfg: StageConfig = field(default_factory=StageConfig)
    t_start: float = 0.0
    prior_s: float = 240.0
    curvature: float = 1.0

    def __post_init__(self):
        self.est = BoilEstimator(self.t_start, self.prior_s,
                                 self.cfg.curvature if self.cfg.curvature is not None else self.curvature)
        if self.cfg.stage_temp:
            self.est.stage_temp.update(self.cfg.stage_temp)
        self._prev_g = self._prev_hp = self._prev_small = None
        self._v_ref = None
        self._mask = None
        self._acc: dict[str, list] = {}
        self._sec = -1
        self.series: list[dict] = []
        self.base: dict = {}
        self.stage = 0
        self._above = {1: 0, 2: 0, 3: 0}
        self.events: list[dict] = []
        self.foam_hist: list[tuple[float, float]] = []

    # ---------------------------------------------------------------- frames
    def update(self, frame: np.ndarray, t: float) -> list[dict]:
        H, W = frame.shape[:2]
        if self._mask is None:
            self._mask = roi_mask(frame.shape, self.roi)
            self._mask_n = int(np.count_nonzero(self._mask))
            self._small_mask = cv2.resize(self._mask, (W // 2, H // 2), interpolation=cv2.INTER_NEAREST) > 0
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        hp = g - cv2.GaussianBlur(g, (0, 0), 3)
        m = self._mask > 0
        feats = {}
        if self._prev_g is not None:
            d = np.abs(g - self._prev_g)
            feats["micro"] = float(np.mean(np.abs(hp - self._prev_hp)[m]))
            feats["motion_area"] = float(np.count_nonzero((d > 18) & m)) / self._mask_n
            # bubble births: compact bright spots that were not there in the previous frame
            spots = ((hp > 22) & m).astype(np.uint8)
            new = spots & (self._prev_spots == 0)
            n, _, st, _ = cv2.connectedComponentsWithStats(new, connectivity=8)
            feats["births"] = float(sum(1 for k in range(1, n) if 2 <= st[k, cv2.CC_STAT_AREA] <= 120))
        self._prev_spots = ((hp > 22) & m).astype(np.uint8)
        # turbulence: dense optical flow on a half-size frame, ~5 Hz
        small = cv2.resize(g, (W // 2, H // 2)).astype(np.uint8)
        if self._prev_small is not None and int(t * self.fps) % max(1, int(self.fps // 5)) == 0:
            flow = cv2.calcOpticalFlowFarneback(self._prev_small, small, None, 0.5, 2, 13, 2, 5, 1.1, 0)
            v = flow[self._small_mask]
            v = v - np.median(v, axis=0)          # remove camera shake / global motion
            mag = np.linalg.norm(v, axis=1)
            feats["flow"] = float(np.median(mag)) * 2
            self._prev_small = small
        elif self._prev_small is None:
            self._prev_small = small
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        if self._v_ref is None:   # foam = clearly brighter than this pot's own liquid at the start
            self._v_ref = float(np.percentile(hsv[..., 2][m], 90))
        v_thr = min(self.cfg.foam_white_v, self._v_ref + 25)
        white = ((hsv[..., 1] < self.cfg.foam_white_s) & (hsv[..., 2] > v_thr) & m).astype(np.uint8)
        white = cv2.morphologyEx(white, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
        feats["foam"] = float(np.count_nonzero(white)) / self._mask_n
        self._prev_g, self._prev_hp = g, hp
        for k, v in feats.items():
            self._acc.setdefault(k, []).append(v)
        sec = int(t)
        out = []
        if sec != self._sec and self._sec >= 0:
            out = self._close_second(self._sec + 1.0)
        self._sec = sec
        return out

    # ------------------------------------------------------------- seconds
    def _close_second(self, t: float) -> list[dict]:
        a = self._acc
        row = {"t": t,
               "micro": float(np.mean(a.get("micro", [0]))),
               "motion_area": float(np.mean(a.get("motion_area", [0]))),
               "births": float(np.sum(a.get("births", [0]))),
               "flow": float(np.mean(a.get("flow", [0]))) if a.get("flow") else (self.series[-1]["flow"] if self.series else 0.0),
               "foam": float(np.mean(a.get("foam", [0])))}
        self._acc = {}
        self.series.append(row)
        events = []
        rel = t - self.t_start
        if rel <= self.cfg.baseline_s:
            self.base = {k: float(np.median([r[k] for r in self.series])) for k in ("micro", "births", "flow", "motion_area", "foam")}
            row["stage"] = 0
            return events
        c, b = self.cfg, self.base
        row["A"] = (np.log2(max(row["micro"], 1e-3) / max(b["micro"], 1e-3))
                    + np.log2(max(row["flow"], 1e-3) / max(b["flow"], 1e-3))
                    + np.log2((row["births"] + 5) / (b["births"] + 5)))
        A = float(np.median([r.get("A", row["A"]) for r in self.series[-c.smooth_s:]]))
        row["A_s"] = A
        cond = {
            1: A >= c.a_stage1,
            2: A >= c.a_stage2,
            3: row["flow"] >= max(c.boil_flow_ratio * b["flow"], c.boil_flow_min) and row["motion_area"] >= c.boil_motion_area,
        }
        for s in (1, 2, 3):
            self._above[s] = self._above[s] + 1 if cond[s] else 0
        for s in (3, 2, 1):
            if s > self.stage and self._above[s] >= c.persist_s:
                onset = t - c.persist_s + 1
                for k in range(self.stage + 1, s + 1):
                    self.est.observe(k, onset)
                    events.append({"t": round(onset, 1), "kind": "stage", "stage": k,
                                   "eta": round(self.est.eta(t)["t_boil"], 1)})
                self.stage = s
                break
        row["stage"] = self.stage
        row["eta"] = self.est.eta(t)
        # foam surge forecast
        self.foam_hist.append((t, max(0.0, row["foam"] - b.get("foam", 0.0))))
        fh = self.foam_hist[-5:]
        if len(fh) >= 4 and fh[-1][1] > 0.04:
            ts, fs = np.array(fh).T
            rate = np.polyfit(ts, fs, 1)[0]
            if rate > 0.004:
                ttf = (0.85 - fs[-1]) / rate
                row["foam_ttf"] = float(ttf)
                if ttf < 25 and not any(e["kind"] == "foam_warning" for e in self.events[-3:]):
                    events.append({"t": t, "kind": "foam_warning", "seconds_to_overflow": round(float(ttf), 1),
                                   "coverage": round(fs[-1], 3)})
        self.events += events
        return events


def watch_video(path: str, roi: Optional[tuple] = None, prior_s: float = 240.0,
                cfg: Optional[StageConfig] = None, max_s: Optional[float] = None) -> PotWatcher:
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    ok, first = cap.read()
    if not ok:
        raise FileNotFoundError(path)
    roi = roi or find_pot(first)
    if roi is None:
        raise ValueError("pot not found; pass roi=(cx, cy, r)")
    w = PotWatcher(fps=fps, roi=roi, cfg=cfg or StageConfig(), prior_s=prior_s)
    f = 0
    frame = first
    while ok:
        t = f / fps
        if max_s and t > max_s:
            break
        w.update(frame, t)
        ok, frame = cap.read()
        f += 1
    cap.release()
    return w
