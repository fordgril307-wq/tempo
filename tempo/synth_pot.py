"""Synthetic top-down pot video with a physically consistent temperature curve.

Used for development and regression tests only. The real claim is checked on
phone footage of real pots (eval/real_clips.py). The renderer reproduces the
visual cues a camera actually gets from above:

  shimmer   refraction of the pot bottom as convection starts (warp of texture)
  bubbles   nucleation dots on the bottom, then rising/popping bubbles
  rolling   large-scale surface turbulence and specular glints
  steam     low-contrast haze drifting over the surface
  foam      white starch foam spreading from the rim after noodles go in
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .physics import BOIL_C, STAGE_TEMP, heat_params, temp_at, time_to


@dataclass
class PotScene:
    name: str = "pot"
    seconds: float = 300.0
    fps: int = 15
    size: tuple = (480, 360)
    volume_ml: float = 600.0
    burner_kw: float = 2.4
    eff: float = 0.45
    T0: float = 16.0
    stage_dT1: float = 0.0
    stage_dT2: float = 0.0
    add_at: Optional[float] = None         # seconds: noodles go in (pot keeps boiling)
    foam_after_add: Optional[float] = None  # foam surge starts this long after add
    heat_down_at: Optional[float] = None    # heat lowered (foam recedes)
    seed: int = 0
    noise: float = 3.0
    pot_offset: tuple = (0, 0)


def _smooth_noise(rng, h, w, cell):
    small = rng.standard_normal((max(2, h // cell), max(2, w // cell))).astype(np.float32)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)


def render(sc: PotScene, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(sc.seed)
    W, H = sc.size
    cx, cy = W // 2 + sc.pot_offset[0], H // 2 + sc.pot_offset[1]
    R_out, R_in = int(H * 0.44), int(H * 0.40)
    T_inf, tau = heat_params(sc.volume_ml, sc.burner_kw, sc.eff)
    t_s1 = time_to(STAGE_TEMP[1] + sc.stage_dT1, sc.T0, T_inf, tau)
    t_s2 = time_to(STAGE_TEMP[2] + sc.stage_dT2, sc.T0, T_inf, tau)
    t_boil = time_to(BOIL_C, sc.T0, T_inf, tau)

    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    rr = np.hypot(xx - cx, yy - cy)
    water = rr < R_in
    rim = (rr >= R_in) & (rr < R_out)

    # brushed stainless bottom texture (what refraction distorts)
    tex = 150 + 18 * _smooth_noise(rng, H, W, 6) + 6 * rng.standard_normal((H, W)).astype(np.float32)
    ang = np.arctan2(yy - cy, xx - cx)
    tex += 10 * np.sin(rr / 3.0 + 0.3 * np.sin(ang * 7))          # spun-metal rings
    tex -= 0.25 * np.clip(rr - R_in * 0.6, 0, None)                  # darker toward wall
    base = np.zeros((H, W, 3), np.float32)
    base[:] = (38, 38, 42)                                           # stove top
    base[rim] = (175, 175, 180)
    for c, k in enumerate((0.96, 0.98, 1.0)):
        base[..., c][water] = tex[water] * k

    sites = [(cx + r * np.cos(a), cy + r * np.sin(a)) for r, a in
             zip(np.sqrt(rng.uniform(0, 1, 140)) * R_in * 0.9, rng.uniform(0, 2 * np.pi, 140))]
    noodles = []
    for _ in range(26):
        x0, y0 = cx + rng.uniform(-0.45, 0.45) * R_in, cy + rng.uniform(-0.45, 0.45) * R_in
        noodles.append((x0, y0, rng.uniform(0, np.pi), rng.uniform(25, 60)))
    foam_seed = _smooth_noise(rng, H, W, 18)

    fields = [(_smooth_noise(rng, H, W, 24), _smooth_noise(rng, H, W, 24)) for _ in range(6)]
    big_fields = [(_smooth_noise(rng, H, W, 60), _smooth_noise(rng, H, W, 60)) for _ in range(6)]
    n = int(sc.seconds * sc.fps)
    vw = cv2.VideoWriter(str(out_dir / f"{sc.name}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), sc.fps, (W, H))
    bubbles: list[list] = []      # [x, y, r, life]
    haze_prev = None
    temps = []
    for f in range(n):
        t = f / sc.fps
        T = temp_at(t, sc.T0, T_inf, tau) if t < t_boil else BOIL_C
        temps.append(T)
        boiling = t >= t_boil
        # --- refraction warp (convection shimmer / turbulence)
        ph = (t * 0.9) % len(fields)
        i0, a = int(ph), ph - int(ph)
        fx = (1 - a) * fields[i0][0] + a * fields[(i0 + 1) % len(fields)][0]
        fy = (1 - a) * fields[i0][1] + a * fields[(i0 + 1) % len(fields)][1]
        s1 = float(np.clip((T - 55) / 40, 0, 1))
        amp = 0.25 + 2.2 * s1 + (7.0 if boiling else 0.0)
        mapx, mapy = xx + amp * fx, yy + amp * fy
        if boiling:
            ph2 = (t * 2.2) % len(big_fields)
            j0, b = int(ph2), ph2 - int(ph2)
            mapx += 6 * ((1 - b) * big_fields[j0][0] + b * big_fields[(j0 + 1) % 6][0])
            mapy += 6 * ((1 - b) * big_fields[j0][1] + b * big_fields[(j0 + 1) % 6][1])
        frame = cv2.remap(base, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        frame[~water] = base[~water]
        # --- noodles
        if sc.add_at is not None and t >= sc.add_at:
            for (x0, y0, th, L) in noodles:
                th2 = th + 0.3 * np.sin(t * 0.7 + x0)
                p1 = (int(x0 - L * np.cos(th2)), int(y0 - L * np.sin(th2)))
                p2 = (int(x0 + L * np.cos(th2)), int(y0 + L * np.sin(th2)))
                cv2.line(frame, p1, p2, (120, 200, 230), 3, cv2.LINE_AA)
        # --- bubbles
        if T >= STAGE_TEMP[1] + sc.stage_dT1:
            rate = 2 + 30 * float(np.clip((T - STAGE_TEMP[1]) / 25, 0, 1)) + (40 if boiling else 0)
            for _ in range(rng.poisson(rate / sc.fps)):
                x, y = sites[rng.integers(len(sites))]
                big = T >= STAGE_TEMP[2] + sc.stage_dT2
                r = rng.uniform(2.5, 6.5) if big else rng.uniform(0.8, 1.8)
                bubbles.append([x + rng.normal(0, 3), y + rng.normal(0, 3), r,
                                rng.uniform(0.3, 1.2) if big else rng.uniform(1.0, 3.0), big])
        nb = []
        for bb in bubbles:
            x, y, r, life, big = bb
            if big:
                bb[0] += rng.normal(0, 1.5); bb[1] += rng.normal(0, 1.5); bb[2] *= 1.03
            cv2.circle(frame, (int(bb[0]), int(bb[1])), max(1, int(round(bb[2]))), (235, 238, 240), -1, cv2.LINE_AA)
            cv2.circle(frame, (int(bb[0]), int(bb[1])), max(1, int(round(bb[2]))) + 1, (90, 90, 95), 1, cv2.LINE_AA)
            bb[3] -= 1 / sc.fps
            if bb[3] > 0:
                nb.append(bb)
        bubbles = nb
        if boiling:
            for _ in range(rng.poisson(25 / sc.fps * 4)):
                ang2, rad = rng.uniform(0, 2 * np.pi), np.sqrt(rng.uniform()) * R_in * 0.9
                cv2.ellipse(frame, (int(cx + rad * np.cos(ang2)), int(cy + rad * np.sin(ang2))),
                            (int(rng.uniform(4, 12)), int(rng.uniform(2, 5))), rng.uniform(0, 180), 0, 360,
                            (250, 250, 250), -1, cv2.LINE_AA)
        # --- foam
        if sc.add_at is not None and sc.foam_after_add is not None:
            ts = sc.add_at + sc.foam_after_add
            if t >= ts:
                cover = min(1.0, (t - ts) / 28.0)
                if sc.heat_down_at is not None and t >= sc.heat_down_at:
                    cover = max(0.0, min(1.0, (sc.heat_down_at - ts) / 28.0) - (t - sc.heat_down_at) / 12.0)
                if cover > 0:
                    edge = (rr / R_in) + 0.35 * foam_seed + 0.08 * np.sin(t * 3 + rr / 9)
                    m = water & (edge > 1.0 - 1.15 * cover)
                    frame[m] = 0.25 * frame[m] + 0.75 * np.array([238, 240, 242], np.float32)
        # --- steam haze
        h = float(np.clip((T - 82) / 18, 0, 1)) * (0.22 + 0.06 * np.sin(t * 1.3))
        if h > 0:
            if haze_prev is None or f % 15 == 0:
                haze_prev = np.clip(_smooth_noise(rng, H, W, 80), -1, 1)
            haze = 0.5 + 0.5 * haze_prev
            frame = frame * (1 - h * haze[..., None]) + 215 * h * haze[..., None]
        frame += rng.normal(0, sc.noise, frame.shape).astype(np.float32)
        vw.write(np.clip(frame, 0, 255).astype(np.uint8))
    vw.release()
    truth = {
        "scene": asdict(sc),
        "t_stage1": round(t_s1, 2), "t_stage2": round(t_s2, 2), "t_boil": round(t_boil, 2),
        "pot": {"cx": cx, "cy": cy, "r": R_in},
        "temps_1hz": [round(x, 2) for x in temps[::sc.fps]],
    }
    (out_dir / f"{sc.name}.json").write_text(json.dumps(truth))
    return truth
