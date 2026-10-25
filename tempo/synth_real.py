"""Photo-realistic synthetic pot footage (clearly labelled as simulated).

Compared with synth_pot.py this renders what a phone on a stand actually
sees: an oblique view of a stainless pot on a black glass cooktop, brushed
metal and specular highlights, refraction of the pot bottom, bubbles that
nucleate, rise and pop with ripples, rolling-boil froth, drifting steam, a
ramen block that loosens, starch foam, lighting flicker, camera jitter,
vignetting, sensor noise and H.264 compression. Ground truth (temperature
curve, stage onsets, boil time, foam timing, pot ellipse) is written next to
the video.
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .physics import BOIL_C, STAGE_TEMP, heat_params, temp_at, time_to

S = 1600                 # top-down canvas
OUT_W, OUT_H = 1280, 720
CX = CY = S // 2


@dataclass
class RealScene:
    name: str = "real_pot"
    seconds: float = 360.0
    fps: int = 15
    volume_ml: float = 600.0
    burner_kw: float = 2.4
    eff: float = 0.45
    T0: float = 16.0
    stage_dT1: float = 0.0
    stage_dT2: float = 0.0
    add_at: Optional[float] = None
    foam_after_add: Optional[float] = None
    heat_down_at: Optional[float] = None
    seed: int = 0
    pot_r: int = 300               # pot rim outer radius on the canvas
    tilt: float = 0.30             # 0 = straight down, larger = more oblique
    light_angle: float = -2.3      # where the kitchen light sits (radians)
    warm: float = 1.0              # white balance warmth
    label: bool = True             # burn "Simulated footage" into the frame
    crf: int = 23
    liquid: str = "clear"          # clear water | broth (pale) | red (kimchi stew)
    addin: str = "ramen"           # what goes in at add_at: ramen | pasta | egg | tofu
    pot: str = "steel"             # steel | black (earthenware-style stew pot)


def _noise(rng, h, w, cell):
    small = rng.standard_normal((max(2, h // cell), max(2, w // cell))).astype(np.float32)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)


def _homography(tilt: float) -> np.ndarray:
    top_w, bot_w = 1680 - tilt * 900, 2400 + tilt * 400
    src = np.float32([[0, 0], [S, 0], [S, S], [0, S]])
    dst = np.float32([[640 - top_w / 2, -230 + tilt * 300], [640 + top_w / 2, -230 + tilt * 300],
                      [640 + bot_w / 2, 1250], [640 - bot_w / 2, 1250]])
    return cv2.getPerspectiveTransform(src, dst)


def render(sc: RealScene, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(sc.seed)
    T_inf, tau = heat_params(sc.volume_ml, sc.burner_kw, sc.eff)
    t_s1 = time_to(STAGE_TEMP[1] + sc.stage_dT1, sc.T0, T_inf, tau)
    t_s2 = time_to(STAGE_TEMP[2] + sc.stage_dT2, sc.T0, T_inf, tau)
    t_boil = time_to(BOIL_C, sc.T0, T_inf, tau)

    yy, xx = np.mgrid[0:S, 0:S].astype(np.float32)
    rr = np.hypot(xx - CX, yy - CY)
    th = np.arctan2(yy - CY, xx - CX)
    R_out, R_lip, R_w = sc.pot_r, sc.pot_r - 16, sc.pot_r - 30
    water = rr < R_w

    # ---------------------------------------------------------------- static layers
    glass = 20 + 4 * _noise(rng, S, S, 3) + 10 * np.exp(-((xx - S * .25) ** 2 + (yy - S * .2) ** 2) / (2 * 260 ** 2))
    base = np.dstack([glass * 1.05, glass, glass * 0.97])
    ring_mark = (np.abs(rr - (R_out + 60)) < 3).astype(np.float32)
    base += ring_mark[..., None] * np.float32([110, 110, 110])
    # brushed stainless rim and inner wall, lit from light_angle
    lit = np.cos(th - sc.light_angle)
    brushed = 6 * _noise(rng, S, S, 2)
    rim = (rr >= R_lip) & (rr < R_out)
    wall = (rr >= R_w) & (rr < R_lip)
    rim_v = 165 + 55 * lit + 35 * np.cos(2 * (th - sc.light_angle)) + brushed
    wall_v = 105 - 45 * lit + 0.6 * (rr - R_w) + brushed          # far wall lit, near wall dark
    if sc.pot == "black":
        rim_v, wall_v = 38 + 0.3 * (rim_v - 110), 26 + 0.25 * (wall_v - 60)
    for arr, m in ((rim_v, rim), (wall_v, wall)):
        base[m] = np.clip(arr[m], 0, 255)[:, None] * np.float32([0.98, 1.0, 1.02])
    # handles
    for a in (0.0, np.pi):
        hx, hy = CX + (R_out + 55) * np.cos(a), CY + (R_out + 55) * np.sin(a)
        cv2.ellipse(base, (int(hx), int(hy)), (70, 26), np.degrees(a), 0, 360, (70, 72, 76), -1, cv2.LINE_AA)
        cv2.ellipse(base, (int(hx), int(hy - 8)), (60, 10), np.degrees(a), 0, 360, (150, 152, 156), -1, cv2.LINE_AA)
    # pot bottom seen through water: spun stainless
    bottom = 150 + 14 * _noise(rng, S, S, 7) + 9 * np.sin(rr / 2.6 + 0.25 * np.sin(th * 9)) - 0.12 * rr
    bottom += 18 * np.exp(-((rr - R_w * 0.55) ** 2) / (2 * 40 ** 2)) * (0.5 + 0.5 * lit)
    bottom_bgr = np.dstack([bottom * 0.97, bottom * 1.0, bottom * 1.0])
    # specular reflection of the ceiling light on the water surface
    lx, ly = CX + 0.42 * R_w * np.cos(sc.light_angle), CY + 0.42 * R_w * np.sin(sc.light_angle)
    spec = np.zeros((S, S), np.float32)
    cv2.ellipse(spec, (int(lx), int(ly)), (60, 24), np.degrees(sc.light_angle) + 90, 0, 360, 1.0, -1, cv2.LINE_AA)
    spec = cv2.GaussianBlur(spec, (0, 0), 12) * 105
    # foam texture: dense small bubbles
    foam_tex = np.full((S, S), 222, np.float32) + 6 * _noise(rng, S, S, 4)
    for _ in range(9000):
        x, y = rng.uniform(CX - R_w, CX + R_w), rng.uniform(CY - R_w, CY + R_w)
        r = int(rng.integers(2, 7))
        cv2.circle(foam_tex, (int(x), int(y)), r, 196, 1, cv2.LINE_AA)
        cv2.circle(foam_tex, (int(x - r * .3), int(y - r * .3)), max(1, r // 3), 245, -1, cv2.LINE_AA)
    foam_shape = _noise(rng, S, S, 40)
    # ramen strands (a block that loosens)
    strands = []
    for _ in range(70):
        x0, y0 = rng.uniform(-1, 1, 2) * 85
        ang, L, freq = rng.uniform(0, np.pi), rng.uniform(80, 170), rng.uniform(0.08, 0.16)
        u = np.linspace(-L / 2, L / 2, 40)
        strands.append((x0, y0, ang, u, freq, rng.uniform(0, 6.28), rng.uniform(0.6, 1.5)))
    if sc.addin == "pasta":                       # long straight spaghetti fanned out
        strands = []
        for _ in range(90):
            x0, y0 = rng.uniform(-1, 1, 2) * np.float32([60, 150])
            ang, L = 0.35 + rng.normal(0, 0.25), rng.uniform(300, 440)
            strands.append((x0, y0, ang, np.linspace(-L / 2, L / 2, 40), rng.uniform(0.01, 0.025),
                            rng.uniform(0, 6.28), rng.uniform(0.3, 0.8)))
    pieces = []                                   # tofu cubes / kimchi pieces / egg ribbons
    if sc.addin in ("tofu", "egg") or sc.liquid == "red":
        for _ in range(26 if sc.liquid == "red" else 0):
            pieces.append(("kimchi", *(rng.uniform(-0.75, 0.75, 2) * R_w), rng.uniform(0, 3.14), rng.uniform(18, 34)))
        for _ in range(16 if sc.addin == "tofu" else 0):
            pieces.append(("tofu", *(rng.uniform(-0.6, 0.6, 2) * R_w), rng.uniform(0, 3.14), rng.uniform(22, 30)))
        for _ in range(34 if sc.addin == "egg" else 0):
            pieces.append(("egg", *(rng.uniform(-0.7, 0.7, 2) * R_w), rng.uniform(0, 3.14), rng.uniform(25, 60)))
    tint = {"clear": None, "broth": (np.float32([120, 185, 215]), 0.32),
            "red": (np.float32([38, 62, 178]), 0.86)}[sc.liquid]
    froth_col = np.float32([95, 140, 225]) if sc.liquid == "red" else np.float32([240, 242, 243])

    # everything that moves is computed on a crop around the pot only
    c0, c1 = CY - R_out - 12, CY + R_out + 12
    C = slice(c0, c1)
    cs = c1 - c0
    cxx, cyy = xx[C, C], yy[C, C]
    crr, cwater = rr[C, C], water[C, C]
    cbottom, cspec, cfoam_tex = bottom_bgr[C, C], spec[C, C], foam_tex[C, C]
    cfoam_shape = foam_shape[C, C]
    fields = [(_noise(rng, cs, cs, 34), _noise(rng, cs, cs, 34)) for _ in range(6)]
    big = [(_noise(rng, cs, cs, 90), _noise(rng, cs, cs, 90)) for _ in range(6)]
    steam = _noise(rng, OUT_H * 3, OUT_W, 90)
    sites = [(CX + r * np.cos(a), CY + r * np.sin(a)) for r, a in
             zip(np.sqrt(rng.uniform(0, 1, 220)) * R_w * 0.95, rng.uniform(0, 2 * np.pi, 220))]
    Hm = _homography(sc.tilt)
    og = np.mgrid[0:OUT_H, 0:OUT_W].astype(np.float32)
    vign = (1 - 0.32 * (((og[1] - OUT_W / 2) / (OUT_W / 2)) ** 2 + ((og[0] - OUT_H / 2) / (OUT_H / 2)) ** 2))[..., None]
    pc = cv2.perspectiveTransform(np.float32([[[CX, CY]]]), Hm)[0, 0]
    steam_mask = np.exp(-(((og[1] - pc[0]) / 520) ** 2 + ((og[0] - pc[1] + 120) / 420) ** 2))

    final = out_dir / f"{sc.name}.mp4"
    enc = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
                            "-s", f"{OUT_W}x{OUT_H}", "-r", str(sc.fps), "-i", "-", "-c:v", "libx264",
                            "-crf", str(sc.crf), "-preset", "medium", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                            str(final)], stdin=subprocess.PIPE)
    bubbles, ripples, temps = [], [], []
    n = int(sc.seconds * sc.fps)
    foam_t0 = sc.add_at + sc.foam_after_add if (sc.add_at is not None and sc.foam_after_add is not None) else None
    for f in range(n):
        t = f / sc.fps
        boiling = t >= t_boil
        T = BOIL_C if boiling else temp_at(t, sc.T0, T_inf, tau)
        if sc.add_at is not None and sc.add_at <= t < sc.add_at + 25:
            T = max(T - 6 * np.exp(-(t - sc.add_at) / 8), 90)     # cold noodles knock the boil back briefly
            boiling = t >= sc.add_at + 18
        temps.append(T)
        heat_low = sc.heat_down_at is not None and t >= sc.heat_down_at
        roll = boiling and not heat_low
        ph = (t * 0.7) % 6; i0, a = int(ph), ph - int(ph)
        fx = (1 - a) * fields[i0][0] + a * fields[(i0 + 1) % 6][0]
        fy = (1 - a) * fields[i0][1] + a * fields[(i0 + 1) % 6][1]
        conv = float(np.clip((T - 40) / 55, 0, 1))
        amp = 0.15 + 2.6 * conv ** 1.5 + (6.5 if roll else 1.5 if boiling else 0)
        mx, my = cxx - c0 + amp * fx, cyy - c0 + amp * fy
        if roll:
            ph2 = (t * 2.4) % 6; j0, b = int(ph2), ph2 - int(ph2)
            mx = mx + 9 * ((1 - b) * big[j0][0] + b * big[(j0 + 1) % 6][0])
            my = my + 9 * ((1 - b) * big[j0][1] + b * big[(j0 + 1) % 6][1])
        crop = base[C, C].copy()
        bot = cv2.remap(cbottom, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        crop[cwater] = bot[cwater]
        if tint is not None:
            col, al = tint
            shade = (bot[..., 1:2] - 120) * (0.35 if sc.liquid == "red" else 0.6)
            crop[cwater] = (crop[cwater] * (1 - al) + (col + shade[cwater]) * al)
        for kind, px0, py0, rot, sz in pieces:
            if kind != "kimchi" and (sc.add_at is None or t < sc.add_at):
                continue
            ix, iy = int(np.clip(CX - c0 + px0, 0, cs - 1)), int(np.clip(CY - c0 + py0, 0, cs - 1))
            qx, qy = mx[iy, ix] - (cxx[iy, ix] - c0), my[iy, ix] - (cyy[iy, ix] - c0)
            ccx, ccy = CX - c0 + px0 + 1.5 * qx, CY - c0 + py0 + 1.5 * qy
            if kind == "egg":
                age = min(1.0, (t - sc.add_at) / 20)
                cv2.ellipse(crop, (int(ccx), int(ccy)), (int(sz * (0.4 + 0.6 * age)), int(sz * 0.3 * (0.4 + 0.6 * age)) + 1),
                            np.degrees(rot + 0.1 * t), 0, 360, (175, 228, 245), -1, cv2.LINE_AA)
                continue
            box = cv2.boxPoints(((ccx, ccy), (sz, sz * (0.55 if kind == "kimchi" else 1.0)),
                                 np.degrees(rot + 0.05 * np.sin(t * .4 + px0))))
            col = (95, 150, 225) if kind == "kimchi" else (215, 228, 236)
            cv2.fillConvexPoly(crop, box.astype(np.int32), col, cv2.LINE_AA)
        crop[~cwater] = base[C, C][~cwater]
        off = np.float32([c0, c0])
        if sc.add_at is not None and t >= sc.add_at and sc.addin in ("ramen", "pasta"):
            cloud = min(0.28, (t - sc.add_at) / 200)
            crop[cwater] = crop[cwater] * (1 - cloud) + np.float32([205, 210, 212]) * cloud
            spread = min(1.0, (t - sc.add_at) / 70)
            under = crop.copy()
            for (x0, y0, ang, u, freq, phs, curl) in strands:
                ca, sa = np.cos(ang + 0.2 * np.sin(t * .3 + phs)), np.sin(ang + 0.2 * np.sin(t * .3 + phs))
                w = 9 * curl * np.sin(u * freq + phs + t * 0.4 * spread)
                px = CX - c0 + x0 * (1 + 1.4 * spread) + u * ca - w * sa
                py = CY - c0 + y0 * (1 + 1.4 * spread) + u * sa + w * ca
                pts = np.stack([px, py], 1).astype(np.int32)
                thick = (5, 3) if sc.addin == "ramen" else (4, 2)
                cv2.polylines(crop, [pts], False, (95, 150, 180), thick[0], cv2.LINE_AA)
                cv2.polylines(crop, [pts], False, (150, 205, 228), thick[1], cv2.LINE_AA)
            crop[~cwater] = under[~cwater]          # noodles stay inside the pot
        if T >= STAGE_TEMP[1] + sc.stage_dT1 - 2:
            rate = 3 + 45 * float(np.clip((T - STAGE_TEMP[1]) / 26, 0, 1)) + (70 if roll else 0)
            for _ in range(rng.poisson(rate / sc.fps)):
                x, y = sites[rng.integers(len(sites))]
                rising = T >= STAGE_TEMP[2] + sc.stage_dT2
                bubbles.append([x - c0 + rng.normal(0, 4), y - c0 + rng.normal(0, 4),
                                rng.uniform(3, 8) if rising else rng.uniform(1.0, 2.2),
                                rng.uniform(0.25, 0.9) if rising else rng.uniform(1.5, 4.0), rising])
        keep = []
        for bb in bubbles:
            if bb[4]:
                bb[0] += rng.normal(0, 1.8); bb[1] += rng.normal(0, 1.8); bb[2] *= 1.04
            ri = max(1, int(round(bb[2])))
            cv2.circle(crop, (int(bb[0]), int(bb[1])), ri, (205, 210, 214), 1, cv2.LINE_AA)
            cv2.circle(crop, (int(bb[0] - ri * .35), int(bb[1] - ri * .35)), max(1, ri // 3), (250, 252, 253), -1, cv2.LINE_AA)
            bb[3] -= 1 / sc.fps
            if bb[3] > 0:
                keep.append(bb)
            elif bb[4]:
                ripples.append([bb[0], bb[1], bb[2], 1.0])
        bubbles = keep
        nr = []
        for rp in ripples:
            rp[2] += 2.2; rp[3] -= 0.12
            if rp[3] > 0:
                cv2.circle(crop, (int(rp[0]), int(rp[1])), int(rp[2]), (190, 195, 200), 1, cv2.LINE_AA)
                nr.append(rp)
        ripples = nr[-400:]
        if roll:
            froth = (cv2.remap(cfoam_shape, mx, my, cv2.INTER_LINEAR) > 1.05) & cwater
            crop[froth] = crop[froth] * 0.45 + froth_col * 0.55
        if foam_t0 is not None and t >= foam_t0:
            cover = min(1.0, (t - foam_t0) / 26.0)
            if heat_low:
                cover = max(0.0, min(1.0, (sc.heat_down_at - foam_t0) / 26.0) - (t - sc.heat_down_at) / 14.0)
            if cover > 0:
                edge = (1 - crr / R_w) + 0.35 * cfoam_shape + 0.05 * np.sin(t * 2.5 + crr / 15)
                m = cwater & (edge > 1.05 - 1.25 * cover)
                fm = cv2.GaussianBlur(m.astype(np.float32), (0, 0), 3)[..., None]
                crop = crop * (1 - 0.9 * fm) + cfoam_tex[..., None] * 0.9 * fm * np.float32([0.98, 1.0, 1.0])
        sp = cv2.remap(cspec, cxx - c0 + 3 * amp * fx, cyy - c0 + 3 * amp * fy, cv2.INTER_LINEAR)
        if roll:
            sp *= (cv2.remap(cfoam_shape, (mx * 1.7) % cs, (my * 1.7) % cs, cv2.INTER_LINEAR) > 0.2)
        crop[cwater] += sp[cwater][:, None]
        frame = base.copy()
        frame[C, C] = crop
        flick = 1 + 0.012 * np.sin(2 * np.pi * t * 0.9) + 0.006 * rng.standard_normal()
        J = np.float32([[1, 0, rng.normal(0, 0.35)], [0, 1, rng.normal(0, 0.35)], [0, 0, 1]])
        img = cv2.warpPerspective(np.clip(frame, 0, 255), J @ Hm, (OUT_W, OUT_H), flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_REPLICATE)
        hz = float(np.clip((T - 78) / 22, 0, 1)) * (0.18 if heat_low else 0.30)
        if hz > 0:
            k = int(t * 40) % (OUT_H * 2)
            st = np.clip(0.5 + 0.8 * steam[k:k + OUT_H], 0, 1) * steam_mask
            img = img * (1 - hz * st[..., None]) + 225 * hz * st[..., None]
        img = img * vign * flick
        img[..., 2] *= 1.0 + 0.05 * sc.warm; img[..., 0] *= 1.0 - 0.04 * sc.warm
        img = cv2.GaussianBlur(img, (0, 0), 0.7)
        img += rng.normal(0, 2.2, img.shape).astype(np.float32)
        img = np.clip(img, 0, 255).astype(np.uint8)
        if sc.label:
            cv2.putText(img, "Simulated footage", (OUT_W - 238, OUT_H - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (225, 225, 225), 1, cv2.LINE_AA)
        enc.stdin.write(img.tobytes())
    enc.stdin.close()
    enc.wait()
    # pot water outline projected into the image
    ang = np.linspace(0, 2 * np.pi, 180)
    ring = np.stack([CX + R_w * np.cos(ang), CY + R_w * np.sin(ang)], 1).astype(np.float32)[None]
    proj = cv2.perspectiveTransform(ring, Hm)[0]
    (ecx, ecy), (ea, eb), eang = cv2.fitEllipse(proj)
    truth = {
        "scene": asdict(sc), "footage": "simulated (photo-realistic renderer)",
        "t_stage1": round(t_s1, 2), "t_stage2": round(t_s2, 2), "t_boil": round(t_boil, 2),
        "pot": {"cx": round(ecx), "cy": round(ecy), "axes": [round(ea / 2), round(eb / 2)], "angle": round(eang, 1)},
        "temps_1hz": [round(x, 2) for x in temps[::sc.fps]],
    }
    (out_dir / f"{sc.name}.json").write_text(json.dumps(truth))
    return truth
