"""Figures for the technical report, all computed from the actual code and data.

    python docs/make_report_figs.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tempo.kitchen import BURNERS, Kitchen, make_scenario  # noqa: E402
from tempo.physics import nominal_boil_s  # noqa: E402
from tempo.vision import PotWatcher, StageConfig, find_pot, roi_mask, watch_video  # noqa: E402

D = ROOT / "docs"
BLUE, ORANGE, GREEN, RED, GREY, INK = "#0071e3", "#ff9f0a", "#34c759", "#ff3b30", "#8e8e93", "#1d1d1f"
plt.rcParams.update({"font.family": "Inter", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#c7c7cc", "axes.labelcolor": INK, "xtick.color": "#6e6e73",
                     "ytick.color": "#6e6e73", "axes.titleweight": "600", "axes.titlesize": 10, "figure.dpi": 200})


def frame_at(cap, t):
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
    ok, f = cap.read()
    return f


# ------------------------------------------------------------------ F5: what the camera sees
def fig_camera_view(clip="data/real_demo/demo_real.mp4"):
    gt = json.loads(Path(ROOT / clip).with_suffix(".json").read_text())
    cap = cv2.VideoCapture(str(ROOT / clip))
    fps = cap.get(cv2.CAP_PROP_FPS)
    first = frame_at(cap, 0)
    roi = find_pot(first)
    m = roi_mask(first.shape, roi) > 0
    moments = [("Cold water", 30), ("Shimmer", gt["t_stage1"] + 6), ("Simmer", gt["t_stage2"] + 6),
               ("Rolling boil", gt["t_boil"] + 3), ("Foam rising", 286)]
    x0, y0, x1, y1 = 330, 120, 950, 640
    tiles_raw, tiles_cv = [], []
    for name, t in moments:
        a, b = frame_at(cap, t - 1 / fps), frame_at(cap, t)
        ga, gb = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
        vis = (b * 0.45).astype(np.uint8)
        # optical flow arrows (median removed, as in tempo/vision.py)
        fl = cv2.calcOpticalFlowFarneback(cv2.resize(ga, None, fx=.5, fy=.5), cv2.resize(gb, None, fx=.5, fy=.5),
                                          None, 0.5, 2, 13, 2, 5, 1.1, 0) * 2
        fl -= np.median(fl[cv2.resize(m.astype(np.uint8), None, fx=.5, fy=.5) > 0], axis=0)
        for yy in range(y0, y1, 22):
            for xx in range(x0, x1, 22):
                if m[yy, xx]:
                    dx, dy = fl[yy // 2, xx // 2] * 6
                    if abs(dx) + abs(dy) > 2:
                        cv2.arrowedLine(vis, (xx, yy), (int(xx + dx), int(yy + dy)), (40, 150, 255), 2, cv2.LINE_AA, tipLength=.3)
        # bubble births
        hpa = ga.astype(np.float32) - cv2.GaussianBlur(ga.astype(np.float32), (0, 0), 3)
        hpb = gb.astype(np.float32) - cv2.GaussianBlur(gb.astype(np.float32), (0, 0), 3)
        new = ((hpb > 22) & m & ~(hpa > 22)).astype(np.uint8)
        n, _, st, cen = cv2.connectedComponentsWithStats(new, 8)
        for k in range(1, n):
            if 2 <= st[k, cv2.CC_STAT_AREA] <= 120:
                cv2.circle(vis, tuple(int(v) for v in cen[k]), 5, (255, 230, 80), 2, cv2.LINE_AA)
        # foam mask
        hsv = cv2.cvtColor(b, cv2.COLOR_BGR2HSV)
        white = (hsv[..., 1] < 60) & (hsv[..., 2] > 205) & m
        vis[white] = (0.5 * vis[white] + 0.5 * np.array([40, 225, 255])).astype(np.uint8)
        cv2.ellipse(vis, (roi[0], roi[1]), (roi[2], roi[3]), roi[4], 0, 360, (90, 220, 120), 3, cv2.LINE_AA)
        tiles_raw.append(cv2.cvtColor(b[y0:y1, x0:x1], cv2.COLOR_BGR2RGB))
        tiles_cv.append(cv2.cvtColor(vis[y0:y1, x0:x1], cv2.COLOR_BGR2RGB))
    fig, ax = plt.subplots(2, 5, figsize=(10.5, 4.3))
    for j, (name, t) in enumerate(moments):
        ax[0, j].imshow(tiles_raw[j]); ax[1, j].imshow(tiles_cv[j])
        ax[0, j].set_title(f"{name}\n{int(t) // 60}:{int(t) % 60:02d}", fontsize=8.5)
        for r in (0, 1):
            ax[r, j].axis("off")
    ax[0, 0].text(-0.08, .5, "Camera", transform=ax[0, 0].transAxes, rotation=90, va="center", ha="right", fontsize=9, weight=600)
    ax[1, 0].text(-0.08, .5, "OpenCV 5 view", transform=ax[1, 0].transAxes, rotation=90, va="center", ha="right", fontsize=9, weight=600)
    fig.text(.5, .015, "green: pot opening (fitEllipse) · cyan rings: new bubbles (connectedComponentsWithStats) · "
             "orange arrows: Farneback flow minus camera motion · yellow: bright-white mask (HSV + morphology); "
             "the static highlight in the first frames is part of the pot's own baseline and is subtracted",
             ha="center", fontsize=7.6, color="#6e6e73")
    fig.subplots_adjust(left=.04, right=.995, top=.9, bottom=.1, wspace=.03, hspace=.05)
    fig.savefig(D / "fig_camera_view.png"); plt.close(fig)


# ------------------------------------------------------------------ F6: signals and the countdown
def fig_signals(clip="data/real_eval/r01.mp4"):
    gt = json.loads(Path(ROOT / clip).with_suffix(".json").read_text())
    sc = gt["scene"]
    prior = nominal_boil_s(sc["volume_ml"], sc["burner_kw"])
    t0 = time.time()
    cal = next(r for r in json.loads((ROOT / "eval/vision_results.json").read_text())["detail"]["realistic"]["per_clip"]
               if r["clip"] == Path(clip).stem)["calibrated_stage_temp"]
    cfg = StageConfig(stage_temp={int(k): v for k, v in cal.items()})
    w = watch_video(str(ROOT / clip), prior_s=prior, cfg=cfg)
    proc_fps = len(w.series) * sc["fps"] / (time.time() - t0)
    s = w.series
    t = np.array([r["t"] for r in s])
    A = np.array([r.get("A_s", np.nan) for r in s])
    flow = np.array([r["flow"] for r in s]); area = np.array([r["motion_area"] for r in s])
    foam = np.array([r["foam"] for r in s])
    eta = np.array([r["eta"]["t_boil"] if "eta" in r else np.nan for r in s])
    pm = np.array([r["eta"]["pm"] if "eta" in r else np.nan for r in s])
    ev = {e["stage"]: e["t"] for e in w.events if e["kind"] == "stage"}
    warns = [e["t"] for e in w.events if e["kind"] == "foam_warning"]
    fig, ax = plt.subplots(3, 1, figsize=(10, 6.4), sharex=True, gridspec_kw={"height_ratios": [1.1, 1, 1.25]})
    a0 = ax[0]
    a0.plot(t, A, color=BLUE, lw=1.6, label="activity index A (vs. own cold baseline)")
    for th, lab in ((cfg.a_stage1, "shimmer threshold"), (cfg.a_stage2, "simmer threshold")):
        a0.axhline(th, color=GREY, lw=.8, ls="--"); a0.text(t[-1], th + .2, lab, ha="right", fontsize=7, color=GREY)
    a0.set_ylabel("A"); a0.legend(frameon=False, loc="upper left", fontsize=7.5)
    a1 = ax[1]
    a1.plot(t, flow, color=ORANGE, lw=1.4, label="turbulence (flow, px)")
    a1b = a1.twinx(); a1b.spines["right"].set_visible(True)
    a1b.plot(t, area, color=GREEN, lw=1.2, label="moving area"); a1b.plot(t, foam, color=RED, lw=1.2, label="foam cover")
    a1.set_ylabel("flow px"); a1b.set_ylabel("share of pot")
    h1, l1 = a1.get_legend_handles_labels(); h2, l2 = a1b.get_legend_handles_labels()
    a1.legend(h1 + h2, l1 + l2, frameon=False, loc="upper left", fontsize=7.5)
    a2 = ax[2]
    ok = ~np.isnan(eta)
    a2.fill_between(t[ok], (eta - pm)[ok], (eta + pm)[ok], color=BLUE, alpha=.15, lw=0)
    a2.plot(t[ok], eta[ok], color=BLUE, lw=1.6, label="Tempo's predicted boil time ± uncertainty")
    a2.axhline(gt["t_boil"], color=INK, lw=1, label=f"true boil ({gt['t_boil']:.0f} s)")
    a2.axhline(prior, color=GREY, lw=1, ls="--", label=f"recipe / nameplate estimate ({prior:.0f} s)")
    a2.set_ylim(gt["t_boil"] - 90, gt["t_boil"] + 90); a2.set_ylabel("boil at (s)"); a2.set_xlabel("time since pot on burner (s)")
    a2.legend(frameon=False, loc="upper right", fontsize=7.5)
    for a in ax:
        for k, key, c in ((1, "t_stage1", GREY), (2, "t_stage2", GREY)):
            a.axvline(gt[key], color=c, lw=.6, ls=":")
        a.axvline(gt["t_boil"], color=INK, lw=.8, ls=":")
        if sc.get("add_at") and sc.get("heat_down_at"):
            a.axvspan(sc["add_at"], sc["heat_down_at"], color=ORANGE, alpha=.06, lw=0)
    for k, tt in ev.items():
        ax[0].plot(tt, np.interp(tt, t, np.nan_to_num(A)), "o", color=BLUE, ms=5)
        ax[0].annotate(["", "shimmer", "simmer", "boil"][k] + f" {tt:.0f}s", (tt, np.interp(tt, t, np.nan_to_num(A))),
                       xytext=(-6, 10 + 10 * (k == 3)), textcoords="offset points", fontsize=7, color=BLUE, ha="right")
    if warns:
        ax[1].axvline(warns[0], color=RED, lw=1); ax[1].text(warns[0] + 2, ax[1].get_ylim()[1] * .85, f"foam warning {warns[0]:.0f}s → heat lowered", fontsize=7, color=RED)
    ax[0].set_title("One pot, second by second: what OpenCV measures and how the countdown converges", loc="left")
    fig.text(.01, .005, f"Held-out simulated clip {Path(clip).stem}: {sc['volume_ml']:.0f} ml on a {sc['burner_kw']:.1f} kW burner whose real efficiency differs from the nameplate. "
             "Dotted lines: true stage onsets and boil. Stage temperatures calibrated leave-one-out on the other clips.",
             fontsize=7, color="#6e6e73")
    fig.tight_layout(rect=(0, .02, 1, 1)); fig.savefig(D / "fig_signals.png"); plt.close(fig)
    return proc_fps


# ------------------------------------------------------------------ F7: service timeline, plain display vs Tempo
def fig_gantt(seed=140):
    fig, axs = plt.subplots(2, 1, figsize=(10, 4.6), sharex=True)
    tick_col = {}
    pal = ["#0071e3", "#ff9f0a", "#34c759", "#af52de", "#ff3b30", "#5ac8fa"]
    out = {}
    for ax, pol, title in ((axs[0], "fifo", "Plain kitchen display (fire on order)"), (axs[1], "tempo", "Tempo")):
        k = Kitchen(make_scenario(seed, n_tickets=6, mean_gap_s=240), pol)
        m = k.run(); out[pol] = m
        rows = [b for b, _ in BURNERS]
        for it in k.items:
            if it.burner is None or it.t_load is None:
                continue
            y = rows.index(it.burner)
            c = tick_col.setdefault(it.tid, pal[len(tick_col) % len(pal)])
            end_heat = it.t_cook_start or it.t_done or it.t_load
            ax.barh(y, end_heat - it.t_load, left=it.t_load, height=.55, color=c, alpha=.35, lw=0)
            if it.t_cook_start and it.t_done:
                ax.barh(y, it.t_done - it.t_cook_start, left=it.t_cook_start, height=.55, color=c, lw=0)
            srv = k.tickets[it.tid].get("served")
            if it.t_done and srv and srv > it.t_done:
                ax.barh(y + .36, srv - it.t_done, left=it.t_done, height=.1, color=INK, lw=0)
            ax.text(it.t_load + 6, y, f"{it.tid}", va="center", fontsize=6.5, color=INK)
        for tid, tk in k.tickets.items():
            if "served" in tk:
                ax.plot(tk["served"], 3.75, "v", color=tick_col.get(tid, GREY), ms=6, clip_on=False)
        ax.set_yticks(range(len(rows))); ax.set_yticklabels(rows); ax.invert_yaxis()
        ax.set_title(f"{title} — cooked food waits {m['food_wait_mean_s']:.0f} s on average, "
                     f"{m['boilovers']} boil-overs", loc="left", fontsize=9)
    axs[1].set_xlabel("service time (s)")
    fig.text(.01, .01, "Light bar: heating · dark bar: cooking after the add · thin black line under a row: that dish is cooked and waiting for the rest of its ticket · ▼ ticket served. "
             "Same six tickets, same burners (simulator, seed 140).", fontsize=7, color="#6e6e73")
    fig.tight_layout(rect=(0, .03, 1, 1)); fig.savefig(D / "fig_gantt.png"); plt.close(fig)
    return out


# ------------------------------------------------------------------ F8: real stock clips
def fig_real():
    res = json.loads((ROOT / "eval/real_check.json").read_text())
    names = ["p1", "p2", "p3", "p8", "p4", "p6", "p5", "p10"]
    labels = ["Rolling boil (pan)", "Stew, gentle simmer", "Near-still water", "Pre-boil bubbles, top-down",
              "Pasta: in → boil → tongs", "Corn, edge simmer", "Broth: hand + ladle", "Phone close-up, pre-boil"]
    verdict = ["correct", "correct", "correct", "correct", "boil found, 2 s early", "ambiguous", "wrong", "wrong"]
    fig, ax = plt.subplots(2, 4, figsize=(10, 3.9))
    for j, n in enumerate(names):
        cap = cv2.VideoCapture(str(ROOT / f"data/stock/{n}.mp4"))
        N = cap.get(cv2.CAP_PROP_FRAME_COUNT); cap.set(cv2.CAP_PROP_POS_FRAMES, int(N * .5)); _, f = cap.read()
        if f.shape[0] > f.shape[1]:
            h = f.shape[1] * 9 // 16; y = (f.shape[0] - h) // 2; f = f[y:y + h]
        a = ax[j // 4, j % 4]; a.imshow(cv2.cvtColor(cv2.resize(f, (480, 270)), cv2.COLOR_BGR2RGB)); a.axis("off")
        rows = res[n]["rows"]; nb = sum(r["boil"] for r in rows)
        col = GREEN if verdict[j] == "correct" else (RED if verdict[j] == "wrong" else ORANGE)
        a.set_title(labels[j], fontsize=8)
        a.text(.02, .04, f"boil {nb}/{len(rows)} s · {verdict[j]}", transform=a.transAxes, fontsize=7.5, color="white",
               bbox=dict(boxstyle="round,pad=.3", fc=col, ec="none"))
    fig.text(.01, .01, "Free-licence stock clips (Pixabay, Pexels), frame from mid-clip. No ground truth; judged by eye; thresholds not tuned on these clips.",
             fontsize=7, color="#6e6e73")
    fig.subplots_adjust(left=.01, right=.99, top=.93, bottom=.06, wspace=.04, hspace=.16)
    fig.savefig(D / "fig_real.png"); plt.close(fig)


# ------------------------------------------------------------------ F9: board clips strip
def fig_board_clips():
    names = [("clip_pasta", "B1 · pasta"), ("demo_pot", "B2 · ramen"), ("clip_egg_soup", "B3 · egg soup"), ("clip_kimchi_stew", "B4 · kimchi stew")]
    fig, ax = plt.subplots(len(names), 4, figsize=(7.6, 6.2))
    for i, (n, lab) in enumerate(names):
        meta = json.loads((ROOT / f"samples/{n}.json").read_text())
        ts = [20, meta["stages"]["2"] or meta["boil_s"] - 30, meta["boil_detect_s"] + 3, (meta["add_s"] or meta["boil_s"]) + 30]
        cap = cv2.VideoCapture(str(ROOT / f"samples/{n}.mp4"))
        for j, t in enumerate(ts):
            f = frame_at(cap, t)
            h, w = f.shape[:2]; f = f[int(h * .08):int(h * .92), int(w * .22):int(w * .78)]
            a = ax[i, j]; a.imshow(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)); a.set_xticks([]); a.set_yticks([])
            for sp in a.spines.values():
                sp.set_visible(False)
            a.set_title(["cold", "simmer", "boil detected", "food in"][j] + f" · {int(t) // 60}:{int(t) % 60:02d}", fontsize=7.5)
        ax[i, 0].set_ylabel(lab, fontsize=8.5, weight=600)
    fig.subplots_adjust(left=.05, right=.995, top=.95, bottom=.01, wspace=.03, hspace=.28)
    fig.savefig(D / "fig_board_clips.png"); plt.close(fig)


if __name__ == "__main__":
    fig_camera_view(); print("camera view")
    fps = fig_signals(); print("signals, processing fps", round(fps, 1))
    g = fig_gantt(); print("gantt", {k: (round(v["food_wait_mean_s"]), v["boilovers"]) for k, v in g.items()})
    fig_real(); print("real")
    fig_board_clips(); print("board clips")
    json.dump({"processing_fps_1cpu": round(fps, 1)}, open(D / "fig_stats.json", "w"))
