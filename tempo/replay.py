"""Build a second-by-second replay of a service for the station screens.

Optionally one burner is driven by real perception: the first pot loaded on
that burner takes its stage / boil / foam timeline from PotWatcher running on
a video clip, and the screen shows that clip in sync.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from .kitchen import (ADD_NOTES, BURNERS, LIQUID_NOTES, LOAD_S, PREP_NOTES, Kitchen, make_scenario)


def _fmt(s: float) -> str:
    s = max(0, int(round(s)))
    return f"{s // 60}:{s % 60:02d}"


def snapshot(k: Kitchen) -> dict:
    t = k.t
    burners = []
    for b, kw in BURNERS:
        it = k.burner_busy[b]
        card = {"name": b, "kw": kw, "item": None, "phase": "empty", "prompt": "", "blink": False, "level": "idle"}
        if it is None:
            nxt = sorted([i for i in k.items if i.burner is None and i.t_plated is None and i.plan.get("burner") == b],
                         key=lambda i: i.plan.get("load", 0))
            if nxt and k.policy != "fifo":
                n = nxt[0]
                due = n.plan["load"] - t
                card["next"] = {"ticket": n.tid, "dish": n.dish.name, "liquid": LIQUID_NOTES[n.dish.name]}
                if due <= 10 and n.prepped:
                    card.update(prompt=f"LOAD NOW · {n.tid} {n.dish.name} · {LIQUID_NOTES[n.dish.name]}", blink=True, level="go")
                else:
                    card.update(prompt=f"Next: {n.tid} {n.dish.name} in {_fmt(due)}", level="idle")
        else:
            card["item"] = {"ticket": it.tid, "dish": it.dish.name}
            if it.t_done is not None:
                card.update(phase="done", prompt=f"PLATE NOW → pass · {it.tid}", blink=True, level="go", progress=1.0)
            elif it.t_cook_start is not None:
                p = (t - it.t_cook_start) / it.dish.cook
                card.update(phase="cooking", progress=round(min(p, 1), 3),
                            prompt=f"{it.dish.name} cooking · {_fmt(it.t_cook_start + it.dish.cook - t)} left", level="ok")
                if any(e["kind"] == "auto_heat_down" and e.get("burner") == b and t - e["t"] < 40 for e in k.events[-40:]):
                    card.update(prompt="Foam rising → heat lowered automatically", level="warn", blink=True)
            elif it.boils and it.t_load is not None:
                seen_boil = it.t_boil_seen is not None and t >= it.t_boil_seen
                if seen_boil:
                    hold = it.plan.get("add", -1e9) - t
                    if hold > 8:
                        card.update(phase="boiling", prompt=f"Boiling · hold, add in {_fmt(hold)} (syncing {it.tid})",
                                    level="ok", progress=1.0)
                    else:
                        card.update(phase="boiling", prompt=f"ADD NOW · {ADD_NOTES[it.dish.name]}", blink=True,
                                    level="go", progress=1.0)
                else:
                    e = it.est.eta(t) if it.est else {"t_boil": it.t_load + 240, "pm": 60, "basis": "prior"}
                    span = max(e["t_boil"] - it.t_load, 1)
                    card.update(phase="heating", stage=it.est.stage if it.est else 0,
                                eta_s=round(e["t_boil"] - t), eta_pm=round(e["pm"]), basis=e["basis"],
                                progress=round(min((t - it.t_load) / span, 0.99), 3),
                                prompt=f"Heating · boils in ~{_fmt(e['t_boil'] - t)} ±{int(e['pm'])}s", level="ok")
            else:
                card.update(phase="loading", prompt="Loading", level="ok")
        burners.append(card)

    a = k.cook_action
    prep_now = None
    if a is not None and a.kind == "prep":
        prep_now = {"ticket": a.item.tid, "dish": a.item.dish.name, "note": PREP_NOTES[a.item.dish.name],
                    "left_s": round(k.cook_until - t), "progress": round(1 - (k.cook_until - t) / a.dur, 3)}
    queue = sorted([i for i in k.items if not i.prepped and i.t_plated is None and not (a and a.kind == "prep" and a.item is i)],
                   key=lambda i: (i.plan.get("prep", k.tickets[i.tid]["t"]), i.tid))
    prep_next = [{"ticket": i.tid, "dish": i.dish.name, "note": PREP_NOTES[i.dish.name],
                  "in_s": round(i.plan.get("prep", t) - t) if k.policy != "fifo" else 0} for i in queue[:3]]
    passes = []
    for tid, tk in k.tickets.items():
        its = [i for i in k.items if i.tid == tid]
        if "served" in tk and t - tk["served"] > 45:
            continue
        passes.append({"ticket": tid, "items": [{"dish": i.dish.name, "state": _state(i, t)} for i in its],
                       "plated": sum(i.t_plated is not None for i in its), "n": len(its),
                       "target_in": round(tk.get("target", t) - t) if "served" not in tk else None,
                       "served": "served" in tk, "age_s": round(t - tk["t"])})
    return {"t": t, "burners": burners, "prep_now": prep_now, "prep_next": prep_next, "pass": passes,
            "cook": None if a is None else {"action": a.kind, "ticket": a.item.tid, "dish": a.item.dish.name}}


def _state(i, t) -> str:
    if i.t_plated is not None:
        return "plated"
    if i.t_done is not None:
        return "ready"
    if i.t_cook_start is not None:
        return "cooking"
    if i.t_load is not None:
        return "boiling" if i.t_boil_seen is not None and t >= i.t_boil_seen else "heating"
    if i.prepped:
        return "prepped"
    return "waiting"


def build(seed: int = 7, n_tickets: int = 6, mean_gap_s: float = 160.0, step: int = 2,
          video=None, out: Optional[Path] = None) -> dict:
    """video: one clip dict or a list of them, one per dish.
    Each clip = {"dish": .., "clip": url, "boil_s": .., "boil_detect_s": .., "stages": {1: s, 2: s},
                 "add_s": .., "foam_warn_s": ..}; it drives the first pot of that dish loaded in the service."""
    sc = make_scenario(seed, n_tickets=n_tickets, mean_gap_s=mean_gap_s)
    k = Kitchen(sc, "tempo", log_events=True)
    videos = [video] if isinstance(video, dict) else list(video or [])
    bound: dict[str, dict] = {}
    if videos:
        orig_finish = k.finish

        def finish(a):
            v = None
            if a.kind == "load" and a.item.boils:
                v = next((x for x in videos if x.get("dish", "ramen") == a.item.dish.name and x["dish"] not in bound
                          and x.get("burner") in (None, "auto", a.burner)), None)
            orig_finish(a)
            if v is not None:
                it = a.item
                it.t_boil = it.t_load + v["boil_s"]
                it.t_boil_seen = it.t_load + v["boil_detect_s"]
                it.stage_times = {int(s): it.t_load + x for s, x in v["stages"].items()}
                it.stage_times[3] = it.t_boil_seen
                if v.get("foam_warn_s") and v.get("add_s") and v["foam_warn_s"] > v["add_s"]:
                    from dataclasses import replace as _r
                    it.truth = _r(it.truth, foam_at=v["foam_warn_s"] - v["add_s"] - it.dish.add, foam_seen=True)
                it.est.curvature = 1.0        # calibrated camera setup, same as the perception run
                bound[v["dish"]] = {**v, "key": it.key, "t_load": it.t_load, "burner": it.burner}
                k.log("camera_clip", burner=it.burner, ticket=it.tid, dish=it.dish.name)
        k.finish = finish
    frames = []
    while k.t < 4 * 3600:
        k.step()
        if int(k.t) % step == 0:
            frames.append(snapshot(k))
        if k._next_ticket == len(sc.tickets) and all("served" in tk for tk in k.tickets.values()):
            frames.append(snapshot(k))
            break
        k.t += 1.0
    rep = {"meta": {"seed": seed, "tickets": sc.tickets, "burners": BURNERS, "step": step,
                    "videos": list(bound.values()),
                    "video": next(iter(bound.values()), None)},
           "frames": frames, "events": k.events, "metrics": k.metrics()}
    if out:
        out.write_text(json.dumps(rep, separators=(",", ":"), default=float))
    return rep
