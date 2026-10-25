"""Kitchen simulator and order-orchestration policies.

The kitchen: one cook, a few burners of different (and not exactly known)
power, a prep board, a pass. Tickets arrive during a rush; each ticket has
several dishes that should reach the table *together*.

The uncertainty that breaks ordinary kitchen display systems is heat-up
time: how long a pot takes to boil depends on the burner, the pot, the
amount and starting temperature of the liquid. Cook timers cover the part
after boiling; nothing covers the part before. Policies compared:

  fifo        fire every dish as soon as possible (what a plain KDS does)
  plan_fixed  backward-schedule each ticket once with nominal heat-up times
  plan_alert  backward-schedule, re-plan on events, camera *detects* boil
              (alert) but does not predict it
  tempo       camera *predicts* boil from surface stages (physics model),
              learns each burner, re-plans continuously, keeps the cook free
              for predicted events, and turns heat down on predicted boil-over
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .physics import (BOIL_C, STAGE_TEMP, BoilEstimator, BurnerModel, heat_params,
                      nominal_boil_s, time_to)

LOAD_S = 10
TIMER_NOTICE_S = 2      # cook timers exist in every kitchen
ALERT_NOTICE_S = 2      # blinking station display
FOAM_WINDOW_S = 20      # foam surge becomes a boil-over if nobody acts within this
REPLAN_EVERY_S = 5
DUE_MARGIN_S = 10


@dataclass(frozen=True)
class Dish:
    name: str
    prep: int
    liquid_ml: int      # 0 = pan dish, no boil phase
    add: int            # seconds of cook work at boil (noodles, tofu...)
    cook: int           # timed cooking after add
    plate: int
    foam: bool = False


MENU = {
    "ramen": Dish("ramen", prep=60, liquid_ml=550, add=15, cook=240, plate=30, foam=True),
    "kimchi_stew": Dish("kimchi_stew", prep=120, liquid_ml=750, add=20, cook=360, plate=35),
    "egg_soup": Dish("egg_soup", prep=45, liquid_ml=500, add=15, cook=90, plate=25),
    "pasta": Dish("pasta", prep=40, liquid_ml=1000, add=15, cook=480, plate=40, foam=True),
    "omelette": Dish("omelette", prep=60, liquid_ml=0, add=0, cook=180, plate=30),
}
PREP_NOTES = {
    "ramen": "scallion 1 stalk, sliced 5 mm",
    "kimchi_stew": "kimchi 150 g, tofu 1/2 block in 2 cm cubes, onion 1/4",
    "egg_soup": "2 eggs beaten, scallion 1/2 stalk",
    "pasta": "garlic 2 cloves sliced, salt 10 g ready",
    "omelette": "3 eggs beaten, carrot 20 g + scallion minced",
}
ADD_NOTES = {
    "ramen": "noodles + soup base",
    "kimchi_stew": "kimchi, tofu, onion",
    "egg_soup": "pour eggs in a thin stream",
    "pasta": "pasta 100 g",
    "omelette": "",
}
LIQUID_NOTES = {"ramen": "water 550 ml", "kimchi_stew": "stock 750 ml", "egg_soup": "stock 500 ml",
                "pasta": "water 1 L", "omelette": "pan, oil"}
MENU_WEIGHTS = {"ramen": 0.32, "kimchi_stew": 0.2, "egg_soup": 0.16, "pasta": 0.14, "omelette": 0.18}
BURNERS = [("B1", 3.0), ("B2", 2.6), ("B3", 2.2), ("B4", 1.7)]


# ----------------------------------------------------------------------------- scenario
@dataclass
class ItemTruth:
    """Physical facts about one dish's pot, independent of the policy."""
    vol_ml: float
    eff: float
    T0: float
    stage_dT: dict          # per-stage offset of the visible-stage temperature
    stage_lag: dict         # camera detection lag per stage
    notice_s: float         # how long an unprompted cook takes to notice a boil
    foam_at: Optional[float]  # seconds after add when foam surges (None = no surge)
    foam_seen: bool         # camera catches the surge in time


@dataclass
class Scenario:
    seed: int
    tickets: list            # [{"id", "t", "items": [dish names]}]
    truth: dict              # (ticket id, idx) -> ItemTruth
    burner_true_kw: dict     # burner name -> actual kW (nameplate is wrong)


def make_scenario(seed: int, n_tickets: int = 12, mean_gap_s: float = 300.0,
                  vision_foam_recall: float = 0.95) -> Scenario:
    rng = np.random.default_rng(seed)
    names, w = list(MENU_WEIGHTS), np.array(list(MENU_WEIGHTS.values()))
    t, tickets, truth = 0.0, [], {}
    for k in range(n_tickets):
        n = int(rng.choice([1, 2, 2, 3, 3, 4]))
        items = list(rng.choice(names, size=n, p=w / w.sum()))
        tickets.append({"id": f"T{k + 1}", "t": round(t), "items": [str(i) for i in items]})
        for j, it in enumerate(items):
            d = MENU[str(it)]
            truth[(f"T{k + 1}", j)] = ItemTruth(
                vol_ml=d.liquid_ml * float(rng.normal(1.0, 0.08)),
                eff=0.45 * float(rng.lognormal(0, 0.14)),
                T0=float(rng.uniform(10, 24)),
                stage_dT={1: float(rng.normal(0, 3.5)), 2: float(rng.normal(0, 2.5))},
                stage_lag={1: max(0.0, float(rng.normal(6, 3))), 2: max(0.0, float(rng.normal(4, 2))),
                           3: max(0.0, float(rng.normal(3, 1.5)))},
                notice_s=float(rng.uniform(4, 25)),
                foam_at=float(rng.uniform(30, 150)) if d.foam and rng.random() < 0.4 else None,
                foam_seen=bool(rng.random() < vision_foam_recall),
            )
        t += float(rng.exponential(mean_gap_s))
    true_kw = {b: kw * float(rng.uniform(0.75, 1.1)) for b, kw in BURNERS}
    return Scenario(seed, tickets, truth, true_kw)


# ----------------------------------------------------------------------------- state
@dataclass
class Item:
    tid: str
    idx: int
    dish: Dish
    truth: ItemTruth
    prepped: bool = False
    burner: Optional[str] = None
    t_load: Optional[float] = None
    t_boil: Optional[float] = None        # true boil time (absolute)
    t_boil_seen: Optional[float] = None   # camera detection time
    stage_times: dict = field(default_factory=dict)  # stage -> time camera reports it
    est: Optional[BoilEstimator] = None
    t_add: Optional[float] = None         # cook started adding
    t_cook_start: Optional[float] = None
    t_done: Optional[float] = None        # timer done
    t_plate: Optional[float] = None       # plating started (pot off burner)
    t_plated: Optional[float] = None
    boilover: bool = False
    foam_handled: bool = False
    plan: dict = field(default_factory=dict)  # due times: prep, load, add

    @property
    def key(self):
        return (self.tid, self.idx)

    @property
    def boils(self) -> bool:
        return self.dish.liquid_ml > 0


@dataclass
class Action:
    kind: str       # prep | load | add | plate
    item: Item
    dur: int
    burner: Optional[str] = None


class Kitchen:
    def __init__(self, sc: Scenario, policy: str, log_events: bool = False):
        assert policy in {"fifo", "plan_fixed", "plan_alert", "tempo"}
        self.sc, self.policy = sc, policy
        self.t = 0.0
        self.items: list[Item] = []
        self.tickets: dict[str, dict] = {}
        self.burner_busy: dict[str, Optional[Item]] = {b: None for b, _ in BURNERS}
        self.models = {b: BurnerModel(b, kw) for b, kw in BURNERS}
        self.cook_until = 0.0
        self.cook_action: Optional[Action] = None
        self.cook_busy_s = 0.0
        self.events: list[dict] = []
        self.log_events = log_events
        self._next_ticket = 0
        self._replan_needed = False
        self._replan_reason = ""

    # -- logging
    def log(self, kind: str, **kw):
        if self.log_events:
            self.events.append({"t": round(self.t, 1), "kind": kind, **kw})

    # -- physics ------------------------------------------------------------
    def _true_boil_time(self, it: Item, burner: str) -> float:
        T_inf, tau = heat_params(it.truth.vol_ml, self.sc.burner_true_kw[burner], it.truth.eff)
        return time_to(BOIL_C, it.truth.T0, T_inf, tau)

    def _stage_time(self, it: Item, s: int) -> float:
        T_inf, tau = heat_params(it.truth.vol_ml, self.sc.burner_true_kw[it.burner], it.truth.eff)
        T = STAGE_TEMP[s] + it.truth.stage_dT.get(s, 0.0)
        return it.t_load + min(time_to(T, it.truth.T0, T_inf, tau), it.t_boil - it.t_load)

    # -- estimates used by planners ------------------------------------------
    def heat_estimate(self, it: Item, burner: str) -> float:
        """Expected seconds from load to boil on a burner (policy's belief)."""
        if not it.boils:
            return 0.0
        if self.policy == "tempo":
            return self.models[burner].prior_s(it.dish.liquid_ml)
        return nominal_boil_s(it.dish.liquid_ml, dict(BURNERS)[burner])

    def boil_eta(self, it: Item) -> float:
        """Belief about absolute boil time of a pot already on a burner."""
        if it.t_boil_seen is not None and self.t >= it.t_boil_seen and self.policy in {"plan_alert", "tempo"}:
            return it.t_boil_seen          # observed - never read the future
        if self.policy == "tempo" and it.est is not None:
            return it.est.eta(self.t)["t_boil"]
        t_load = it.t_load if it.t_load is not None else self.cook_until
        eta = t_load + self.heat_estimate(it, it.burner)
        return max(eta, self.t + 1) if self.policy == "plan_alert" else eta

    def post_boil(self, it: Item) -> float:
        return it.dish.add + it.dish.cook + it.dish.plate

    # -- planning -----------------------------------------------------------
    def plan_all(self, reason: str = ""):
        before = {i.key: (i.plan.get("load"), i.plan.get("add")) for i in self.items}
        self._plan_all()
        if self.log_events and reason:
            changes = []
            for i in self.items:
                if i.t_plated is not None or i.key not in before:
                    continue
                if i.burner is None:
                    old, new = before[i.key][0], i.plan.get("load")
                    step = "start"
                elif i.boils and i.t_add is None:
                    old, new = before[i.key][1], i.plan.get("add")
                    step = "add"
                else:
                    continue
                if old is not None and new is not None and old > -1e8 and new > -1e8 and abs(new - old) >= 15:
                    changes.append({"ticket": i.tid, "dish": i.dish.name, "step": step, "shift_s": round(new - old)})
            if changes:
                self.log("replan", reason=reason, changes=changes)

    def _plan_all(self):
        """Global backward-aligned plan that never idles a burner someone needs.

        1. Forward pass in ticket order: every waiting dish gets the burner on
           which it would finish first, given when burners free up (believed
           durations) and when the cook can do the prep.
        2. Each ticket's target time T = the latest finish among its dishes.
        3. Backward shift: a dish that would finish early is started later,
           but only by as much as (a) aligns it with T and (b) keeps its
           burner free in time for the next dish that needs it. Boiled pots
           of the same kind can be held before the add step, under the same
           two limits.
        """
        open_items = [i for i in self.items if i.t_plate is None and i.t_plated is None]
        if not open_items:
            return
        order = {tid: k for k, tid in enumerate(self.tickets)}
        free_at = {b: self.t for b, _ in BURNERS}
        uses: dict[str, list] = {b: [] for b, _ in BURNERS}   # (start, release, key) per burner
        fin, start_of, burner_of, release_of = {}, {}, {}, {}
        for i in open_items:
            if i.burner is not None:
                rel = self._believed_cook_end(i)
                fin[i.key], burner_of[i.key], release_of[i.key] = rel + i.dish.plate, i.burner, rel
                start_of[i.key] = i.t_load if i.t_load is not None else self.t
                free_at[i.burner] = max(free_at[i.burner], rel)
                uses[i.burner].append((start_of[i.key], rel, i.key))
        cook_free = max(self.t, self.cook_until)
        pending = [i for i in open_items if i.burner is None]
        pending.sort(key=lambda i: (order[i.tid], -(self.heat_estimate(i, BURNERS[0][0]) + self.post_boil(i))))
        for i in pending:
            ready = cook_free + (0 if i.prepped else i.dish.prep)
            best = None
            for b, kw in sorted(BURNERS, key=lambda x: -x[1]):
                st = max(ready, free_at[b])
                f = st + LOAD_S + self.heat_estimate(i, b) + self.post_boil(i)
                if best is None or f < best[0] - 1:
                    best = (f, b, st)
            prev = i.plan.get("burner")
            if prev and prev != best[1]:      # hysteresis: screens should not reshuffle for small gains
                st_p = max(ready, free_at[prev])
                f_p = st_p + LOAD_S + self.heat_estimate(i, prev) + self.post_boil(i)
                if f_p <= best[0] + 30:
                    best = (f_p, prev, st_p)
            f, b, st = best
            if not i.prepped:
                cook_free = ready
            fin[i.key], burner_of[i.key], start_of[i.key] = f, b, st
            release_of[i.key] = f - i.dish.plate
            free_at[b] = release_of[i.key]
            uses[b].append((st, release_of[i.key], i.key))
        T = {}
        for i in open_items:
            T[i.tid] = max(T.get(i.tid, -1e9), fin[i.key])
        for tid, v in T.items():
            self.tickets[tid]["target"] = v
        # backward shift limited by the next user of the same burner
        for b, lst in uses.items():
            lst.sort()
            for k, (st, rel, key) in enumerate(lst):
                nxt = lst[k + 1][0] if k + 1 < len(lst) else math.inf
                it = next(x for x in open_items if x.key == key)
                room = min(T[it.tid] - fin[key], nxt - rel)
                room = max(0.0, room)
                if it.burner is None:
                    load = st + room
                    it.plan.update(burner=b, load=load, prep=load - it.dish.prep,
                                   add=load + LOAD_S + self.heat_estimate(it, b) + room * 0,
                                   finish=fin[key] + room)
                    it.plan["add"] = -1e9 if not it.boils else it.plan["add"]
                else:
                    if it.boils and it.t_add is None:
                        it.plan["add"] = max(self.boil_eta(it), self.t) + room
                    it.plan["finish"] = fin[key] + room

    def _believed_cook_end(self, it: Item) -> float:
        if it.t_cook_start is not None:
            return it.t_cook_start + it.dish.cook
        if not it.boils:
            return (it.t_load or self.t) + LOAD_S + it.dish.cook
        return max(self.boil_eta(it), self.t) + it.dish.add + it.dish.cook

    # -- the cook -----------------------------------------------------------
    def candidate_actions(self) -> list[tuple[float, Action]]:
        """(priority, action) — lower priority value is chosen first."""
        out = []
        t = self.t
        for it in self.items:
            if it.t_plated is not None or it.t_plate is not None:
                continue
            # plate when the timer rang
            if it.t_done is not None and t >= it.t_done + TIMER_NOTICE_S:
                out.append((0, Action("plate", it, it.dish.plate)))
                continue
            # add ingredients once the pot boils and the cook knows it
            if it.boils and it.t_boil is not None and it.t_add is None and t >= it.t_boil:
                if self.policy in {"plan_alert", "tempo"}:
                    known = it.t_boil_seen is not None and t >= it.t_boil_seen + ALERT_NOTICE_S
                else:
                    known = t >= it.t_boil + it.truth.notice_s
                due = it.plan.get("add", -1e9) if self.policy != "fifo" else -1e9
                if known and t >= due - DUE_MARGIN_S:
                    out.append((1, Action("add", it, it.dish.add)))
                continue
            if it.burner is None and (it.boils or it.t_cook_start is None):
                if not it.prepped:
                    due = it.plan.get("prep", -1e9) if self.policy != "fifo" else -1e9
                    if t >= due - DUE_MARGIN_S:
                        out.append((2 + self._rank(it, due), Action("prep", it, it.dish.prep)))
                else:
                    due = it.plan.get("load", -1e9) if self.policy != "fifo" else -1e9
                    b = self._pick_burner(it)
                    if b and t >= due - DUE_MARGIN_S:
                        out.append((2 + self._rank(it, due), Action("load", it, LOAD_S, b)))
        return sorted(out, key=lambda x: x[0])

    def _rank(self, it: Item, due: float) -> float:
        if self.policy == "fifo":
            return self.tickets[it.tid]["t"] / 1e6 + it.idx / 1e9
        return (due - self.t) / 1e5 + 0.5

    def _pick_burner(self, it: Item) -> Optional[str]:
        free = [b for b, _ in BURNERS if self.burner_busy[b] is None]
        if not free:
            return None
        if self.policy != "fifo" and it.plan.get("burner") in free:
            return it.plan["burner"]
        return max(free, key=lambda b: dict(BURNERS)[b])

    def next_critical(self) -> float:
        """Earliest predicted moment the cook will be needed at a pot."""
        nxt = math.inf
        for it in self.items:
            if it.t_plate is not None or it.t_plated is not None:
                continue
            if it.t_cook_start is not None and it.t_done is None:
                nxt = min(nxt, it.t_cook_start + it.dish.cook)
            elif it.burner and it.boils and it.t_add is None and it.t_load is not None:
                eta = self.boil_eta(it)
                if it.plan.get("add", -1e9) <= eta + 5:    # held pots are discretionary, not critical
                    nxt = min(nxt, eta)
        return nxt

    def choose(self) -> Optional[Action]:
        cands = self.candidate_actions()
        if not cands:
            return None
        if self.policy != "tempo":
            return cands[0][1]
        crit = self.next_critical()
        for pri, a in cands:
            if a.kind in {"plate", "add"}:
                return a
            overdue = self.t - (a.item.plan.get(a.kind, -1e9))
            if self.t + a.dur <= crit + 5 or overdue > 20:
                return a
        return None   # keep hands free for the predicted event

    def start(self, a: Action):
        it, t = a.item, self.t
        self.cook_action, self.cook_until = a, t + a.dur
        self.cook_busy_s += a.dur
        if a.kind == "load":
            it.burner = a.burner
            self.burner_busy[a.burner] = it
        if a.kind == "add":
            it.t_add = t
        if a.kind == "plate":
            it.t_plate = t
            if it.burner:
                self.burner_busy[it.burner] = None
        self.log("cook_start", action=a.kind, ticket=it.tid, dish=it.dish.name, burner=it.burner)

    def finish(self, a: Action):
        it, t = a.item, self.t
        if a.kind == "prep":
            it.prepped = True
        elif a.kind == "load":
            it.t_load = t
            if it.boils:
                it.t_boil = t + self._true_boil_time(it, it.burner)
                for s in (1, 2):
                    it.stage_times[s] = self._stage_time(it, s) + it.truth.stage_lag[s]
                it.t_boil_seen = it.t_boil + it.truth.stage_lag[3]
                it.stage_times[3] = it.t_boil_seen
                it.est = BoilEstimator(t, self.models[it.burner].prior_s(it.dish.liquid_ml),
                                       curvature=self.models[it.burner].curvature)
            else:
                it.t_cook_start = t
        elif a.kind == "add":
            it.t_cook_start = t
        elif a.kind == "plate":
            it.t_plated = t
            tk = self.tickets[it.tid]
            if all(i.t_plated is not None for i in self.items if i.tid == it.tid):
                tk["served"] = t
                self.log("served", ticket=it.tid)
        self.cook_action = None

    # -- main loop ----------------------------------------------------------
    def step(self):
        t = self.t
        while self._next_ticket < len(self.sc.tickets) and self.sc.tickets[self._next_ticket]["t"] <= t:
            tk = self.sc.tickets[self._next_ticket]
            self.tickets[tk["id"]] = {"id": tk["id"], "t": tk["t"], "items": tk["items"]}
            for j, name in enumerate(tk["items"]):
                self.items.append(Item(tk["id"], j, MENU[name], self.sc.truth[(tk["id"], j)]))
            self.log("order", ticket=tk["id"], items=tk["items"])
            if self.policy != "fifo":
                self.plan_all(f"order {tk['id']} arrived")
            self._next_ticket += 1
        for it in self.items:
            if it.est is not None and it.t_add is None:
                for s, ts in it.stage_times.items():
                    if ts <= t and s not in it.est.seen:
                        it.est.observe(s, ts)
                        if self.policy == "tempo" or s == 3:
                            self._replan_needed = True
                            names = {1: "shimmer", 2: "simmer", 3: "rolling boil"}
                            self._replan_reason = (f"camera: {it.burner} {it.dish.name} reached {names[s]}"
                                                   + (f", boil now predicted at {it.est.eta(t)['t_boil']:.0f}s" if s < 3 else ""))
                        self.log("stage", ticket=it.tid, dish=it.dish.name, burner=it.burner, stage=s,
                                 eta=round(it.est.eta(t)["t_boil"], 1))
                        if s == 3 and self.policy == "tempo":
                            self.models[it.burner].update(it.dish.liquid_ml, ts - it.t_load, it.est)
            if it.t_cook_start is not None and it.t_done is None and t >= it.t_cook_start + it.dish.cook:
                it.t_done = it.t_cook_start + it.dish.cook
            # foam surge during cooking
            if it.truth.foam_at is not None and it.t_cook_start is not None and not it.foam_handled \
                    and not it.boilover and it.t_plate is None:
                ft = it.t_cook_start + it.truth.foam_at
                if t >= ft:
                    if self.policy == "tempo" and it.truth.foam_seen:
                        it.foam_handled = True
                        self.log("auto_heat_down", ticket=it.tid, dish=it.dish.name, burner=it.burner)
                    elif self.cook_action is None:
                        it.foam_handled = True      # a free cook can react in time
                    elif t >= ft + FOAM_WINDOW_S:
                        it.boilover = True
                        self.log("boilover", ticket=it.tid, dish=it.dish.name, burner=it.burner)
        if self.cook_action is not None and t >= self.cook_until:
            self.finish(self.cook_action)
        if self.policy in {"plan_alert", "tempo"} and self._replan_needed:
            self.plan_all(self._replan_reason)
            self._replan_needed = False
        if self.cook_action is None:
            a = self.choose()
            if a:
                self.start(a)

    def run(self, max_t: float = 4 * 3600) -> dict:
        while self.t < max_t:
            self.step()
            if self._next_ticket == len(self.sc.tickets) and all("served" in tk for tk in self.tickets.values()):
                break
            self.t += 1.0
        return self.metrics()

    def metrics(self) -> dict:
        waits, leads, idle_boil, overcook = [], [], [], []
        for tk in self.tickets.values():
            leads.append(tk.get("served", self.t) - tk["t"])
        for it in self.items:
            served = self.tickets[it.tid].get("served", self.t)
            if it.t_done is not None:
                waits.append(served - it.t_done)          # finished food waiting (cooling / going soggy)
                overcook.append((it.t_plate or self.t) - it.t_done)
            if it.boils and it.t_boil is not None:
                idle_boil.append(max(0.0, (it.t_add or self.t) - it.t_boil))
        n = len(self.items)
        return {
            "policy": self.policy, "seed": self.sc.seed, "dishes": n, "tickets": len(self.tickets),
            "food_wait_mean_s": float(np.mean(waits)), "food_wait_p90_s": float(np.percentile(waits, 90)),
            "overcook_mean_s": float(np.mean(overcook)),
            "lead_mean_s": float(np.mean(leads)), "lead_p90_s": float(np.percentile(leads, 90)),
            "idle_boil_mean_s": float(np.mean(idle_boil)) if idle_boil else 0.0,
            "boilovers": int(sum(i.boilover for i in self.items)),
            "cook_util": self.cook_busy_s / max(self.t, 1),
            "makespan_s": self.t,
        }


def simulate(seed: int, policy: str, log_events: bool = False, **kw) -> tuple[dict, Kitchen]:
    k = Kitchen(make_scenario(seed, **kw), policy, log_events)
    return k.run(), k
