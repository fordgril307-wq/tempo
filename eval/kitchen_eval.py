"""Service-level comparison of order policies (common random numbers).

    python -m eval.kitchen_eval [N]

Two conditions: 'calm' (tickets far apart: isolates table sync) and 'rush'
(a ticket every 5 min on average with a 4-burner, 1-cook line).
"""
import json
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from tempo.kitchen import simulate

POLICIES = ["fifo", "plan_fixed", "plan_alert", "tempo"]
KEYS = ["food_wait_mean_s", "food_wait_p90_s", "lead_mean_s", "lead_p90_s", "idle_boil_mean_s", "boilovers",
        "overcook_mean_s", "dishes"]
CONDITIONS = {"calm": 5000.0, "rush": 300.0}


def one(args):
    seed, policy, gap = args
    m, _ = simulate(seed, policy, mean_gap_s=gap)
    return seed, policy, gap, m


def main(n: int):
    out = {"n_services": n, "conditions": {}}
    jobs = [(s, p, g) for g in CONDITIONS.values() for s in range(n) for p in POLICIES]
    with ProcessPoolExecutor() as ex:
        res = list(ex.map(one, jobs, chunksize=8))
    for name, gap in CONDITIONS.items():
        cond = {}
        for p in POLICIES:
            ms = [m for s, pp, g, m in res if pp == p and g == gap]
            d = {k: round(float(np.mean([m[k] for m in ms])), 1) for k in KEYS}
            d["boilovers_per_100_dishes"] = round(100 * sum(m["boilovers"] for m in ms) / sum(m["dishes"] for m in ms), 2)
            cond[p] = d
        base = cond["fifo"]
        for p in POLICIES:
            cond[p]["food_wait_vs_fifo_pct"] = round(100 * (cond[p]["food_wait_mean_s"] / base["food_wait_mean_s"] - 1), 1)
            cond[p]["lead_vs_fifo_pct"] = round(100 * (cond[p]["lead_mean_s"] / base["lead_mean_s"] - 1), 1)
        # paired bootstrap CI for tempo vs plan_fixed on food wait
        a = np.array([m["food_wait_mean_s"] for s, pp, g, m in sorted(res, key=lambda x: x[0]) if pp == "tempo" and g == gap])
        b = np.array([m["food_wait_mean_s"] for s, pp, g, m in sorted(res, key=lambda x: x[0]) if pp == "plan_fixed" and g == gap])
        rng = np.random.default_rng(0)
        boots = [np.mean((a - b)[rng.integers(0, len(a), len(a))]) for _ in range(2000)]
        cond["tempo_minus_fixed_food_wait_s_95ci"] = [round(float(np.percentile(boots, 2.5)), 1),
                                                      round(float(np.percentile(boots, 97.5)), 1)]
        out["conditions"][name] = cond
        print(name, json.dumps(cond, indent=1))
    out["summary"] = {c: {p: {k: v[k] for k in ("food_wait_mean_s", "food_wait_p90_s", "lead_mean_s", "boilovers_per_100_dishes")}
                          for p, v in d.items() if isinstance(v, dict)} for c, d in out["conditions"].items()}
    open("eval/kitchen_results.json", "w").write(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 200)
