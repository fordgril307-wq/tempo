"""Render the photo-realistic (simulated) evaluation set and the demo clip.

    python -m eval.make_real eval 0 4     # clips 0..3
    python -m eval.make_real demo
    python -m eval.make_real dish pasta   # one clip per boiling dish for the live board
"""
import sys
from pathlib import Path

import numpy as np

from tempo.physics import BOIL_C, heat_params, time_to
from tempo.synth_real import RealScene, render


def eval_scene(k: int) -> RealScene:
    rng = np.random.default_rng(7000 + k)
    vol = float(rng.uniform(450, 1000)); kw = float(rng.uniform(1.8, 3.0)); eff = 0.45 * float(rng.lognormal(0, 0.12))
    T0 = float(rng.uniform(10, 24))
    tb = time_to(BOIL_C, T0, *heat_params(vol, kw, eff))
    foam = k % 2 == 0
    return RealScene(name=f"r{k:02d}", seconds=min(tb + (80 if foam else 25), 720), fps=10, volume_ml=vol, burner_kw=kw,
                     eff=eff, T0=T0, stage_dT1=float(rng.normal(0, 3)), stage_dT2=float(rng.normal(0, 2)),
                     add_at=tb + 6 if foam else None, foam_after_add=float(rng.uniform(15, 35)) if foam else None,
                     seed=300 + k, tilt=float(rng.uniform(0.15, 0.4)), light_angle=float(rng.uniform(-2.8, -1.8)),
                     warm=float(rng.uniform(0.3, 1.2)))


DISHES = {   # dish -> scene; burner power matches the burner the dish lands on in the demo service
    "pasta": dict(volume_ml=1000, burner_kw=3.0, liquid="clear", addin="pasta", foam=30, seed=51),
    "egg_soup": dict(volume_ml=500, burner_kw=2.2, liquid="broth", addin="egg", foam=None, seed=52),
    "kimchi_stew": dict(volume_ml=750, burner_kw=1.7, liquid="red", addin="tofu", pot="black", foam=None, seed=53),
}


def dish_scene(name: str) -> RealScene:
    d = dict(DISHES[name])
    foam = d.pop("foam")
    tb = time_to(BOIL_C, 17, *heat_params(d["volume_ml"], d["burner_kw"], 0.45))
    add = round(tb + 6)
    return RealScene(name=f"clip_{name}", seconds=tb + (95 if foam else 60), fps=15, T0=17, add_at=add,
                     foam_after_add=foam, heat_down_at=(add + foam + 11) if foam else None, **d)


if __name__ == "__main__":
    if sys.argv[1] == "dish":
        t = render(dish_scene(sys.argv[2]), Path("data/real_dishes"))
        print(sys.argv[2], "boil", t["t_boil"], flush=True)
    elif sys.argv[1] == "demo":
        sc = RealScene(name="demo_real", seconds=370, fps=15, volume_ml=550, burner_kw=2.6, T0=17, add_at=238,
                       foam_after_add=40, heat_down_at=238 + 40 + 11, seed=42)
        t = render(sc, Path("data/real_demo"))
        print("demo boil", t["t_boil"], flush=True)
    else:
        a, b = int(sys.argv[2]), int(sys.argv[3])
        for k in range(a, b):
            t = render(eval_scene(k), Path("data/real_eval"))
            print(f"r{k:02d} boil {t['t_boil']}", flush=True)
