"""Render held-out synthetic pots with varied physics and camera conditions."""
import sys
from pathlib import Path
import numpy as np
from tempo.synth_pot import PotScene, render
from tempo.physics import heat_params, time_to, BOIL_C

out = Path("data/heldout")
rng = np.random.default_rng(2026)
for k in range(int(sys.argv[1]) if len(sys.argv) > 1 else 8):
    vol = float(rng.uniform(450, 1100)); kw = float(rng.uniform(1.6, 3.0)); eff = 0.45 * float(rng.lognormal(0, 0.12))
    T0 = float(rng.uniform(10, 24))
    T_inf, tau = heat_params(vol, kw, eff)
    tb = time_to(BOIL_C, T0, T_inf, tau)
    foam = k % 2 == 0
    sc = PotScene(name=f"h{k:02d}", seconds=min(tb + (75 if foam else 25), 700), fps=10, volume_ml=vol, burner_kw=kw,
                  eff=eff, T0=T0, stage_dT1=float(rng.normal(0, 3)), stage_dT2=float(rng.normal(0, 2)),
                  add_at=tb + 8 if foam else None, foam_after_add=float(rng.uniform(15, 35)) if foam else None,
                  seed=100 + k, noise=float(rng.uniform(2, 6)),
                  pot_offset=(int(rng.integers(-40, 40)), int(rng.integers(-20, 20))))
    t = render(sc, out)
    print(sc.name, round(vol), round(kw, 2), round(T0, 1), "boil", t["t_boil"], flush=True)
