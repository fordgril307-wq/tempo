"""Physics-informed boil prediction.

A pot of liquid on a burner heats roughly as a first-order system
(Newton's law with constant input power):

    C dT/dt = P_eff - h (T - T_env)   =>   T(t) = T_inf - (T_inf - T0) exp(-t / tau)

Temperature is invisible to an RGB camera, but the *surface* goes through
visible stages at roughly known temperatures: still -> shimmer / first
bubbles on the bottom (~72 C) -> simmer, bubbles rising (~90 C) -> rolling
boil (~99.5 C). The vision module reports the moment each stage first
appears; the estimator turns those timestamps into a temperature trajectory
and extrapolates the time the pot will reach a rolling boil.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

BOIL_C = 99.5
STAGE_NAMES = ["still", "shimmer", "simmer", "rolling_boil"]
STAGE_TEMP = {1: 72.0, 2: 90.0, 3: BOIL_C}
WATER_J_PER_KG_K = 4186.0
POT_J_PER_K = 600.0
T_ENV = 24.0
LOSS_W_PER_K = 4.0
TAP_C = 17.0


def heat_params(volume_ml: float, burner_kw: float, efficiency: float = 0.45) -> tuple[float, float]:
    """(T_inf, tau) of the first-order heating model."""
    C = volume_ml / 1000.0 * WATER_J_PER_KG_K + POT_J_PER_K
    P = burner_kw * 1000.0 * efficiency
    return T_ENV + P / LOSS_W_PER_K, C / LOSS_W_PER_K


def temp_at(t: float, T0: float, T_inf: float, tau: float) -> float:
    return T_inf - (T_inf - T0) * math.exp(-max(t, 0.0) / tau)


def time_to(T_target: float, T0: float, T_inf: float, tau: float) -> float:
    if T_target <= T0:
        return 0.0
    if T_target >= T_inf:
        return math.inf
    return -tau * math.log((T_inf - T_target) / (T_inf - T0))


def nominal_boil_s(volume_ml: float, burner_kw: float) -> float:
    T_inf, tau = heat_params(volume_ml, burner_kw)
    return time_to(BOIL_C, TAP_C, T_inf, tau)


@dataclass
class BoilEstimator:
    """Online time-to-boil estimate for one pot, from visual stage events.

    prior_s     : expected heat-up time before any visual evidence (from the
                  burner's calibrated model)
    curvature   : total-time correction for extrapolating a straight line
                  through a slightly concave heating curve; learned per burner
    """
    t_start: float
    prior_s: float
    curvature: float = 1.05
    seen: dict = field(default_factory=dict)  # stage -> first time seen
    stage_temp: dict = field(default_factory=lambda: dict(STAGE_TEMP))  # calibrated per camera setup

    def observe(self, stage: int, t: float) -> None:
        for s in range(1, stage + 1):          # seeing a later stage implies the earlier ones
            self.seen.setdefault(s, t)

    @property
    def stage(self) -> int:
        return max(self.seen) if self.seen else 0

    def eta(self, now: float) -> dict:
        """Predicted absolute boil time, an uncertainty half-width and the basis."""
        if 3 in self.seen:
            return {"t_boil": self.seen[3], "pm": 0.0, "basis": "observed"}
        if self.seen:
            # One ratio fit per observed stage, anchored at the start (tap
            # temperature): t_boil = t0 + (t_k - t0) * (B - T0) / (T_k - T0).
            # Later stages sit closer to the boil and get more weight.
            fits, ws = [], []
            for s, t_k in self.seen.items():
                frac = (self.stage_temp[s] - TAP_C) / (BOIL_C - TAP_C)
                if frac <= 0.05:
                    continue
                fits.append((t_k - self.t_start) / frac)
                ws.append(frac ** 2)
            t_b = self.t_start + float(np.average(fits, weights=ws)) * self.curvature
            rem = max(t_b - max(self.seen.values()), 0.0)
            pm = 0.18 * rem + 4
            basis = "two-stage fit" if len(fits) > 1 else "one-stage fit"
        else:
            t_b, pm = self.t_start + self.prior_s, 0.2 * self.prior_s
            basis = "burner prior"
        # Evidence that the pot is *not* boiling yet bounds the estimate from below.
        if t_b < now + 2:
            t_b, pm = now + 2, max(pm, 5.0)
        return {"t_boil": t_b, "pm": pm, "basis": basis}


@dataclass
class BurnerModel:
    """Per-burner calibration learned online from finished heat-ups."""
    name: str
    kw: float
    gain: float = 1.0          # actual / nominal heat-up time
    curvature: float = 1.05
    n: int = 0

    def prior_s(self, volume_ml: float) -> float:
        return nominal_boil_s(volume_ml, self.kw) * self.gain

    def update(self, volume_ml: float, actual_s: float, est: BoilEstimator) -> None:
        a = 0.35 if self.n < 3 else 0.2
        self.gain = (1 - a) * self.gain + a * actual_s / nominal_boil_s(volume_ml, self.kw)
        if 1 in est.seen:
            frac = (est.stage_temp[1] - TAP_C) / (BOIL_C - TAP_C)
            lin = (est.seen[1] - est.t_start) / frac
            if lin > 5:
                self.curvature = (1 - a) * self.curvature + a * min(max(actual_s / lin, 0.85), 1.4)
        self.n += 1
