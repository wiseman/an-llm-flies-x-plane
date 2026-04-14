from __future__ import annotations

from dataclasses import dataclass
import math

from sim_pilot.control.pid import PIDController
from sim_pilot.core.config import PIDGains


@dataclass
class PitchController:
    gains: PIDGains

    def __post_init__(self) -> None:
        self._pid = PIDController(
            kp=self.gains.kp,
            ki=self.gains.ki,
            kd=self.gains.kd,
            integrator_limit=15.0,
            output_limit=(-1.0, 1.0),
        )

    def update(self, target_pitch_deg: float, pitch_deg: float, q_rad_s: float, dt: float) -> float:
        pitch_rate_deg_s = math.degrees(q_rad_s)
        error = target_pitch_deg - pitch_deg
        return self._pid.update(error=error, rate_feedback=pitch_rate_deg_s, dt=dt)
