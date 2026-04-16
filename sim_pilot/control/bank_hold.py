from __future__ import annotations

from dataclasses import dataclass
import math

from sim_pilot.control.pid import PIDController
from sim_pilot.core.config import PIDGains
from sim_pilot.core.types import clamp


@dataclass
class BankController:
    gains: PIDGains

    def __post_init__(self) -> None:
        self._pid = PIDController(
            kp=self.gains.kp,
            ki=self.gains.ki,
            kd=self.gains.kd,
            integrator_limit=15.0,
            output_limit=(-1.0, 1.0),
        )

    def update(self, target_bank_deg: float, roll_deg: float, p_rad_s: float, dt: float) -> float:
        roll_rate_deg_s = math.degrees(p_rad_s)
        error = target_bank_deg - roll_deg
        return self._pid.update(error=error, rate_feedback=roll_rate_deg_s, dt=dt)

    def reset(self) -> None:
        self._pid.reset()


@dataclass(slots=True)
class CoordinationController:
    beta_gain: float = 0.04
    yaw_gain: float = 0.03
    bank_feedforward_gain: float = 0.015

    def update(self, target_bank_deg: float, roll_deg: float, yaw_rate_rad_s: float, beta_deg: float | None, dt: float) -> float:
        del dt
        yaw_rate_deg_s = math.degrees(yaw_rate_rad_s)
        slip_term = 0.0 if beta_deg is None else self.beta_gain * beta_deg
        rudder_cmd = ((target_bank_deg - roll_deg) * self.bank_feedforward_gain) + slip_term - (yaw_rate_deg_s * self.yaw_gain)
        return clamp(rudder_cmd, -1.0, 1.0)
