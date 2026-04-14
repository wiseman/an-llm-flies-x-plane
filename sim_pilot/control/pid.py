from __future__ import annotations

from dataclasses import dataclass

from sim_pilot.core.types import clamp


@dataclass(slots=True)
class PIDController:
    kp: float
    ki: float
    kd: float
    integrator_limit: float = 1.0
    output_limit: tuple[float, float] = (-1.0, 1.0)
    integrator: float = 0.0

    def reset(self) -> None:
        self.integrator = 0.0

    def update(self, error: float, rate_feedback: float, dt: float) -> float:
        if dt <= 0.0:
            return 0.0

        self.integrator = clamp(
            self.integrator + (error * dt),
            -self.integrator_limit,
            self.integrator_limit,
        )
        output = (self.kp * error) + (self.ki * self.integrator) - (self.kd * rate_feedback)
        low, high = self.output_limit
        clamped = clamp(output, low, high)

        if clamped != output and ((clamped == high and error > 0.0) or (clamped == low and error < 0.0)):
            self.integrator = clamp(
                self.integrator - (error * dt),
                -self.integrator_limit,
                self.integrator_limit,
            )
            output = (self.kp * error) + (self.ki * self.integrator) - (self.kd * rate_feedback)
            clamped = clamp(output, low, high)
        return clamped
