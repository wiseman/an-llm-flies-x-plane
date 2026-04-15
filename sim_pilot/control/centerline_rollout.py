from __future__ import annotations

from dataclasses import dataclass

from sim_pilot.core.types import clamp


@dataclass(slots=True)
class CenterlineRolloutController:
    """Takeoff/rollout rudder controller.

    Sign conventions:
      centerline_error_ft  — positive when aircraft is right of centerline
      track_error_deg      — positive when runway course is greater than the
                             aircraft's current track/heading (i.e. the nose is
                             pointing left of the runway; we want right rudder)
      yaw_rate_deg_s       — positive when yawing right (clockwise looking down)
      rudder output        — positive is right rudder

    Tuning rationale: the C172 at takeoff power has a steady left-yawing
    torque (P-factor + slipstream + torque). A pure P controller can't
    compensate because it needs some steady-state error to produce any
    output; so we need an integrator. The P gain is kept small so it does
    not saturate at small/moderate errors (which causes oscillation), the
    integrator accumulates steady-state bias, and yaw-rate damping is
    aggressive enough to actually arrest transient swings before they
    overshoot. Reset the integrator between flights with ``reset()``.
    """

    centerline_gain: float = 0.015
    track_gain: float = 0.012
    track_integrator_gain: float = 0.025
    yaw_rate_gain: float = 0.20
    integrator_limit_deg_s: float = 20.0
    _integrator: float = 0.0

    def reset(self) -> None:
        self._integrator = 0.0

    def update(
        self,
        *,
        centerline_error_ft: float,
        track_error_deg: float,
        yaw_rate_deg_s: float,
        gs_kt: float,
        dt: float,
    ) -> float:
        speed_scale = 1.0 if gs_kt >= 40.0 else 1.4
        centerline_term = -centerline_error_ft * self.centerline_gain * speed_scale
        track_term = track_error_deg * self.track_gain
        self._integrator = clamp(
            self._integrator + (track_error_deg * dt),
            -self.integrator_limit_deg_s,
            self.integrator_limit_deg_s,
        )
        integral_term = self._integrator * self.track_integrator_gain
        damping_term = -yaw_rate_deg_s * self.yaw_rate_gain
        rudder_cmd = centerline_term + track_term + integral_term + damping_term
        return clamp(rudder_cmd, -1.0, 1.0)
