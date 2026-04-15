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
    """

    centerline_gain: float = 0.025
    track_gain: float = 0.04
    yaw_rate_gain: float = 0.05

    def update(
        self,
        *,
        centerline_error_ft: float,
        track_error_deg: float,
        yaw_rate_deg_s: float,
        gs_kt: float,
    ) -> float:
        speed_scale = 1.0 if gs_kt >= 40.0 else 1.4
        centerline_term = -centerline_error_ft * self.centerline_gain * speed_scale
        track_term = track_error_deg * self.track_gain
        damping_term = -yaw_rate_deg_s * self.yaw_rate_gain
        rudder_cmd = centerline_term + track_term + damping_term
        return clamp(rudder_cmd, -1.0, 1.0)
