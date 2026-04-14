from __future__ import annotations

from dataclasses import dataclass

from sim_pilot.core.types import clamp


@dataclass(slots=True)
class CenterlineRolloutController:
    centerline_gain: float = 0.025
    track_gain: float = 0.04

    def update(self, centerline_error_ft: float, track_error_deg: float, gs_kt: float) -> float:
        speed_scale = 1.0 if gs_kt >= 40.0 else 1.4
        rudder_cmd = -((centerline_error_ft * self.centerline_gain * speed_scale) + (track_error_deg * self.track_gain))
        return clamp(rudder_cmd, -1.0, 1.0)
