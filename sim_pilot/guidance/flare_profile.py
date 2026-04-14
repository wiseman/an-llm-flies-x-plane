from __future__ import annotations

from dataclasses import dataclass

from sim_pilot.core.config import FlareConfig
from sim_pilot.core.types import clamp


@dataclass(slots=True)
class FlareController:
    config: FlareConfig

    def target_pitch_deg(self, alt_agl_ft: float, sink_rate_fpm: float, ias_error_kt: float) -> float:
        ratio = clamp((self.config.flare_start_ft - alt_agl_ft) / self.config.flare_start_ft, 0.0, 1.0)
        sink_bias = clamp(((-sink_rate_fpm) - 200.0) / 250.0, 0.0, 2.0)
        speed_bias = clamp(-ias_error_kt * 0.12, -1.5, 1.0)
        pitch_deg = 2.5 + (ratio * 4.5) + sink_bias + speed_bias
        return clamp(pitch_deg, 1.0, self.config.max_flare_pitch_deg)
