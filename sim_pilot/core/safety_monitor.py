from __future__ import annotations

from dataclasses import dataclass

from sim_pilot.core.config import ConfigBundle
from sim_pilot.core.types import AircraftState, FlightPhase, GuidanceTargets, clamp


@dataclass(slots=True, frozen=True)
class SafetyStatus:
    request_go_around: bool
    reason: str | None
    bank_limit_deg: float


@dataclass(slots=True)
class SafetyMonitor:
    config: ConfigBundle

    def evaluate(self, state: AircraftState, phase: FlightPhase) -> SafetyStatus:
        bank_limit_deg = self.bank_limit_deg(phase)
        if phase is FlightPhase.FINAL and state.alt_agl_ft < 200.0 and abs(state.centerline_error_ft or 0.0) > self.config.limits.unstable_centerline_error_ft:
            return SafetyStatus(True, "unstable_lateral", bank_limit_deg)
        if phase in {FlightPhase.FINAL, FlightPhase.ROUNDOUT} and state.alt_agl_ft < 200.0 and state.vs_fpm < -self.config.limits.unstable_sink_rate_fpm:
            return SafetyStatus(True, "unstable_vertical", bank_limit_deg)
        if phase is FlightPhase.FINAL and state.alt_agl_ft < 50.0 and state.stall_margin < max(1.1, self.config.limits.min_stall_margin - 0.1):
            return SafetyStatus(True, "low_energy", bank_limit_deg)
        return SafetyStatus(False, None, bank_limit_deg)

    def apply_limits(self, guidance: GuidanceTargets, phase: FlightPhase) -> GuidanceTargets:
        if guidance.target_bank_deg is not None:
            guidance.target_bank_deg = clamp(guidance.target_bank_deg, -self.bank_limit_deg(phase), self.bank_limit_deg(phase))
        if guidance.target_pitch_deg is not None:
            guidance.target_pitch_deg = clamp(
                guidance.target_pitch_deg,
                -self.config.limits.max_pitch_down_deg,
                self.config.limits.max_pitch_up_deg,
            )
        return guidance

    def bank_limit_deg(self, phase: FlightPhase) -> float:
        if phase in {FlightPhase.FINAL, FlightPhase.ROUNDOUT, FlightPhase.FLARE, FlightPhase.ROLLOUT}:
            return self.config.limits.max_bank_final_deg
        if phase in {FlightPhase.PATTERN_ENTRY, FlightPhase.DOWNWIND, FlightPhase.BASE}:
            return self.config.limits.max_bank_pattern_deg
        return self.config.limits.max_bank_enroute_deg
