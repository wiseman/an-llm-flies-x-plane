from __future__ import annotations

from dataclasses import dataclass

from sim_pilot.core.config import TECSGains
from sim_pilot.core.types import FlightPhase, clamp


@dataclass(slots=True)
class TECSLite:
    gains: TECSGains
    integrator: float = 0.0

    def reset(self) -> None:
        self.integrator = 0.0

    def update(
        self,
        *,
        phase: FlightPhase,
        target_alt_ft: float,
        target_speed_kt: float,
        alt_ft: float,
        vs_fpm: float,
        ias_kt: float,
        dt: float,
        throttle_limit: tuple[float, float],
    ) -> tuple[float, float]:
        throttle_trim, pitch_trim = self._trim_for_phase(phase)
        total_altitude_weight, total_speed_weight, balance_altitude_weight, balance_speed_weight, min_pitch_deg, max_pitch_deg = self._phase_weights(phase)
        altitude_error_ft = target_alt_ft - alt_ft
        speed_error_kt = target_speed_kt - ias_kt

        energy_total_error = (total_altitude_weight * altitude_error_ft) + (total_speed_weight * speed_error_kt)
        self.integrator = clamp(self.integrator + (energy_total_error * dt), -80.0, 80.0)
        throttle_cmd = throttle_trim + (self.gains.kp_total * energy_total_error) + (self.gains.ki_total * self.integrator)

        energy_balance_error = (balance_altitude_weight * altitude_error_ft) - (balance_speed_weight * speed_error_kt)
        pitch_cmd = pitch_trim + (self.gains.kp_balance * energy_balance_error) + (self.gains.kd_balance * (-vs_fpm))
        return (
            clamp(pitch_cmd, min_pitch_deg, max_pitch_deg),
            clamp(throttle_cmd, throttle_limit[0], throttle_limit[1]),
        )

    @staticmethod
    def _trim_for_phase(phase: FlightPhase) -> tuple[float, float]:
        if phase in {FlightPhase.ROTATE, FlightPhase.INITIAL_CLIMB, FlightPhase.ENROUTE_CLIMB, FlightPhase.GO_AROUND}:
            return (0.85, 6.0)
        if phase is FlightPhase.CRUISE:
            return (0.58, 2.0)
        if phase is FlightPhase.DESCENT:
            return (0.35, -1.5)
        if phase in {FlightPhase.PATTERN_ENTRY, FlightPhase.DOWNWIND}:
            return (0.45, 1.0)
        if phase is FlightPhase.BASE:
            return (0.32, 0.0)
        if phase is FlightPhase.FINAL:
            return (0.28, -0.5)
        return (0.5, 1.0)

    @staticmethod
    def _phase_weights(phase: FlightPhase) -> tuple[float, float, float, float, float, float]:
        if phase is FlightPhase.FINAL:
            # balance_altitude_weight was 1.2, which gave only ~2.4°
            # of pitch-down for a 100 ft high-on-glidepath error.
            # Combined with kd_balance * -vs_fpm damping, commanded
            # pitch saturated around -2° and the aircraft descended at
            # roughly the glidepath slope — tracking parallel to the
            # glidepath but never closing the error. Bumping to 3.5
            # gives ~-6° pitch response to 100 ft of error, and the
            # wider [-10, 10] pitch envelope lets short-field visual
            # approaches sustain a steep descent when arriving high
            # on glidepath.
            return (0.004, 10.0, 3.5, 12.0, -10.0, 10.0)
        if phase is FlightPhase.BASE:
            # Same reasoning as FINAL but less aggressive, since BASE
            # is transitioning from level pattern flight to descent.
            return (0.008, 8.5, 2.5, 14.0, -8.0, 10.0)
        if phase is FlightPhase.DESCENT:
            return (0.009, 8.0, 1.0, 14.0, -6.0, 10.0)
        return (0.01, 8.0, 1.0, 15.0, -4.0, 12.0)
