from __future__ import annotations

from dataclasses import dataclass, field
import dataclasses

from sim_pilot.core.config import ConfigBundle
from sim_pilot.core.mission_manager import PilotCore
from sim_pilot.core.types import FlightPhase, Vec2
from sim_pilot.guidance.runway_geometry import RunwayFrame
from sim_pilot.sim.simple_dynamics import SimpleAircraftModel


@dataclass(slots=True)
class ScenarioResult:
    success: bool
    final_phase: FlightPhase
    duration_s: float
    touchdown_runway_x_ft: float | None
    touchdown_centerline_ft: float | None
    touchdown_sink_fpm: float | None
    max_final_bank_deg: float
    phases_seen: tuple[FlightPhase, ...]
    history: tuple[object, ...] = field(default_factory=tuple)


@dataclass(slots=True)
class ScenarioRunner:
    config: ConfigBundle
    wind_vector_kt: Vec2 = field(default_factory=lambda: Vec2(0.0, 0.0))
    dt: float = 0.2
    max_time_s: float = 1800.0

    def run(self) -> ScenarioResult:
        model = SimpleAircraftModel(self.config, self.wind_vector_kt)
        pilot = PilotCore(self.config)
        runway_frame = RunwayFrame(self.config.airport.runway)
        raw_state = model.initial_state()

        history: list[object] = []
        phases_seen: list[FlightPhase] = []
        touchdown_runway_x_ft: float | None = None
        touchdown_centerline_ft: float | None = None
        touchdown_sink_fpm: float | None = None
        max_final_bank_deg = 0.0
        prev_airborne = False

        while raw_state.time_s <= self.max_time_s:
            estimated_state, commands = pilot.update(raw_state, self.dt)
            history.append(estimated_state)
            if not phases_seen or phases_seen[-1] is not pilot.phase:
                phases_seen.append(pilot.phase)
            if pilot.phase in {FlightPhase.FINAL, FlightPhase.ROUNDOUT, FlightPhase.FLARE}:
                max_final_bank_deg = max(max_final_bank_deg, abs(estimated_state.roll_deg))

            pre_step_state = dataclasses.replace(raw_state)
            raw_state = model.step(raw_state, commands, self.dt)

            if prev_airborne and raw_state.on_ground and touchdown_runway_x_ft is None:
                touchdown_position_ft = runway_frame.to_runway_frame(raw_state.position_ft)
                touchdown_runway_x_ft = touchdown_position_ft.x
                touchdown_centerline_ft = touchdown_position_ft.y
                touchdown_sink_fpm = pre_step_state.vertical_speed_ft_s * 60.0
            prev_airborne = not raw_state.on_ground

            if pilot.phase is FlightPhase.TAXI_CLEAR:
                break

        touchdown_in_zone = touchdown_runway_x_ft is not None and 0.0 <= touchdown_runway_x_ft <= (self.config.airport.runway.length_ft / 3.0)
        centerline_ok = touchdown_centerline_ft is not None and abs(touchdown_centerline_ft) < 20.0
        sink_ok = touchdown_sink_fpm is not None and abs(touchdown_sink_fpm) < 350.0
        bank_ok = max_final_bank_deg <= self.config.limits.max_bank_final_deg + 0.5
        phase_ok = pilot.phase in {FlightPhase.ROLLOUT, FlightPhase.TAXI_CLEAR}

        return ScenarioResult(
            success=bool(touchdown_in_zone and centerline_ok and sink_ok and bank_ok and phase_ok),
            final_phase=pilot.phase,
            duration_s=raw_state.time_s,
            touchdown_runway_x_ft=touchdown_runway_x_ft,
            touchdown_centerline_ft=touchdown_centerline_ft,
            touchdown_sink_fpm=touchdown_sink_fpm,
            max_final_bank_deg=max_final_bank_deg,
            phases_seen=tuple(phases_seen),
            history=tuple(history),
        )
