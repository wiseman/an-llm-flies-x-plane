from __future__ import annotations

from dataclasses import dataclass, field
import math
import threading

from sim_pilot.control.bank_hold import BankController, CoordinationController
from sim_pilot.control.centerline_rollout import CenterlineRolloutController
from sim_pilot.control.pitch_hold import PitchController
from sim_pilot.control.tecs_lite import TECSLite
from sim_pilot.core.config import ConfigBundle
from sim_pilot.core.profiles import (
    Axis,
    GuidanceProfile,
    IdleLateralProfile,
    IdleSpeedProfile,
    IdleVerticalProfile,
    PatternFlyProfile,
    ProfileContribution,
)
from sim_pilot.core.state_estimator import estimate_aircraft_state
from sim_pilot.core.types import (
    ActuatorCommands,
    AircraftState,
    FlightPhase,
    GuidanceTargets,
    LateralMode,
    VerticalMode,
    clamp,
    wrap_degrees_180,
)
from sim_pilot.guidance.flare_profile import FlareController
from sim_pilot.guidance.runway_geometry import RunwayFrame
from sim_pilot.sim.simple_dynamics import DynamicsState


@dataclass(slots=True, frozen=True)
class StatusSnapshot:
    t_sim: float
    active_profiles: tuple[str, ...]
    phase: FlightPhase | None
    state: AircraftState
    last_commands: ActuatorCommands
    last_guidance: GuidanceTargets | None = None
    go_around_reason: str | None = None


@dataclass
class PilotCore:
    config: ConfigBundle

    def __post_init__(self) -> None:
        self.runway_frame = RunwayFrame(self.config.airport.runway)
        self.bank_controller = BankController(self.config.controllers.bank)
        self.coordination = CoordinationController()
        self.pitch_controller = PitchController(self.config.controllers.pitch)
        self.rollout_controller = CenterlineRolloutController()
        self.tecs = TECSLite(self.config.controllers.tecs)
        self.flare_controller = FlareController(self.config.flare)
        self._lock = threading.RLock()
        self.active_profiles: list[GuidanceProfile] = [
            IdleLateralProfile(),
            IdleVerticalProfile(),
            IdleSpeedProfile(default_speed_kt=self.config.performance.cruise_speed_kt),
        ]
        self.latest_snapshot: StatusSnapshot | None = None

    # ---- profile management ----

    def engage_profile(self, profile: GuidanceProfile) -> list[str]:
        """Engage a profile, auto-disengaging any conflict on owned axes.

        Returns the names of profiles that were displaced.
        """
        with self._lock:
            displaced: list[str] = []
            remaining: list[GuidanceProfile] = []
            for existing in self.active_profiles:
                if existing.owns & profile.owns:
                    displaced.append(existing.name)
                else:
                    remaining.append(existing)
            remaining.append(profile)
            self.active_profiles = remaining
            return displaced

    def engage_profiles(self, profiles: list[GuidanceProfile]) -> list[str]:
        """Atomically engage multiple profiles under a single lock.

        Each profile displaces any existing profile that owns an
        overlapping axis. Because the whole operation runs inside one
        ``with self._lock`` block, the control loop cannot observe an
        intermediate state where some of the new profiles are engaged
        and axes are orphaned — important when swapping a
        ``pattern_fly`` (which owns all three axes) for three
        single-axis holds.
        """
        with self._lock:
            all_displaced: set[str] = set()
            for profile in profiles:
                displaced: list[str] = []
                remaining: list[GuidanceProfile] = []
                for existing in self.active_profiles:
                    if existing.owns & profile.owns:
                        displaced.append(existing.name)
                    else:
                        remaining.append(existing)
                remaining.append(profile)
                self.active_profiles = remaining
                all_displaced.update(displaced)
            # Don't report a profile as "displaced" if a later
            # profile in the list is actually that same profile
            # being re-engaged.
            new_names = {p.name for p in profiles}
            return sorted(all_displaced - new_names)

    def disengage_profile(self, name: str) -> list[str]:
        """Remove profile(s) with the given name. Re-adds idle profiles for orphaned axes.

        Returns the names of idle profiles that were re-added to cover orphans.
        """
        with self._lock:
            removed = [p for p in self.active_profiles if p.name == name]
            if not removed:
                return []
            self.active_profiles = [p for p in self.active_profiles if p.name != name]
            orphaned: set[Axis] = set()
            for p in removed:
                orphaned.update(p.owns)
            covered: set[Axis] = set()
            for p in self.active_profiles:
                covered.update(p.owns)
            still_orphaned = orphaned - covered
            added: list[str] = []
            for axis in still_orphaned:
                idle = self._make_idle_profile(axis)
                self.active_profiles.append(idle)
                added.append(idle.name)
            return added

    def list_profile_names(self) -> list[str]:
        with self._lock:
            return [p.name for p in self.active_profiles]

    def find_profile(self, name: str) -> GuidanceProfile | None:
        with self._lock:
            for p in self.active_profiles:
                if p.name == name:
                    return p
            return None

    def _make_idle_profile(self, axis: Axis) -> GuidanceProfile:
        if axis is Axis.LATERAL:
            return IdleLateralProfile()
        if axis is Axis.VERTICAL:
            return IdleVerticalProfile()
        return IdleSpeedProfile(default_speed_kt=self.config.performance.cruise_speed_kt)

    # ---- per-tick update ----

    def update(self, raw_state: DynamicsState, dt: float) -> tuple[AircraftState, ActuatorCommands]:
        with self._lock:
            state = estimate_aircraft_state(raw_state, self.config, self.runway_frame, dt)
            guidance = self._compose_guidance(state, dt)
            commands = self._commands_from_guidance(state, guidance)
            go_around_reason: str | None = None
            for profile in self.active_profiles:
                if isinstance(profile, PatternFlyProfile) and profile.last_go_around_reason is not None:
                    go_around_reason = profile.last_go_around_reason
                    break
            self.latest_snapshot = StatusSnapshot(
                t_sim=state.t_sim,
                active_profiles=tuple(p.name for p in self.active_profiles),
                phase=self._current_phase(),
                state=state,
                last_commands=commands,
                last_guidance=guidance,
                go_around_reason=go_around_reason,
            )
            return state, commands

    @property
    def phase(self) -> FlightPhase:
        """Backwards-compatible accessor for the mission phase.

        Returns the current phase from the active `PatternFlyProfile` if one is
        engaged, otherwise falls back to PREFLIGHT. Kept for tests and external
        observers that reported the legacy phase field.
        """
        resolved = self._current_phase()
        return resolved if resolved is not None else FlightPhase.PREFLIGHT

    def _current_phase(self) -> FlightPhase | None:
        for profile in self.active_profiles:
            if isinstance(profile, PatternFlyProfile):
                return profile.phase
        return None

    def _compose_guidance(self, state: AircraftState, dt: float) -> GuidanceTargets:
        contributions: list[tuple[GuidanceProfile, ProfileContribution]] = [
            (profile, profile.contribute(state, dt, self)) for profile in self.active_profiles
        ]
        targets = GuidanceTargets(
            lateral_mode=LateralMode.BANK_HOLD,
            vertical_mode=VerticalMode.PITCH_HOLD,
            target_bank_deg=0.0,
            target_pitch_deg=0.0,
            throttle_limit=(0.0, 0.0),
        )
        for profile, c in contributions:
            if Axis.LATERAL in profile.owns:
                if c.lateral_mode is not None:
                    targets.lateral_mode = c.lateral_mode
                if c.target_bank_deg is not None:
                    targets.target_bank_deg = c.target_bank_deg
                targets.target_heading_deg = c.target_heading_deg
                targets.target_track_deg = c.target_track_deg
                targets.target_path = c.target_path
                targets.target_waypoint = c.target_waypoint
            if Axis.VERTICAL in profile.owns:
                if c.vertical_mode is not None:
                    targets.vertical_mode = c.vertical_mode
                targets.target_altitude_ft = c.target_altitude_ft
                if c.target_pitch_deg is not None:
                    targets.target_pitch_deg = c.target_pitch_deg
                targets.glidepath = c.glidepath
                if c.throttle_limit is not None:
                    targets.throttle_limit = c.throttle_limit
                targets.flaps_cmd = c.flaps_cmd
                targets.gear_down = c.gear_down
                if c.brakes is not None:
                    targets.brakes = c.brakes
                if c.tecs_phase_override is not None:
                    targets.tecs_phase_override = c.tecs_phase_override
            if Axis.SPEED in profile.owns:
                if c.target_speed_kt is not None:
                    targets.target_speed_kt = c.target_speed_kt
        return targets

    def _commands_from_guidance(self, state: AircraftState, guidance: GuidanceTargets) -> ActuatorCommands:
        if guidance.lateral_mode is LateralMode.ROLLOUT_CENTERLINE:
            track_reference_deg = state.heading_deg if state.on_ground else state.track_deg
            track_error_deg = wrap_degrees_180(self.runway_frame.runway.course_deg - track_reference_deg)
            rudder = self.rollout_controller.update(
                centerline_error_ft=state.centerline_error_ft or 0.0,
                track_error_deg=track_error_deg,
                yaw_rate_deg_s=math.degrees(state.r_rad_s),
                gs_kt=state.gs_kt,
                dt=state.dt,
            )
            aileron = self.bank_controller.update(0.0, state.roll_deg, state.p_rad_s, state.dt)
        else:
            target_bank_deg = guidance.target_bank_deg or 0.0
            aileron = self.bank_controller.update(target_bank_deg, state.roll_deg, state.p_rad_s, state.dt)
            rudder = self.coordination.update(target_bank_deg, state.roll_deg, state.r_rad_s, None, state.dt)

        throttle_limit = guidance.throttle_limit or (0.0, 1.0)
        if guidance.vertical_mode in {VerticalMode.TECS, VerticalMode.GLIDEPATH_TRACK}:
            tecs_phase = guidance.tecs_phase_override or self._current_phase() or FlightPhase.CRUISE
            pitch_cmd_deg, throttle_cmd = self.tecs.update(
                phase=tecs_phase,
                target_alt_ft=guidance.target_altitude_ft or state.alt_msl_ft,
                target_speed_kt=guidance.target_speed_kt or state.ias_kt,
                alt_ft=state.alt_msl_ft,
                vs_fpm=state.vs_fpm,
                ias_kt=state.ias_kt,
                dt=state.dt,
                throttle_limit=throttle_limit,
            )
        elif guidance.vertical_mode is VerticalMode.FLARE_TRACK:
            pitch_cmd_deg = self.flare_controller.target_pitch_deg(
                alt_agl_ft=state.alt_agl_ft,
                sink_rate_fpm=state.vs_fpm,
                ias_error_kt=(guidance.target_speed_kt or state.ias_kt) - state.ias_kt,
            )
            throttle_cmd = 0.0
        else:
            pitch_cmd_deg = guidance.target_pitch_deg or 0.0
            throttle_cmd = throttle_limit[1]

        elevator = self.pitch_controller.update(pitch_cmd_deg, state.pitch_deg, state.q_rad_s, state.dt)
        return ActuatorCommands(
            aileron=clamp(aileron, -1.0, 1.0),
            elevator=clamp(elevator, -1.0, 1.0),
            rudder=clamp(rudder, -1.0, 1.0),
            throttle=clamp(throttle_cmd, 0.0, 1.0),
            flaps=guidance.flaps_cmd,
            gear_down=True,
            brakes=guidance.brakes,
        )
