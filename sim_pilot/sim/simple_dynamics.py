from __future__ import annotations

from dataclasses import dataclass, field
import math

from sim_pilot.core.config import ConfigBundle
from sim_pilot.core.types import ActuatorCommands, G_FT_S2, KT_TO_FPS, Vec2, clamp, heading_to_vector, wrap_degrees_180, wrap_degrees_360
from sim_pilot.guidance.runway_geometry import RunwayFrame


@dataclass(slots=True)
class DynamicsState:
    position_ft: Vec2
    altitude_ft: float
    heading_deg: float
    roll_deg: float
    pitch_deg: float
    ias_kt: float
    throttle_pos: float
    on_ground: bool
    time_s: float = 0.0
    p_rad_s: float = 0.0
    q_rad_s: float = 0.0
    r_rad_s: float = 0.0
    ground_velocity_ft_s: Vec2 = field(default_factory=lambda: Vec2(0.0, 0.0))
    vertical_speed_ft_s: float = 0.0
    flap_index: int = 0
    gear_down: bool = True


@dataclass
class SimpleAircraftModel:
    config: ConfigBundle
    wind_vector_kt: Vec2 = field(default_factory=lambda: Vec2(0.0, 0.0))

    def __post_init__(self) -> None:
        self.runway_frame = RunwayFrame(self.config.airport.runway)
        self.wind_vector_ft_s = self.wind_vector_kt * KT_TO_FPS

    def initial_state(self) -> DynamicsState:
        return DynamicsState(
            position_ft=self.runway_frame.runway.threshold_ft,
            altitude_ft=self.config.airport.field_elevation_ft,
            heading_deg=self.runway_frame.runway.course_deg,
            roll_deg=0.0,
            pitch_deg=0.0,
            ias_kt=0.0,
            throttle_pos=0.0,
            on_ground=True,
        )

    def step(self, state: DynamicsState, commands: ActuatorCommands, dt: float) -> DynamicsState:
        target_throttle = clamp(commands.throttle, 0.0, 1.0)
        state.throttle_pos += clamp(target_throttle - state.throttle_pos, -1.2 * dt, 1.2 * dt)

        roll_rate_deg_s = math.degrees(state.p_rad_s)
        roll_accel_deg_s2 = (72.0 * commands.aileron) - (3.0 * roll_rate_deg_s) - (0.6 * state.roll_deg)
        if state.on_ground:
            roll_accel_deg_s2 -= 1.5 * state.roll_deg
        roll_rate_deg_s += roll_accel_deg_s2 * dt
        state.roll_deg = clamp(state.roll_deg + (roll_rate_deg_s * dt), -45.0, 45.0)
        state.p_rad_s = math.radians(roll_rate_deg_s)

        pitch_rate_deg_s = math.degrees(state.q_rad_s)
        pitch_authority = 42.0 if not state.on_ground else 18.0
        pitch_accel_deg_s2 = (pitch_authority * commands.elevator) - (2.5 * pitch_rate_deg_s) - (0.8 * state.pitch_deg)
        pitch_rate_deg_s += pitch_accel_deg_s2 * dt
        pitch_limit_up = 15.0 if not state.on_ground else 12.0
        state.pitch_deg = clamp(state.pitch_deg + (pitch_rate_deg_s * dt), -10.0, pitch_limit_up)
        state.q_rad_s = math.radians(pitch_rate_deg_s)

        thrust_term = 10.5 * state.throttle_pos
        drag_term = (0.085 * state.ias_kt) + (0.025 * abs(state.roll_deg)) + (0.1 * max(state.pitch_deg, 0.0))
        brake_term = commands.brakes * 24.0
        state.ias_kt = max(0.0, state.ias_kt + ((thrust_term - drag_term - brake_term) * dt))
        airspeed_ft_s = max(1.0, state.ias_kt * KT_TO_FPS)

        if state.on_ground:
            yaw_rate_deg_s = math.degrees(state.r_rad_s)
            runway_error_deg = wrap_degrees_180(self.runway_frame.runway.course_deg - state.heading_deg)
            yaw_accel_deg_s2 = (16.0 * commands.rudder) - (3.0 * yaw_rate_deg_s) + (0.8 * runway_error_deg)
            yaw_rate_deg_s += yaw_accel_deg_s2 * dt
            state.vertical_speed_ft_s = 0.0
            state.altitude_ft = self.config.airport.field_elevation_ft
            if state.ias_kt >= self.config.performance.vr_kt and state.pitch_deg >= 4.0 and state.throttle_pos >= 0.7:
                state.on_ground = False
                state.altitude_ft += 1.0
        else:
            desired_turn_rate_deg_s = math.degrees((G_FT_S2 * math.tan(math.radians(state.roll_deg))) / max(airspeed_ft_s, 80.0))
            yaw_rate_deg_s = math.degrees(state.r_rad_s)
            yaw_rate_deg_s += (desired_turn_rate_deg_s - yaw_rate_deg_s) * min(1.0, 2.0 * dt)
            climb_efficiency = clamp((state.ias_kt - (self.config.performance.vr_kt - 5.0)) / 20.0, 0.0, 1.2)
            flight_path_deg = state.pitch_deg - 2.0
            state.vertical_speed_ft_s = (airspeed_ft_s * math.sin(math.radians(flight_path_deg)) * climb_efficiency) - (max(0.0, 0.25 - state.throttle_pos) * 12.0)
            state.altitude_ft += state.vertical_speed_ft_s * dt
            if state.altitude_ft <= self.config.airport.field_elevation_ft:
                state.altitude_ft = self.config.airport.field_elevation_ft
                state.on_ground = True
                state.vertical_speed_ft_s = 0.0
                state.pitch_deg = min(state.pitch_deg, 4.0)
                yaw_rate_deg_s *= 0.4

        state.r_rad_s = math.radians(yaw_rate_deg_s)
        state.heading_deg = wrap_degrees_360(state.heading_deg + (yaw_rate_deg_s * dt))

        if state.on_ground:
            ground_speed_ft_s = state.ias_kt * KT_TO_FPS
            velocity_ft_s = heading_to_vector(state.heading_deg, ground_speed_ft_s)
        else:
            air_velocity_ft_s = heading_to_vector(state.heading_deg, airspeed_ft_s)
            velocity_ft_s = air_velocity_ft_s + self.wind_vector_ft_s

        state.ground_velocity_ft_s = velocity_ft_s
        state.position_ft = state.position_ft + (velocity_ft_s * dt)
        state.time_s += dt
        if commands.flaps is not None:
            state.flap_index = commands.flaps
        if commands.gear_down is not None:
            state.gear_down = commands.gear_down
        return state
