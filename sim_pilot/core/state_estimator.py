from __future__ import annotations

from sim_pilot.core.config import ConfigBundle
from sim_pilot.core.types import AircraftState, KT_TO_FPS, vector_to_heading
from sim_pilot.guidance.runway_geometry import RunwayFrame
from sim_pilot.sim.simple_dynamics import DynamicsState


def estimate_aircraft_state(
    raw_state: DynamicsState,
    config: ConfigBundle,
    runway_frame: RunwayFrame,
    dt: float,
) -> AircraftState:
    runway_position_ft = runway_frame.to_runway_frame(raw_state.position_ft)
    ground_speed_kt = raw_state.ground_velocity_ft_s.length() / KT_TO_FPS
    track_deg = vector_to_heading(raw_state.ground_velocity_ft_s) if ground_speed_kt > 1.0 else raw_state.heading_deg
    runway_dist_remaining_ft = None
    if 0.0 <= runway_position_ft.x <= runway_frame.runway.length_ft:
        runway_dist_remaining_ft = runway_frame.runway.length_ft - runway_position_ft.x

    distance_to_touchdown_ft = raw_state.position_ft.distance_to(runway_frame.touchdown_point_ft())
    return AircraftState(
        t_sim=raw_state.time_s,
        dt=dt,
        position_ft=raw_state.position_ft,
        alt_msl_ft=raw_state.altitude_ft,
        alt_agl_ft=max(0.0, raw_state.altitude_ft - config.airport.field_elevation_ft),
        pitch_deg=raw_state.pitch_deg,
        roll_deg=raw_state.roll_deg,
        heading_deg=raw_state.heading_deg,
        track_deg=track_deg,
        p_rad_s=raw_state.p_rad_s,
        q_rad_s=raw_state.q_rad_s,
        r_rad_s=raw_state.r_rad_s,
        ias_kt=raw_state.ias_kt,
        tas_kt=raw_state.ias_kt,
        gs_kt=ground_speed_kt,
        vs_fpm=raw_state.vertical_speed_ft_s * 60.0,
        ground_velocity_ft_s=raw_state.ground_velocity_ft_s,
        flap_index=raw_state.flap_index,
        gear_down=raw_state.gear_down,
        on_ground=raw_state.on_ground,
        throttle_pos=raw_state.throttle_pos,
        runway_id=runway_frame.runway.id,
        runway_dist_remaining_ft=runway_dist_remaining_ft,
        runway_x_ft=runway_position_ft.x,
        runway_y_ft=runway_position_ft.y,
        centerline_error_ft=runway_position_ft.y,
        threshold_abeam=abs(runway_position_ft.x) <= config.pattern.abeam_window_ft,
        distance_to_touchdown_ft=distance_to_touchdown_ft,
        stall_margin=raw_state.ias_kt / config.performance.vso_landing_kt,
    )
