from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class DatarefSpec:
    name: str
    index: int | None = None


LATITUDE_DEG = DatarefSpec("sim/flightmodel/position/latitude")
LONGITUDE_DEG = DatarefSpec("sim/flightmodel/position/longitude")
ELEVATION_M = DatarefSpec("sim/flightmodel/position/elevation")
PITCH_DEG = DatarefSpec("sim/flightmodel/position/theta")
ROLL_DEG = DatarefSpec("sim/flightmodel/position/phi")
HEADING_DEG = DatarefSpec("sim/flightmodel/position/psi")
IAS_KT = DatarefSpec("sim/cockpit2/gauges/indicators/airspeed_kts_pilot")
LOCAL_VX_M_S = DatarefSpec("sim/flightmodel/position/local_vx")
LOCAL_VZ_M_S = DatarefSpec("sim/flightmodel/position/local_vz")
VS_FPM = DatarefSpec("sim/flightmodel/position/vh_ind_fpm")
P_DEG_S = DatarefSpec("sim/flightmodel/position/P")
Q_DEG_S = DatarefSpec("sim/flightmodel/position/Q")
R_DEG_S = DatarefSpec("sim/flightmodel/position/R")
THROTTLE_ALL = DatarefSpec("sim/cockpit2/engine/actuators/throttle_ratio_all")
FLAP_HANDLE_DEPLOY_RATIO = DatarefSpec("sim/cockpit2/controls/flap_handle_deploy_ratio")
FLAP_HANDLE_REQUEST_RATIO = DatarefSpec("sim/cockpit2/controls/flap_handle_request_ratio")
GEAR_HANDLE_DOWN = DatarefSpec("sim/cockpit2/controls/gear_handle_down")
Y_AGL_M = DatarefSpec("sim/flightmodel/position/y_agl")
SIM_TIME_S = DatarefSpec("sim/time/total_running_time_sec")
ON_GROUND_0 = DatarefSpec("sim/flightmodel2/gear/on_ground", index=0)

YOKE_PITCH_RATIO = DatarefSpec("sim/joystick/yoke_pitch_ratio")
YOKE_ROLL_RATIO = DatarefSpec("sim/joystick/yoke_roll_ratio")
YOKE_HEADING_RATIO = DatarefSpec("sim/joystick/yoke_heading_ratio")
LEFT_BRAKE_RATIO = DatarefSpec("sim/cockpit2/controls/left_brake_ratio")
RIGHT_BRAKE_RATIO = DatarefSpec("sim/cockpit2/controls/right_brake_ratio")

COM1_FREQUENCY_HZ_833 = DatarefSpec("sim/cockpit2/radios/actuators/com1_frequency_hz_833")
COM2_FREQUENCY_HZ_833 = DatarefSpec("sim/cockpit2/radios/actuators/com2_frequency_hz_833")

STATE_DATAREFS: tuple[DatarefSpec, ...] = (
    LATITUDE_DEG,
    LONGITUDE_DEG,
    ELEVATION_M,
    PITCH_DEG,
    ROLL_DEG,
    HEADING_DEG,
    IAS_KT,
    LOCAL_VX_M_S,
    LOCAL_VZ_M_S,
    VS_FPM,
    P_DEG_S,
    Q_DEG_S,
    R_DEG_S,
    THROTTLE_ALL,
    FLAP_HANDLE_DEPLOY_RATIO,
    GEAR_HANDLE_DOWN,
    Y_AGL_M,
    SIM_TIME_S,
    ON_GROUND_0,
)

COMMAND_DATAREFS: tuple[DatarefSpec, ...] = (
    YOKE_PITCH_RATIO,
    YOKE_ROLL_RATIO,
    YOKE_HEADING_RATIO,
    THROTTLE_ALL,
    GEAR_HANDLE_DOWN,
    FLAP_HANDLE_REQUEST_RATIO,
    LEFT_BRAKE_RATIO,
    RIGHT_BRAKE_RATIO,
)

BOOTSTRAP_DATAREFS: tuple[DatarefSpec, ...] = (
    LATITUDE_DEG,
    LONGITUDE_DEG,
    ELEVATION_M,
    PITCH_DEG,
    ROLL_DEG,
    HEADING_DEG,
    Y_AGL_M,
    ON_GROUND_0,
)
