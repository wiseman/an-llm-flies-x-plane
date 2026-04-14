from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import math

NM_TO_FT = 6076.12
KT_TO_FPS = 1.6878098571011957
G_FT_S2 = 32.174


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def wrap_degrees_360(angle_deg: float) -> float:
    return angle_deg % 360.0


def wrap_degrees_180(angle_deg: float) -> float:
    wrapped = wrap_degrees_360(angle_deg + 180.0) - 180.0
    if wrapped == -180.0:
        return 180.0
    return wrapped


def heading_to_vector(heading_deg: float, magnitude: float = 1.0) -> "Vec2":
    radians = math.radians(heading_deg)
    return Vec2(math.sin(radians) * magnitude, math.cos(radians) * magnitude)


def vector_to_heading(vector: "Vec2") -> float:
    if vector.length() <= 1e-9:
        return 0.0
    return wrap_degrees_360(math.degrees(math.atan2(vector.x, vector.y)))


def course_between(start_ft: "Vec2", end_ft: "Vec2") -> float:
    return vector_to_heading(end_ft - start_ft)


class FlightPhase(StrEnum):
    PREFLIGHT = "preflight"
    TAKEOFF_ROLL = "takeoff_roll"
    ROTATE = "rotate"
    INITIAL_CLIMB = "initial_climb"
    ENROUTE_CLIMB = "enroute_climb"
    CRUISE = "cruise"
    DESCENT = "descent"
    PATTERN_ENTRY = "pattern_entry"
    DOWNWIND = "downwind"
    BASE = "base"
    FINAL = "final"
    ROUNDOUT = "roundout"
    FLARE = "flare"
    ROLLOUT = "rollout"
    TAXI_CLEAR = "taxi_clear"
    GO_AROUND = "go_around"


class TrafficSide(StrEnum):
    LEFT = "left"
    RIGHT = "right"


class LateralMode(StrEnum):
    BANK_HOLD = "bank_hold"
    TRACK_HOLD = "track_hold"
    PATH_FOLLOW = "path_follow"
    CENTERLINE_INTERCEPT = "centerline_intercept"
    ROLLOUT_CENTERLINE = "rollout_centerline"


class VerticalMode(StrEnum):
    PITCH_HOLD = "pitch_hold"
    ALTITUDE_HOLD = "altitude_hold"
    TECS = "tecs"
    GLIDEPATH_TRACK = "glidepath_track"
    FLARE_TRACK = "flare_track"


@dataclass(slots=True, frozen=True)
class Vec2:
    x: float
    y: float

    def __add__(self, other: "Vec2") -> "Vec2":
        return Vec2(self.x + other.x, self.y + other.y)

    def __sub__(self, other: "Vec2") -> "Vec2":
        return Vec2(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar: float) -> "Vec2":
        return Vec2(self.x * scalar, self.y * scalar)

    __rmul__ = __mul__

    def dot(self, other: "Vec2") -> float:
        return self.x * other.x + self.y * other.y

    def cross(self, other: "Vec2") -> float:
        return self.x * other.y - self.y * other.x

    def length(self) -> float:
        return math.hypot(self.x, self.y)

    def normalized(self) -> "Vec2":
        magnitude = self.length()
        if magnitude <= 1e-9:
            return Vec2(0.0, 0.0)
        return Vec2(self.x / magnitude, self.y / magnitude)

    def distance_to(self, other: "Vec2") -> float:
        return (self - other).length()


@dataclass(slots=True, frozen=True)
class Waypoint:
    name: str
    position_ft: Vec2
    altitude_ft: float | None = None


@dataclass(slots=True, frozen=True)
class Runway:
    id: str
    threshold_ft: Vec2
    course_deg: float
    length_ft: float
    touchdown_zone_ft: float
    traffic_side: TrafficSide


@dataclass(slots=True, frozen=True)
class StraightLeg:
    start_ft: Vec2
    end_ft: Vec2

    @property
    def course_deg(self) -> float:
        return course_between(self.start_ft, self.end_ft)


@dataclass(slots=True, frozen=True)
class Glidepath:
    slope_deg: float
    threshold_crossing_height_ft: float
    aimpoint_ft_from_threshold: float


@dataclass(slots=True)
class GuidanceTargets:
    lateral_mode: LateralMode
    vertical_mode: VerticalMode
    target_bank_deg: float | None = None
    target_heading_deg: float | None = None
    target_track_deg: float | None = None
    target_path: StraightLeg | None = None
    target_waypoint: Waypoint | None = None
    target_altitude_ft: float | None = None
    target_speed_kt: float | None = None
    target_pitch_deg: float | None = None
    glidepath: Glidepath | None = None
    throttle_limit: tuple[float, float] | None = None
    flaps_cmd: int | None = None
    gear_down: bool | None = None
    brakes: float = 0.0


@dataclass(slots=True, frozen=True)
class ActuatorCommands:
    aileron: float
    elevator: float
    rudder: float
    throttle: float
    flaps: int | None
    gear_down: bool | None
    brakes: float


@dataclass(slots=True, frozen=True)
class AircraftState:
    t_sim: float
    dt: float
    position_ft: Vec2
    alt_msl_ft: float
    alt_agl_ft: float
    pitch_deg: float
    roll_deg: float
    heading_deg: float
    track_deg: float
    p_rad_s: float
    q_rad_s: float
    r_rad_s: float
    ias_kt: float
    tas_kt: float
    gs_kt: float
    vs_fpm: float
    ground_velocity_ft_s: Vec2
    flap_index: int
    gear_down: bool
    on_ground: bool
    throttle_pos: float
    runway_id: str | None
    runway_dist_remaining_ft: float | None
    runway_x_ft: float | None
    runway_y_ft: float | None
    centerline_error_ft: float | None
    threshold_abeam: bool
    distance_to_touchdown_ft: float | None
    stall_margin: float
