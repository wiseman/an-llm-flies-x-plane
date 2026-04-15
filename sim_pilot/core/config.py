from __future__ import annotations

from dataclasses import dataclass
import importlib.resources
from pathlib import Path
from typing import TypeAlias

from sim_pilot.core.types import Runway, TrafficSide, Vec2

Scalar: TypeAlias = bool | int | float | str
YamlMap: TypeAlias = dict[str, object]


def _parse_scalar(raw_value: str) -> Scalar:
    lowered = raw_value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    try:
        if "." in raw_value:
            return float(raw_value)
        return int(raw_value)
    except ValueError:
        return raw_value


def parse_simple_yaml(text: str) -> YamlMap:
    root: YamlMap = {}
    stack: list[tuple[int, YamlMap]] = [(-1, root)]

    for raw_line in text.splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        if indent % 2 != 0:
            raise ValueError(f"Unsupported indentation in line: {raw_line!r}")
        line = raw_line.strip()
        if ":" not in line:
            raise ValueError(f"Unsupported YAML line: {raw_line!r}")

        while indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]

        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if not value:
            child: YamlMap = {}
            parent[key] = child
            stack.append((indent, child))
            continue
        parent[key] = _parse_scalar(value)

    return root


def _package_path(name: str) -> Path:
    resource = importlib.resources.files("sim_pilot.config").joinpath(name)
    return Path(str(resource))


def _read_yaml(name: str) -> YamlMap:
    return parse_simple_yaml(_package_path(name).read_text(encoding="utf-8"))


@dataclass(slots=True, frozen=True)
class PerformanceConfig:
    aircraft: str
    vr_kt: float
    vx_kt: float
    vy_kt: float
    cruise_altitude_ft: float
    cruise_speed_kt: float
    descent_speed_kt: float
    downwind_speed_kt: float
    base_speed_kt: float
    final_speed_kt: float
    vapp_kt: float
    vref_kt: float
    vso_landing_kt: float


@dataclass(slots=True, frozen=True)
class PatternConfig:
    altitude_agl_ft: float
    downwind_offset_ft: float
    abeam_window_ft: float
    default_extension_ft: float


@dataclass(slots=True, frozen=True)
class FlareConfig:
    roundout_height_ft: float
    flare_start_ft: float
    max_flare_pitch_deg: float


@dataclass(slots=True, frozen=True)
class PIDGains:
    kp: float
    kd: float
    ki: float


@dataclass(slots=True, frozen=True)
class TECSGains:
    kp_total: float
    ki_total: float
    kp_balance: float
    kd_balance: float


@dataclass(slots=True, frozen=True)
class ControllerConfig:
    bank: PIDGains
    pitch: PIDGains
    tecs: TECSGains


@dataclass(slots=True, frozen=True)
class SafetyLimits:
    max_bank_enroute_deg: float
    max_bank_pattern_deg: float
    max_bank_final_deg: float
    max_pitch_up_deg: float
    max_pitch_down_deg: float
    min_stall_margin: float
    unstable_sink_rate_fpm: float
    unstable_centerline_error_ft: float


@dataclass(slots=True, frozen=True)
class MissionConfig:
    entry_start_runway_ft: Vec2


@dataclass(slots=True, frozen=True)
class AirportConfig:
    airport: str | None
    field_elevation_ft: float
    runway: Runway
    mission: MissionConfig


@dataclass(slots=True, frozen=True)
class ConfigBundle:
    performance: PerformanceConfig
    pattern: PatternConfig
    flare: FlareConfig
    controllers: ControllerConfig
    limits: SafetyLimits
    airport: AirportConfig

    @property
    def cruise_altitude_ft(self) -> float:
        return self.performance.cruise_altitude_ft

    @property
    def pattern_altitude_msl_ft(self) -> float:
        return self.airport.field_elevation_ft + self.pattern.altitude_agl_ft


def load_default_config_bundle() -> ConfigBundle:
    aircraft_data = _read_yaml("aircraft_c172.yaml")
    controller_data = _read_yaml("controller_gains.yaml")
    safety_data = _read_yaml("safety_limits.yaml")
    airport_data = _read_yaml("airport_defaults.yaml")

    performance = PerformanceConfig(
        aircraft=str(aircraft_data["aircraft"]),
        vr_kt=float(aircraft_data["vr_kt"]),
        vx_kt=float(aircraft_data["vx_kt"]),
        vy_kt=float(aircraft_data["vy_kt"]),
        cruise_altitude_ft=float(aircraft_data["cruise_altitude_ft"]),
        cruise_speed_kt=float(aircraft_data["cruise_speed_kt"]),
        descent_speed_kt=float(aircraft_data["descent_speed_kt"]),
        downwind_speed_kt=float(aircraft_data["downwind_speed_kt"]),
        base_speed_kt=float(aircraft_data["base_speed_kt"]),
        final_speed_kt=float(aircraft_data["final_speed_kt"]),
        vapp_kt=float(aircraft_data["vapp_kt"]),
        vref_kt=float(aircraft_data["vref_kt"]),
        vso_landing_kt=float(aircraft_data["vso_landing_kt"]),
    )
    pattern_data = aircraft_data["pattern"]
    flare_data = aircraft_data["flare"]
    pattern = PatternConfig(
        altitude_agl_ft=float(pattern_data["altitude_agl_ft"]),
        downwind_offset_ft=float(pattern_data["downwind_offset_ft"]),
        abeam_window_ft=float(pattern_data["abeam_window_ft"]),
        default_extension_ft=float(pattern_data["default_extension_ft"]),
    )
    flare = FlareConfig(
        roundout_height_ft=float(flare_data["roundout_height_ft"]),
        flare_start_ft=float(flare_data["flare_start_ft"]),
        max_flare_pitch_deg=float(flare_data["max_flare_pitch_deg"]),
    )

    bank_data = controller_data["bank"]
    pitch_data = controller_data["pitch"]
    tecs_data = controller_data["tecs"]
    controllers = ControllerConfig(
        bank=PIDGains(
            kp=float(bank_data["kp"]),
            kd=float(bank_data["kd"]),
            ki=float(bank_data["ki"]),
        ),
        pitch=PIDGains(
            kp=float(pitch_data["kp"]),
            kd=float(pitch_data["kd"]),
            ki=float(pitch_data["ki"]),
        ),
        tecs=TECSGains(
            kp_total=float(tecs_data["kp_total"]),
            ki_total=float(tecs_data["ki_total"]),
            kp_balance=float(tecs_data["kp_balance"]),
            kd_balance=float(tecs_data["kd_balance"]),
        ),
    )

    limits = SafetyLimits(
        max_bank_enroute_deg=float(safety_data["max_bank_enroute_deg"]),
        max_bank_pattern_deg=float(safety_data["max_bank_pattern_deg"]),
        max_bank_final_deg=float(safety_data["max_bank_final_deg"]),
        max_pitch_up_deg=float(safety_data["max_pitch_up_deg"]),
        max_pitch_down_deg=float(safety_data["max_pitch_down_deg"]),
        min_stall_margin=float(safety_data["min_stall_margin"]),
        unstable_sink_rate_fpm=float(safety_data["unstable_sink_rate_fpm"]),
        unstable_centerline_error_ft=float(safety_data["unstable_centerline_error_ft"]),
    )

    runway_data = airport_data["runway"]
    mission_data = airport_data["mission"]
    airport = AirportConfig(
        airport=str(airport_data["airport"]),
        field_elevation_ft=float(airport_data["field_elevation_ft"]),
        runway=Runway(
            id=str(runway_data["id"]),
            threshold_ft=Vec2(
                float(runway_data["threshold_x_ft"]),
                float(runway_data["threshold_y_ft"]),
            ),
            course_deg=float(runway_data["course_deg"]),
            length_ft=float(runway_data["length_ft"]),
            touchdown_zone_ft=float(runway_data["touchdown_zone_ft"]),
            traffic_side=TrafficSide(str(runway_data["traffic_side"])),
        ),
        mission=MissionConfig(
            entry_start_runway_ft=Vec2(
                float(mission_data["entry_start_runway_x_ft"]),
                float(mission_data["entry_start_runway_y_ft"]),
            ),
        ),
    )
    return ConfigBundle(
        performance=performance,
        pattern=pattern,
        flare=flare,
        controllers=controllers,
        limits=limits,
        airport=airport,
    )
