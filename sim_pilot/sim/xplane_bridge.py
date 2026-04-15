from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
import threading
import time
from typing import Any
from urllib import error, parse, request

from websockets.sync.client import ClientConnection, connect as ws_connect

from sim_pilot.core.types import ActuatorCommands, Vec2, clamp
from sim_pilot.sim.datarefs import (
    BOOTSTRAP_DATAREFS,
    COMMAND_DATAREFS,
    DatarefSpec,
    ELEVATION_M,
    FLAP_HANDLE_DEPLOY_RATIO,
    FLAP_HANDLE_REQUEST_RATIO,
    GEAR_HANDLE_DOWN,
    HEADING_DEG,
    IAS_KT,
    LATITUDE_DEG,
    LEFT_BRAKE_RATIO,
    LOCAL_VX_M_S,
    LOCAL_VZ_M_S,
    LONGITUDE_DEG,
    ON_GROUND_0,
    OVERRIDE_JOYSTICK_HEADING,
    P_DEG_S,
    PITCH_DEG,
    Q_DEG_S,
    R_DEG_S,
    RIGHT_BRAKE_RATIO,
    ROLL_DEG,
    SIM_TIME_S,
    STATE_DATAREFS,
    THROTTLE_ALL,
    VS_FPM,
    Y_AGL_M,
    YOKE_HEADING_RATIO,
    YOKE_PITCH_RATIO,
    YOKE_ROLL_RATIO,
)
from sim_pilot.sim.simple_dynamics import DynamicsState

M_TO_FT = 3.280839895013123
EARTH_RADIUS_FT = 20_925_524.9
DEFAULT_WEB_PORT = 8086


@dataclass(slots=True, frozen=True)
class GeoReference:
    threshold_lat_deg: float
    threshold_lon_deg: float


@dataclass(slots=True, frozen=True)
class PositionSample:
    lat_deg: float
    lon_deg: float
    altitude_msl_m: float
    roll_deg: float
    pitch_deg: float
    heading_deg: float


@dataclass(slots=True, frozen=True)
class BootstrapSample:
    posi: PositionSample
    alt_agl_ft: float
    on_ground: bool


@dataclass
class XPlaneWebBridge:
    georef: GeoReference
    host: str = "127.0.0.1"
    port: int = DEFAULT_WEB_PORT
    connect_timeout_s: float = 5.0
    initial_snapshot_timeout_s: float = 3.0
    _ws: ClientConnection = field(init=False, repr=False)
    _read_ids: dict[str, int] = field(init=False, default_factory=dict, repr=False)
    _command_ids: dict[str, int] = field(init=False, default_factory=dict, repr=False)
    _cached_values: dict[str, float] = field(init=False, default_factory=dict, repr=False)
    _last_flap_ratio: float = field(init=False, default=0.0, repr=False)
    _last_gear_down: bool = field(init=False, default=True, repr=False)
    _req_id: int = field(init=False, default=0, repr=False)
    _ws_send_lock: threading.Lock = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._ws_send_lock = threading.Lock()
        _check_capabilities(self._rest_base(), self.connect_timeout_s)
        specs = tuple({spec.name: spec for spec in STATE_DATAREFS + COMMAND_DATAREFS}.values())
        resolved = _resolve_dataref_ids(
            self._rest_base(),
            specs,
            timeout_s=self.connect_timeout_s,
        )
        for spec in STATE_DATAREFS:
            self._read_ids[spec.name] = resolved[spec.name]
        for spec in COMMAND_DATAREFS:
            self._command_ids[spec.name] = resolved[spec.name]
        self._ws = ws_connect(
            f"ws://{self.host}:{self.port}/api/v3",
            open_timeout=self.connect_timeout_s,
        )
        self._subscribe_state()
        self._wait_for_initial_snapshot()
        # Without this, writes to yoke_heading_ratio are stored but not
        # applied to the rudder / nosewheel. See OVERRIDE_JOYSTICK_HEADING
        # docstring in datarefs.py for the investigation notes.
        self._send_message(
            {
                "req_id": self._next_req_id(),
                "type": "dataref_set_values",
                "params": {
                    "datarefs": [
                        {"id": self._command_ids[OVERRIDE_JOYSTICK_HEADING.name], "value": 1},
                    ]
                },
            }
        )

    def read_state(self) -> DynamicsState:
        self._drain_pending_updates()
        missing = [spec.name for spec in STATE_DATAREFS if spec.name not in self._cached_values]
        if missing:
            raise RuntimeError(f"Missing dataref values after drain: {missing}")
        values = self._cached_values
        lat_deg = values[LATITUDE_DEG.name]
        lon_deg = values[LONGITUDE_DEG.name]
        altitude_m = values[ELEVATION_M.name]
        local_vx_m_s = values[LOCAL_VX_M_S.name]
        local_vz_m_s = values[LOCAL_VZ_M_S.name]
        vh_ind_fpm = values[VS_FPM.name]
        y_agl_m = values[Y_AGL_M.name]
        sim_time_s = values[SIM_TIME_S.name]
        on_ground_ref = values[ON_GROUND_0.name]
        throttle_pos = clamp(values[THROTTLE_ALL.name], 0.0, 1.0)
        flap_ratio = clamp(values[FLAP_HANDLE_DEPLOY_RATIO.name], 0.0, 1.0)
        gear_down = values[GEAR_HANDLE_DOWN.name] >= 0.5
        self._last_flap_ratio = flap_ratio
        self._last_gear_down = gear_down

        position_ft = _geodetic_offset_ft(
            lat_deg=lat_deg,
            lon_deg=lon_deg,
            georef=self.georef,
        )
        ground_velocity_ft_s = Vec2(local_vx_m_s * M_TO_FT, -local_vz_m_s * M_TO_FT)
        return DynamicsState(
            position_ft=position_ft,
            altitude_ft=altitude_m * M_TO_FT,
            heading_deg=values[HEADING_DEG.name],
            roll_deg=values[ROLL_DEG.name],
            pitch_deg=values[PITCH_DEG.name],
            ias_kt=values[IAS_KT.name],
            throttle_pos=throttle_pos,
            on_ground=(on_ground_ref >= 0.5) or (y_agl_m <= 0.5),
            time_s=sim_time_s,
            p_rad_s=math.radians(values[P_DEG_S.name]),
            q_rad_s=math.radians(values[Q_DEG_S.name]),
            r_rad_s=math.radians(values[R_DEG_S.name]),
            ground_velocity_ft_s=ground_velocity_ft_s,
            vertical_speed_ft_s=vh_ind_fpm / 60.0,
            flap_index=_flap_ratio_to_setting(flap_ratio),
            gear_down=gear_down,
        )

    def write_commands(self, commands: ActuatorCommands) -> None:
        flap_ratio = self._last_flap_ratio if commands.flaps is None else _flap_setting_to_ratio(commands.flaps)
        gear_down = self._last_gear_down if commands.gear_down is None else commands.gear_down
        brake_ratio = clamp(commands.brakes, 0.0, 1.0)
        writes: list[dict[str, Any]] = [
            {"id": self._command_ids[YOKE_PITCH_RATIO.name], "value": clamp(commands.elevator, -1.0, 1.0)},
            {"id": self._command_ids[YOKE_ROLL_RATIO.name], "value": clamp(commands.aileron, -1.0, 1.0)},
            {"id": self._command_ids[YOKE_HEADING_RATIO.name], "value": clamp(commands.rudder, -1.0, 1.0)},
            {"id": self._command_ids[THROTTLE_ALL.name], "value": clamp(commands.throttle, 0.0, 1.0)},
            {"id": self._command_ids[GEAR_HANDLE_DOWN.name], "value": 1.0 if gear_down else 0.0},
            {"id": self._command_ids[FLAP_HANDLE_REQUEST_RATIO.name], "value": clamp(flap_ratio, 0.0, 1.0)},
            {"id": self._command_ids[LEFT_BRAKE_RATIO.name], "value": brake_ratio},
            {"id": self._command_ids[RIGHT_BRAKE_RATIO.name], "value": brake_ratio},
        ]
        self._send_message(
            {
                "req_id": self._next_req_id(),
                "type": "dataref_set_values",
                "params": {"datarefs": writes},
            }
        )

    def get_dataref_value(self, name: str) -> float | None:
        """Return the most recently cached value for a subscribed dataref, or None.

        Read-only accessor over the WebSocket-streamed cache. Useful for tools
        that want a one-shot value (lat/lon, COM frequency, etc.) without
        reaching into the private cache directly.
        """
        return self._cached_values.get(name)

    def write_dataref_values(self, updates: dict[str, float | int]) -> None:
        """Write arbitrary datarefs by name. Resolves unknown dataref IDs on demand.

        Used by tools that write cockpit-system datarefs (radios, transponder, lights, etc.)
        without needing to pre-register them at construction time.
        """
        if not updates:
            return
        unresolved = [name for name in updates if name not in self._command_ids and name not in self._read_ids]
        if unresolved:
            specs = tuple(DatarefSpec(name) for name in unresolved)
            resolved = _resolve_dataref_ids(
                self._rest_base(),
                specs,
                timeout_s=self.connect_timeout_s,
            )
            for name, dataref_id in resolved.items():
                self._command_ids[name] = dataref_id
        writes: list[dict[str, Any]] = []
        for name, value in updates.items():
            dataref_id = self._command_ids.get(name) or self._read_ids.get(name)
            if dataref_id is None:
                raise RuntimeError(f"Failed to resolve dataref id for {name!r}")
            writes.append({"id": dataref_id, "value": float(value)})
        self._send_message(
            {
                "req_id": self._next_req_id(),
                "type": "dataref_set_values",
                "params": {"datarefs": writes},
            }
        )

    def close(self) -> None:
        try:
            self._send_message(
                {
                    "req_id": self._next_req_id(),
                    "type": "dataref_set_values",
                    "params": {
                        "datarefs": [
                            {"id": self._command_ids[OVERRIDE_JOYSTICK_HEADING.name], "value": 0},
                        ]
                    },
                }
            )
        except Exception:
            pass
        try:
            self._ws.close()
        except Exception:
            pass

    def _rest_base(self) -> str:
        return f"http://{self.host}:{self.port}/api/v3"

    def _next_req_id(self) -> int:
        self._req_id += 1
        return self._req_id

    def _send_message(self, payload: dict[str, Any]) -> None:
        with self._ws_send_lock:
            self._ws.send(json.dumps(payload))

    def _subscribe_state(self) -> None:
        params_datarefs: list[dict[str, Any]] = []
        for spec in STATE_DATAREFS:
            entry: dict[str, Any] = {"id": self._read_ids[spec.name]}
            if spec.index is not None:
                entry["index"] = spec.index
            params_datarefs.append(entry)
        self._send_message(
            {
                "req_id": self._next_req_id(),
                "type": "dataref_subscribe_values",
                "params": {"datarefs": params_datarefs},
            }
        )

    def _wait_for_initial_snapshot(self) -> None:
        deadline = time.monotonic() + self.initial_snapshot_timeout_s
        needed = {spec.name for spec in STATE_DATAREFS}
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                missing = sorted(needed - self._cached_values.keys())
                raise RuntimeError(f"Timed out waiting for initial dataref snapshot; missing {missing}")
            try:
                raw = self._ws.recv(timeout=remaining)
            except TimeoutError as exc:
                missing = sorted(needed - self._cached_values.keys())
                raise RuntimeError(f"Timed out waiting for initial dataref snapshot; missing {missing}") from exc
            self._handle_message(raw)
            if needed <= self._cached_values.keys():
                return

    def _drain_pending_updates(self) -> None:
        while True:
            try:
                raw = self._ws.recv(timeout=0)
            except TimeoutError:
                return
            self._handle_message(raw)

    def _handle_message(self, raw: str | bytes) -> None:
        text = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else raw
        message = json.loads(text)
        message_type = message.get("type")
        if message_type == "dataref_update_values":
            data = message.get("data", {})
            if not isinstance(data, dict):
                return
            id_to_name = {str(dataref_id): name for name, dataref_id in self._read_ids.items()}
            for raw_id, value in data.items():
                name = id_to_name.get(str(raw_id))
                if name is None:
                    continue
                self._cached_values[name] = _coerce_scalar(value)
            return
        if message_type == "result":
            if not message.get("success", False):
                raise RuntimeError(f"X-Plane web API returned failure: {message!r}")


def probe_bootstrap_sample(
    *,
    host: str = "127.0.0.1",
    port: int = DEFAULT_WEB_PORT,
    timeout_s: float = 5.0,
) -> BootstrapSample:
    base = f"http://{host}:{port}/api/v3"
    _check_capabilities(base, timeout_s)
    resolved = _resolve_dataref_ids(base, BOOTSTRAP_DATAREFS, timeout_s=timeout_s)
    values: dict[str, float] = {}
    for spec in BOOTSTRAP_DATAREFS:
        raw_value = _read_dataref_value(base, resolved[spec.name], timeout_s=timeout_s)
        values[spec.name] = _select_index(raw_value, spec.index)
    return BootstrapSample(
        posi=PositionSample(
            lat_deg=values[LATITUDE_DEG.name],
            lon_deg=values[LONGITUDE_DEG.name],
            altitude_msl_m=values[ELEVATION_M.name],
            roll_deg=values[ROLL_DEG.name],
            pitch_deg=values[PITCH_DEG.name],
            heading_deg=values[HEADING_DEG.name],
        ),
        alt_agl_ft=values[Y_AGL_M.name] * M_TO_FT,
        on_ground=values[ON_GROUND_0.name] >= 0.5,
    )


def _check_capabilities(base_url: str, timeout_s: float) -> None:
    try:
        with request.urlopen(f"{base_url.removesuffix('/api/v3')}/api/capabilities", timeout=timeout_s) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except error.URLError as exc:
        raise RuntimeError(f"Could not reach X-Plane web API at {base_url}: {exc.reason}") from exc
    versions = payload.get("api", {}).get("versions", [])
    if "v3" not in versions:
        raise RuntimeError(f"X-Plane web API does not advertise v3 support: {payload!r}")


def _resolve_dataref_ids(
    base_url: str,
    specs: tuple[DatarefSpec, ...],
    *,
    timeout_s: float,
) -> dict[str, int]:
    query = parse.urlencode([("filter[name]", spec.name) for spec in specs])
    url = f"{base_url}/datarefs?{query}"
    try:
        with request.urlopen(url, timeout=timeout_s) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except error.URLError as exc:
        raise RuntimeError(f"Dataref lookup request failed: {exc.reason}") from exc
    data = payload.get("data", [])
    resolved: dict[str, int] = {}
    for entry in data:
        if isinstance(entry, dict) and isinstance(entry.get("name"), str) and isinstance(entry.get("id"), int):
            resolved[entry["name"]] = int(entry["id"])
    missing = [spec.name for spec in specs if spec.name not in resolved]
    if missing:
        raise RuntimeError(f"X-Plane web API did not return ids for datarefs: {missing}")
    return resolved


def _read_dataref_value(base_url: str, dataref_id: int, *, timeout_s: float) -> Any:
    url = f"{base_url}/datarefs/{dataref_id}/value"
    try:
        with request.urlopen(url, timeout=timeout_s) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except error.URLError as exc:
        raise RuntimeError(f"Dataref value request failed for {dataref_id}: {exc.reason}") from exc
    return payload.get("data")


def _select_index(value: Any, index: int | None) -> float:
    if isinstance(value, list):
        if index is None:
            return float(value[0]) if value else 0.0
        if index < 0 or index >= len(value):
            raise RuntimeError(f"Index {index} out of range for array dataref value of length {len(value)}")
        return float(value[index])
    return float(value)


def _coerce_scalar(value: Any) -> float:
    if isinstance(value, list):
        return float(value[0]) if value else 0.0
    return float(value)


def _geodetic_offset_ft(*, lat_deg: float, lon_deg: float, georef: GeoReference) -> Vec2:
    lat_delta_rad = math.radians(lat_deg - georef.threshold_lat_deg)
    lon_delta_rad = math.radians(lon_deg - georef.threshold_lon_deg)
    mean_lat_rad = math.radians((lat_deg + georef.threshold_lat_deg) * 0.5)
    east_ft = EARTH_RADIUS_FT * lon_delta_rad * math.cos(mean_lat_rad)
    north_ft = EARTH_RADIUS_FT * lat_delta_rad
    return Vec2(east_ft, north_ft)


def _flap_ratio_to_setting(flap_ratio: float) -> int:
    settings = (0, 10, 20, 30)
    target = flap_ratio * 30.0
    return min(settings, key=lambda setting: abs(setting - target))


def _flap_setting_to_ratio(setting: int) -> float:
    return clamp(setting / 30.0, 0.0, 1.0)
