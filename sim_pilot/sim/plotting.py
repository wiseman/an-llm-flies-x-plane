from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from sim_pilot.core.config import ConfigBundle
from sim_pilot.core.types import FlightPhase, Vec2
from sim_pilot.guidance.runway_geometry import RunwayFrame
from sim_pilot.sim.scenario import ScenarioLogRow, ScenarioResult


@dataclass(slots=True, frozen=True)
class PlotArtifacts:
    overview_svg: Path
    ground_path_svg: Path


PHASE_COLORS: dict[FlightPhase, str] = {
    FlightPhase.TAKEOFF_ROLL: "#475569",
    FlightPhase.ROTATE: "#7c3aed",
    FlightPhase.INITIAL_CLIMB: "#2563eb",
    FlightPhase.ENROUTE_CLIMB: "#0891b2",
    FlightPhase.CRUISE: "#0f766e",
    FlightPhase.DESCENT: "#65a30d",
    FlightPhase.PATTERN_ENTRY: "#f59e0b",
    FlightPhase.DOWNWIND: "#ea580c",
    FlightPhase.BASE: "#dc2626",
    FlightPhase.FINAL: "#be123c",
    FlightPhase.ROUNDOUT: "#db2777",
    FlightPhase.FLARE: "#c026d3",
    FlightPhase.ROLLOUT: "#4338ca",
    FlightPhase.TAXI_CLEAR: "#334155",
    FlightPhase.GO_AROUND: "#b91c1c",
    FlightPhase.PREFLIGHT: "#94a3b8",
}


def write_scenario_plots(
    result: ScenarioResult,
    config: ConfigBundle,
    scenario_name: str,
    output_dir: str | Path,
) -> PlotArtifacts:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)

    overview_svg = directory / "flight_overview.svg"
    ground_path_svg = directory / "ground_path.svg"

    overview_svg.write_text(_build_overview_svg(result, scenario_name), encoding="utf-8")
    ground_path_svg.write_text(_build_ground_path_svg(result, config, scenario_name), encoding="utf-8")
    return PlotArtifacts(overview_svg=overview_svg, ground_path_svg=ground_path_svg)


def _build_overview_svg(result: ScenarioResult, scenario_name: str) -> str:
    rows = result.history
    width = 1280
    height = 1180
    margin_left = 96.0
    margin_right = 28.0
    margin_top = 52.0
    margin_bottom = 84.0
    panel_gap = 22.0
    panel_count = 5
    plot_width = width - margin_left - margin_right
    panel_height = (height - margin_top - margin_bottom - (panel_gap * (panel_count - 1))) / panel_count

    panels: list[tuple[str, list[tuple[str, str, list[float]]], tuple[float, float] | None]] = [
        (
            "Altitude AGL (ft)",
            [("altitude_agl_ft", "#0f766e", [row.altitude_agl_ft for row in rows])],
            None,
        ),
        (
            "Speed (kt)",
            [
                ("ground_speed", "#2563eb", [row.gs_kt for row in rows]),
                ("ias", "#16a34a", [row.ias_kt for row in rows]),
            ],
            None,
        ),
        (
            "Attitude (deg)",
            [
                ("bank", "#dc2626", [row.bank_deg for row in rows]),
                ("pitch", "#d97706", [row.pitch_deg for row in rows]),
            ],
            None,
        ),
        (
            "Throttle (0..1)",
            [
                ("throttle_pos", "#7c3aed", [row.throttle_pos for row in rows]),
                ("throttle_cmd", "#6b7280", [row.throttle_cmd for row in rows]),
            ],
            (0.0, 1.0),
        ),
        (
            "Heading (deg)",
            [("heading", "#0891b2", [row.heading_deg for row in rows])],
            (0.0, 360.0),
        ),
    ]

    x_values = [row.time_s for row in rows]
    x_min = x_values[0] if x_values else 0.0
    x_max = x_values[-1] if x_values else 1.0
    phase_segments = _phase_segments(rows)

    elements: list[str] = [
        _svg_header(width, height),
        '<rect width="100%" height="100%" fill="#f8fafc" />',
        f'<text x="{margin_left:.1f}" y="28" font-family="monospace" font-size="20" fill="#0f172a">Scenario: {scenario_name}</text>',
        f'<text x="{margin_left:.1f}" y="46" font-family="monospace" font-size="12" fill="#334155">duration={result.duration_s:.1f}s final_phase={result.final_phase.value} success={str(result.success).lower()}</text>',
    ]

    for index, (title, series_definitions, fixed_domain) in enumerate(panels):
        panel_top = margin_top + index * (panel_height + panel_gap)
        panel_bottom = panel_top + panel_height
        series_values = [value for _, _, values in series_definitions for value in values]
        y_min, y_max = fixed_domain if fixed_domain is not None else _expanded_domain(series_values)
        elements.append(
            f'<rect x="{margin_left:.1f}" y="{panel_top:.1f}" width="{plot_width:.1f}" height="{panel_height:.1f}" fill="white" stroke="#cbd5e1" stroke-width="1" />'
        )
        elements.extend(_panel_grid(panel_top, panel_height, margin_left, plot_width, y_min, y_max))
        for phase_segment in phase_segments[1:]:
            x = _scale(phase_segment.start_time_s, x_min, x_max, margin_left, margin_left + plot_width)
            elements.append(f'<line x1="{x:.1f}" y1="{panel_top:.1f}" x2="{x:.1f}" y2="{panel_bottom:.1f}" stroke="#e2e8f0" stroke-width="1" />')

        legend_x = margin_left + 8.0
        legend_y = panel_top + 16.0
        elements.append(
            f'<text x="{legend_x:.1f}" y="{legend_y:.1f}" font-family="monospace" font-size="13" fill="#0f172a">{title}</text>'
        )
        for series_index, (series_label, color, values) in enumerate(series_definitions):
            polyline = _build_polyline(
                x_values=x_values,
                y_values=values,
                x_domain=(x_min, x_max),
                y_domain=(y_min, y_max),
                bounds=(margin_left, panel_top, plot_width, panel_height),
            )
            elements.append(f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{polyline}" />')
            elements.append(
                f'<text x="{legend_x + 180 + series_index * 150:.1f}" y="{legend_y:.1f}" font-family="monospace" font-size="12" fill="{color}">{series_label}</text>'
            )

        elements.append(
            f'<text x="{margin_left - 36:.1f}" y="{panel_top + panel_height / 2:.1f}" font-family="monospace" font-size="11" fill="#475569">{y_max:.1f}</text>'
        )
        elements.append(
            f'<text x="{margin_left - 36:.1f}" y="{panel_bottom - 4:.1f}" font-family="monospace" font-size="11" fill="#475569">{y_min:.1f}</text>'
        )

    phase_axis_top = height - margin_bottom + 18.0
    phase_axis_height = 26.0
    elements.append(
        f'<text x="{margin_left:.1f}" y="{phase_axis_top - 8.0:.1f}" font-family="monospace" font-size="12" fill="#475569">phase by time</text>'
    )
    elements.append(
        f'<rect x="{margin_left:.1f}" y="{phase_axis_top:.1f}" width="{plot_width:.1f}" height="{phase_axis_height:.1f}" fill="white" stroke="#cbd5e1" stroke-width="1" />'
    )
    for phase_segment in phase_segments:
        start_x = _scale(phase_segment.start_time_s, x_min, x_max, margin_left, margin_left + plot_width)
        end_x = _scale(phase_segment.end_time_s, x_min, x_max, margin_left, margin_left + plot_width)
        segment_width = max(1.5, end_x - start_x)
        phase_color = _phase_color(phase_segment.phase)
        elements.append(
            f'<rect x="{start_x:.1f}" y="{phase_axis_top:.1f}" width="{segment_width:.1f}" height="{phase_axis_height:.1f}" fill="{phase_color}" fill-opacity="0.22" stroke="{phase_color}" stroke-width="0.8" />'
        )
        label = _phase_label(phase_segment.phase)
        font_size = 9 if segment_width < 70.0 else 10
        if segment_width >= 24.0:
            elements.append(
                f'<text x="{(start_x + end_x) / 2.0:.1f}" y="{phase_axis_top + 16.5:.1f}" text-anchor="middle" font-family="monospace" font-size="{font_size}" fill="#0f172a">{label}</text>'
            )

    elements.append(
        f'<text x="{margin_left:.1f}" y="{height - 18:.1f}" font-family="monospace" font-size="12" fill="#475569">time_s: {x_min:.1f}..{x_max:.1f}</text>'
    )
    elements.append("</svg>")
    return "\n".join(elements)


def _build_ground_path_svg(result: ScenarioResult, config: ConfigBundle, scenario_name: str) -> str:
    rows = result.history
    width = 1280
    height = 900
    margin_left = 90.0
    margin_right = 40.0
    margin_top = 55.0
    margin_bottom = 70.0
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    runway_length_ft = config.airport.runway.length_ft
    runway_half_width_ft = 75.0
    touchdown_zone_end_ft = min(config.airport.runway.touchdown_zone_ft, runway_length_ft / 3.0)
    runway_frame = RunwayFrame(config.airport.runway)

    runway_polygon_world = [
        runway_frame.to_world_frame(point_ft)
        for point_ft in (
            Vec2(0.0, -runway_half_width_ft),
            Vec2(0.0, runway_half_width_ft),
            Vec2(runway_length_ft, runway_half_width_ft),
            Vec2(runway_length_ft, -runway_half_width_ft),
        )
    ]
    touchdown_polygon_world = [
        runway_frame.to_world_frame(point_ft)
        for point_ft in (
            Vec2(0.0, -runway_half_width_ft),
            Vec2(0.0, runway_half_width_ft),
            Vec2(touchdown_zone_end_ft, runway_half_width_ft),
            Vec2(touchdown_zone_end_ft, -runway_half_width_ft),
        )
    ]
    runway_centerline_world = (
        runway_frame.to_world_frame(Vec2(0.0, 0.0)),
        runway_frame.to_world_frame(Vec2(runway_length_ft, 0.0)),
    )
    runway_threshold_world = runway_frame.to_world_frame(Vec2(0.0, 0.0))
    runway_departure_end_world = runway_frame.to_world_frame(Vec2(runway_length_ft, 0.0))

    xs = [row.position_x_ft for row in rows] + [point.x for point in runway_polygon_world]
    ys = [row.position_y_ft for row in rows] + [point.y for point in runway_polygon_world]
    x_min, x_max = _expanded_domain(xs, extra_padding=0.08)
    y_min, y_max = _expanded_domain(ys, extra_padding=0.08)
    x_min, x_max, y_min, y_max = _expand_to_equal_aspect(
        x_domain=(x_min, x_max),
        y_domain=(y_min, y_max),
        plot_width=plot_width,
        plot_height=plot_height,
    )
    phase_segments = _phase_segments(rows)

    def scale_x(value: float) -> float:
        return _scale(value, x_min, x_max, margin_left, margin_left + plot_width)

    def scale_y(value: float) -> float:
        return _scale(value, y_min, y_max, margin_top + plot_height, margin_top)

    elements: list[str] = [
        _svg_header(width, height),
        '<rect width="100%" height="100%" fill="#f8fafc" />',
        f'<text x="{margin_left:.1f}" y="28" font-family="monospace" font-size="20" fill="#0f172a">Ground Path: {scenario_name}</text>',
        f'<text x="{margin_left:.1f}" y="46" font-family="monospace" font-size="12" fill="#334155">north-up world frame: x=east_ft y=north_ft</text>',
        f'<rect x="{margin_left:.1f}" y="{margin_top:.1f}" width="{plot_width:.1f}" height="{plot_height:.1f}" fill="white" stroke="#cbd5e1" stroke-width="1" />',
    ]

    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
        x = margin_left + fraction * plot_width
        y = margin_top + fraction * plot_height
        elements.append(f'<line x1="{x:.1f}" y1="{margin_top:.1f}" x2="{x:.1f}" y2="{margin_top + plot_height:.1f}" stroke="#e2e8f0" stroke-width="1" />')
        elements.append(f'<line x1="{margin_left:.1f}" y1="{y:.1f}" x2="{margin_left + plot_width:.1f}" y2="{y:.1f}" stroke="#e2e8f0" stroke-width="1" />')

    runway_polygon = " ".join(f"{scale_x(point.x):.2f},{scale_y(point.y):.2f}" for point in runway_polygon_world)
    touchdown_polygon = " ".join(f"{scale_x(point.x):.2f},{scale_y(point.y):.2f}" for point in touchdown_polygon_world)
    centerline = " ".join(
        f"{scale_x(point.x):.2f},{scale_y(point.y):.2f}" for point in runway_centerline_world
    )

    elements.append(
        f'<polygon points="{runway_polygon}" fill="#cbd5e1" stroke="#64748b" stroke-width="1.5" />'
    )
    elements.append(
        f'<polygon points="{touchdown_polygon}" fill="#94a3b8" opacity="0.45" />'
    )
    elements.append(
        f'<polyline points="{centerline}" fill="none" stroke="white" stroke-width="2" stroke-dasharray="16 10" />'
    )
    elements.append(
        f'<text x="{scale_x(runway_threshold_world.x) + 8:.1f}" y="{scale_y(runway_threshold_world.y) - 8:.1f}" font-family="monospace" font-size="12" fill="#334155">Runway {config.airport.runway.id} threshold</text>'
    )
    elements.append(
        f'<text x="{scale_x(runway_departure_end_world.x) + 8:.1f}" y="{scale_y(runway_departure_end_world.y) - 8:.1f}" font-family="monospace" font-size="12" fill="#334155">departure end</text>'
    )
    elements.append(
        f'<text x="{margin_left + plot_width - 72:.1f}" y="{margin_top + 18:.1f}" font-family="monospace" font-size="12" fill="#334155">N</text>'
    )
    elements.append(
        f'<line x1="{margin_left + plot_width - 66:.1f}" y1="{margin_top + 24:.1f}" x2="{margin_left + plot_width - 66:.1f}" y2="{margin_top + 52:.1f}" stroke="#334155" stroke-width="1.5" />'
    )
    elements.append(
        f'<polygon points="{margin_left + plot_width - 66:.1f},{margin_top + 14:.1f} {margin_left + plot_width - 72:.1f},{margin_top + 26:.1f} {margin_left + plot_width - 60:.1f},{margin_top + 26:.1f}" fill="#334155" />'
    )

    for segment in phase_segments:
        segment_rows = rows[segment.start_index : segment.end_index + 1]
        if len(segment_rows) < 2:
            continue
        polyline = " ".join(f"{scale_x(row.position_x_ft):.2f},{scale_y(row.position_y_ft):.2f}" for row in segment_rows)
        phase_color = _phase_color(segment.phase)
        elements.append(f'<polyline fill="none" stroke="{phase_color}" stroke-width="2.5" points="{polyline}" />')

    legend_x = margin_left + 12.0
    legend_y = margin_top + 22.0
    seen_phases: list[FlightPhase] = []
    for segment in phase_segments:
        if segment.phase not in seen_phases:
            seen_phases.append(segment.phase)
    for index, phase in enumerate(seen_phases):
        row_idx = index % 5
        col_idx = index // 5
        x = legend_x + col_idx * 180.0
        y = legend_y + row_idx * 18.0
        phase_color = _phase_color(phase)
        elements.append(f'<rect x="{x:.1f}" y="{y - 9.0:.1f}" width="12" height="12" fill="{phase_color}" />')
        elements.append(
            f'<text x="{x + 18.0:.1f}" y="{y + 1.0:.1f}" font-family="monospace" font-size="11" fill="#334155">{_phase_label(phase)}</text>'
        )

    if rows:
        start = rows[0]
        end = rows[-1]
        elements.append(f'<circle cx="{scale_x(start.position_x_ft):.1f}" cy="{scale_y(start.position_y_ft):.1f}" r="5" fill="#16a34a" />')
        elements.append(f'<circle cx="{scale_x(end.position_x_ft):.1f}" cy="{scale_y(end.position_y_ft):.1f}" r="5" fill="#dc2626" />')
        elements.append(
            f'<text x="{scale_x(start.position_x_ft) + 8:.1f}" y="{scale_y(start.position_y_ft) - 8:.1f}" font-family="monospace" font-size="11" fill="#166534">start</text>'
        )
        elements.append(
            f'<text x="{scale_x(end.position_x_ft) + 8:.1f}" y="{scale_y(end.position_y_ft) - 8:.1f}" font-family="monospace" font-size="11" fill="#991b1b">end</text>'
        )

    elements.append(
        f'<text x="{margin_left:.1f}" y="{height - 18:.1f}" font-family="monospace" font-size="12" fill="#475569">x: east-west world feet, y: north-south world feet, north up</text>'
    )
    elements.append("</svg>")
    return "\n".join(elements)


def _svg_header(width: int, height: int) -> str:
    return f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'


def _expanded_domain(values: list[float], extra_padding: float = 0.06) -> tuple[float, float]:
    lower = min(values)
    upper = max(values)
    if abs(upper - lower) <= 1e-9:
        return lower - 1.0, upper + 1.0
    padding = (upper - lower) * extra_padding
    return lower - padding, upper + padding


def _expand_to_equal_aspect(
    *,
    x_domain: tuple[float, float],
    y_domain: tuple[float, float],
    plot_width: float,
    plot_height: float,
) -> tuple[float, float, float, float]:
    x_min, x_max = x_domain
    y_min, y_max = y_domain
    x_span = max(1.0, x_max - x_min)
    y_span = max(1.0, y_max - y_min)
    data_aspect = x_span / y_span
    plot_aspect = plot_width / plot_height

    if data_aspect > plot_aspect:
        desired_y_span = x_span / plot_aspect
        extra_y = (desired_y_span - y_span) / 2.0
        return x_min, x_max, y_min - extra_y, y_max + extra_y

    desired_x_span = y_span * plot_aspect
    extra_x = (desired_x_span - x_span) / 2.0
    return x_min - extra_x, x_max + extra_x, y_min, y_max


def _scale(value: float, domain_min: float, domain_max: float, screen_min: float, screen_max: float) -> float:
    if abs(domain_max - domain_min) <= 1e-9:
        return (screen_min + screen_max) / 2.0
    ratio = (value - domain_min) / (domain_max - domain_min)
    return screen_min + ratio * (screen_max - screen_min)


def _build_polyline(
    *,
    x_values: list[float],
    y_values: list[float],
    x_domain: tuple[float, float],
    y_domain: tuple[float, float],
    bounds: tuple[float, float, float, float],
) -> str:
    left, top, width, height = bounds
    x_min, x_max = x_domain
    y_min, y_max = y_domain
    return " ".join(
        f"{_scale(x, x_min, x_max, left, left + width):.2f},{_scale(y, y_min, y_max, top + height, top):.2f}"
        for x, y in zip(x_values, y_values, strict=True)
    )


@dataclass(slots=True, frozen=True)
class PhaseSegment:
    phase: FlightPhase
    start_time_s: float
    end_time_s: float
    start_index: int
    end_index: int


def _phase_segments(rows: tuple[ScenarioLogRow, ...]) -> list[PhaseSegment]:
    if not rows:
        return []
    segments: list[PhaseSegment] = []
    current_phase = rows[0].phase
    start_index = 0
    start_time_s = rows[0].time_s
    for index, row in enumerate(rows[1:], start=1):
        if row.phase is current_phase:
            continue
        segments.append(
            PhaseSegment(
                phase=current_phase,
                start_time_s=start_time_s,
                end_time_s=rows[index - 1].time_s,
                start_index=start_index,
                end_index=index - 1,
            )
        )
        current_phase = row.phase
        start_index = index - 1
        start_time_s = rows[index - 1].time_s
    segments.append(
        PhaseSegment(
            phase=current_phase,
            start_time_s=start_time_s,
            end_time_s=rows[-1].time_s,
            start_index=start_index,
            end_index=len(rows) - 1,
        )
    )
    return segments


def _phase_color(phase: FlightPhase) -> str:
    return PHASE_COLORS.get(phase, "#334155")


def _phase_label(phase: FlightPhase) -> str:
    return phase.value.replace("_", " ")


def _panel_grid(panel_top: float, panel_height: float, left: float, width: float, y_min: float, y_max: float) -> list[str]:
    lines: list[str] = []
    for fraction in (0.25, 0.5, 0.75):
        y = panel_top + panel_height * fraction
        lines.append(f'<line x1="{left:.1f}" y1="{y:.1f}" x2="{left + width:.1f}" y2="{y:.1f}" stroke="#e2e8f0" stroke-width="1" />')
        label_value = y_max - ((y_max - y_min) * fraction)
        lines.append(
            f'<text x="{left + width - 48:.1f}" y="{y - 4:.1f}" font-family="monospace" font-size="10" fill="#64748b">{label_value:.1f}</text>'
        )
    return lines
