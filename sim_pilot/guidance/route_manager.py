from __future__ import annotations

from dataclasses import dataclass

from sim_pilot.core.types import Vec2, Waypoint


@dataclass(slots=True)
class RouteManager:
    waypoints: list[Waypoint]
    active_index: int = 0
    switch_radius_ft: float = 2500.0

    def active_waypoint(self) -> Waypoint | None:
        if self.active_index >= len(self.waypoints):
            return None
        return self.waypoints[self.active_index]

    def advance_if_needed(self, position_ft: Vec2) -> None:
        waypoint = self.active_waypoint()
        if waypoint is None:
            return
        if position_ft.distance_to(waypoint.position_ft) <= self.switch_radius_ft and self.active_index < (len(self.waypoints) - 1):
            self.active_index += 1

    def is_complete(self) -> bool:
        return self.active_waypoint() is None
