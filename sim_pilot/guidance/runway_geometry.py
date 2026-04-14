from __future__ import annotations

from dataclasses import dataclass

from sim_pilot.core.types import Runway, StraightLeg, Vec2, heading_to_vector


@dataclass(slots=True, frozen=True)
class RunwayFrame:
    runway: Runway

    @property
    def forward(self) -> Vec2:
        return heading_to_vector(self.runway.course_deg)

    @property
    def right(self) -> Vec2:
        return heading_to_vector(self.runway.course_deg + 90.0)

    def to_runway_frame(self, point_ft: Vec2) -> Vec2:
        delta = point_ft - self.runway.threshold_ft
        return Vec2(delta.dot(self.forward), delta.dot(self.right))

    def to_world_frame(self, point_ft: Vec2) -> Vec2:
        return self.runway.threshold_ft + (self.forward * point_ft.x) + (self.right * point_ft.y)

    @property
    def touchdown_runway_x_ft(self) -> float:
        return min(max(self.runway.touchdown_zone_ft * 0.5, 500.0), self.runway.length_ft / 3.0)

    def touchdown_point_ft(self) -> Vec2:
        return self.to_world_frame(Vec2(self.touchdown_runway_x_ft, 0.0))

    def departure_leg(self, length_ft: float = 12000.0) -> StraightLeg:
        return StraightLeg(
            start_ft=self.runway.threshold_ft,
            end_ft=self.to_world_frame(Vec2(length_ft, 0.0)),
        )

    def final_leg(self, start_x_ft: float) -> StraightLeg:
        return StraightLeg(
            start_ft=self.to_world_frame(Vec2(start_x_ft, 0.0)),
            end_ft=self.to_world_frame(Vec2(self.touchdown_runway_x_ft, 0.0)),
        )
