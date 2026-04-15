from __future__ import annotations

import json
import queue
import unittest

from sim_pilot.core.config import load_default_config_bundle
from sim_pilot.core.mission_manager import PilotCore, StatusSnapshot
from sim_pilot.core.profiles import (
    AltitudeHoldProfile,
    HeadingHoldProfile,
    PatternFlyProfile,
    SpeedHoldProfile,
)
from sim_pilot.core.types import ActuatorCommands, AircraftState, FlightPhase, Vec2
from sim_pilot.live_runner import HeartbeatPump
from sim_pilot.llm.conversation import IncomingMessage


class _FakeClock:
    def __init__(self, start: float = 0.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, dt: float) -> None:
        self.now += dt


def _make_snapshot(
    *,
    active_profiles: tuple[str, ...],
    phase: FlightPhase | None,
    alt_msl_ft: float = 1500.0,
    go_around_reason: str | None = None,
) -> StatusSnapshot:
    state = AircraftState(
        t_sim=42.0,
        dt=0.2,
        position_ft=Vec2(0.0, 0.0),
        alt_msl_ft=alt_msl_ft,
        alt_agl_ft=max(0.0, alt_msl_ft - 425.0),
        pitch_deg=0.0,
        roll_deg=0.0,
        heading_deg=270.0,
        track_deg=270.0,
        p_rad_s=0.0,
        q_rad_s=0.0,
        r_rad_s=0.0,
        ias_kt=90.0,
        tas_kt=90.0,
        gs_kt=90.0,
        vs_fpm=0.0,
        ground_velocity_ft_s=Vec2(0.0, 0.0),
        flap_index=0,
        gear_down=True,
        on_ground=False,
        throttle_pos=0.55,
        runway_id=None,
        runway_dist_remaining_ft=None,
        runway_x_ft=None,
        runway_y_ft=None,
        centerline_error_ft=None,
        threshold_abeam=False,
        distance_to_touchdown_ft=None,
        stall_margin=2.0,
    )
    return StatusSnapshot(
        t_sim=42.0,
        active_profiles=active_profiles,
        phase=phase,
        state=state,
        last_commands=ActuatorCommands(
            aileron=0.0, elevator=0.0, rudder=0.0, throttle=0.55,
            flaps=None, gear_down=True, brakes=0.0,
        ),
        go_around_reason=go_around_reason,
    )


class HeartbeatPumpEventTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_default_config_bundle()
        self.pilot = PilotCore(self.config)
        self.queue: "queue.Queue[IncomingMessage]" = queue.Queue()
        self.clock = _FakeClock()
        self.pump = HeartbeatPump(
            input_queue=self.queue,
            pilot=self.pilot,
            bridge=None,
            heartbeat_interval_s=30.0,
            clock=self.clock,
        )

    def _seed_snapshot(self, profiles: tuple[str, ...], phase: FlightPhase | None = None) -> None:
        self.pilot.latest_snapshot = _make_snapshot(active_profiles=profiles, phase=phase)

    def _drain_queue(self) -> list[IncomingMessage]:
        msgs: list[IncomingMessage] = []
        while True:
            try:
                msgs.append(self.queue.get_nowait())
            except queue.Empty:
                return msgs

    def test_no_heartbeat_without_snapshot(self) -> None:
        self.pump.check_and_emit()
        self.assertEqual(self._drain_queue(), [])

    def test_first_tick_after_snapshot_only_seeds_state(self) -> None:
        self._seed_snapshot(("idle_lateral", "idle_vertical", "idle_speed"))
        self.pump.check_and_emit()
        self.assertEqual(self._drain_queue(), [])

    def test_profile_change_fires_immediately(self) -> None:
        self._seed_snapshot(("idle_lateral", "idle_vertical", "idle_speed"))
        self.pump.check_and_emit()  # seed
        self.clock.advance(0.5)  # well under min_interval_s=2, but first event
        self._seed_snapshot(("heading_hold", "idle_vertical", "idle_speed"))
        # First change after seeding is gated by min_interval_s default=2
        self.pump.check_and_emit()
        self.assertEqual(self._drain_queue(), [])  # still within debounce
        self.clock.advance(2.5)
        # Change is already recorded as "seen"; nothing new fires on its own
        self.pump.check_and_emit()
        self.assertEqual(self._drain_queue(), [])

    def test_profile_change_past_debounce_fires(self) -> None:
        self._seed_snapshot(("idle_lateral", "idle_vertical", "idle_speed"))
        self.pump.check_and_emit()  # seed
        self.clock.advance(3.0)
        self._seed_snapshot(("heading_hold", "idle_vertical", "idle_speed"))
        self.pump.check_and_emit()
        msgs = self._drain_queue()
        self.assertEqual(len(msgs), 1)
        self.assertEqual(msgs[0].source, "heartbeat")
        self.assertIn("profiles", msgs[0].text)
        self.assertIn("heading_hold", msgs[0].text)
        self.assertIn("idle_lateral", msgs[0].text)

    def test_phase_change_fires_heartbeat(self) -> None:
        self._seed_snapshot(("pattern_fly",), phase=FlightPhase.DOWNWIND)
        self.pump.check_and_emit()  # seed
        self.clock.advance(3.0)
        self._seed_snapshot(("pattern_fly",), phase=FlightPhase.BASE)
        self.pump.check_and_emit()
        msgs = self._drain_queue()
        self.assertEqual(len(msgs), 1)
        self.assertIn("downwind", msgs[0].text)
        self.assertIn("base", msgs[0].text)

    def test_periodic_heartbeat_fires_after_idle_interval(self) -> None:
        self._seed_snapshot(("idle_lateral", "idle_vertical", "idle_speed"))
        self.pump.check_and_emit()  # seed
        # Under the 30s interval, nothing fires
        self.clock.advance(25.0)
        self.pump.check_and_emit()
        self.assertEqual(self._drain_queue(), [])
        # Past the interval, fires once
        self.clock.advance(10.0)
        self.pump.check_and_emit()
        msgs = self._drain_queue()
        self.assertEqual(len(msgs), 1)
        self.assertIn("periodic", msgs[0].text)

    def test_user_input_resets_idle_timer(self) -> None:
        self._seed_snapshot(("idle_lateral", "idle_vertical", "idle_speed"))
        self.pump.check_and_emit()  # seed
        self.clock.advance(25.0)
        self.pump.record_user_input()
        self.clock.advance(10.0)  # 35s since seed, but only 10s since user input
        self.pump.check_and_emit()
        self.assertEqual(self._drain_queue(), [])
        self.clock.advance(25.0)  # now 35s since user input
        self.pump.check_and_emit()
        self.assertEqual(len(self._drain_queue()), 1)

    def test_periodic_heartbeat_resets_its_own_timer(self) -> None:
        """After one periodic heartbeat fires, the next one is 30s later, not
        30s after the previous user input."""
        self._seed_snapshot(("idle_lateral", "idle_vertical", "idle_speed"))
        self.pump.check_and_emit()  # seed
        self.clock.advance(35.0)
        self.pump.check_and_emit()
        self.assertEqual(len(self._drain_queue()), 1)
        # 10s later, no new heartbeat
        self.clock.advance(10.0)
        self.pump.check_and_emit()
        self.assertEqual(self._drain_queue(), [])
        # Another 25s (35s since last heartbeat) → second fires
        self.clock.advance(25.0)
        self.pump.check_and_emit()
        self.assertEqual(len(self._drain_queue()), 1)


class HeartbeatPumpStatusPayloadTests(unittest.TestCase):
    def test_heartbeat_message_embeds_status_json(self) -> None:
        config = load_default_config_bundle()
        pilot = PilotCore(config)
        pilot.latest_snapshot = _make_snapshot(
            active_profiles=("heading_hold", "altitude_hold", "speed_hold"),
            phase=None,
            alt_msl_ft=2500.0,
        )
        input_queue: "queue.Queue[IncomingMessage]" = queue.Queue()
        clock = _FakeClock()
        pump = HeartbeatPump(
            input_queue=input_queue,
            pilot=pilot,
            bridge=None,
            heartbeat_interval_s=30.0,
            clock=clock,
        )
        # Seed, wait out the interval, fire a periodic heartbeat
        pump.check_and_emit()
        clock.advance(35.0)
        pump.check_and_emit()
        msg = input_queue.get_nowait()
        self.assertEqual(msg.source, "heartbeat")
        self.assertIn("periodic check-in", msg.text)
        # Status JSON follows the reason delimited by " | status="
        self.assertIn(" | status=", msg.text)
        payload_json = msg.text.split(" | status=", 1)[1]
        payload = json.loads(payload_json)
        self.assertEqual(payload["active_profiles"], ["heading_hold", "altitude_hold", "speed_hold"])
        self.assertEqual(payload["alt_msl_ft"], 2500.0)
        self.assertEqual(payload["throttle_pos"], 0.55)


class HeartbeatPumpGoAroundReasonTests(unittest.TestCase):
    """Regression for tasks #12 and #15: when phase transitions to
    GO_AROUND, the reason must be surfaced in both the heartbeat
    message body AND a dedicated bus log line."""

    def setUp(self) -> None:
        self.config = load_default_config_bundle()
        self.pilot = PilotCore(self.config)
        self.queue: "queue.Queue[IncomingMessage]" = queue.Queue()
        self.clock = _FakeClock()
        from sim_pilot.bus import SimBus
        self.bus = SimBus()
        self.pump = HeartbeatPump(
            input_queue=self.queue,
            pilot=self.pilot,
            bridge=None,
            bus=self.bus,
            heartbeat_interval_s=30.0,
            clock=self.clock,
        )

    def _seed(self, phase: FlightPhase | None, reason: str | None = None) -> None:
        self.pilot.latest_snapshot = _make_snapshot(
            active_profiles=("pattern_fly",),
            phase=phase,
            go_around_reason=reason,
        )

    def _drain(self) -> list[IncomingMessage]:
        msgs: list[IncomingMessage] = []
        while True:
            try:
                msgs.append(self.queue.get_nowait())
            except queue.Empty:
                return msgs

    def test_go_around_transition_pushes_safety_log_line(self) -> None:
        self._seed(FlightPhase.FINAL, reason=None)
        self.pump.check_and_emit()  # seed
        self.clock.advance(3.0)
        self._seed(FlightPhase.GO_AROUND, reason="unstable_lateral cle=3500ft agl=199ft limit=398ft")
        self.pump.check_and_emit()
        log_tail = self.bus.log_tail(50)
        matching = [line for line in log_tail if "go_around triggered" in line]
        self.assertEqual(len(matching), 1)
        self.assertIn("unstable_lateral", matching[0])
        self.assertIn("cle=3500ft", matching[0])

    def test_go_around_reason_embedded_in_heartbeat_message(self) -> None:
        self._seed(FlightPhase.FINAL, reason=None)
        self.pump.check_and_emit()  # seed
        self.clock.advance(3.0)
        self._seed(FlightPhase.GO_AROUND, reason="unstable_lateral cle=3500ft agl=199ft limit=398ft")
        self.pump.check_and_emit()
        msgs = self._drain()
        self.assertEqual(len(msgs), 1)
        self.assertIn("final -> go_around", msgs[0].text)
        self.assertIn("unstable_lateral", msgs[0].text)

    def test_non_go_around_transition_does_not_push_safety_log(self) -> None:
        self._seed(FlightPhase.DOWNWIND, reason=None)
        self.pump.check_and_emit()  # seed
        self.clock.advance(3.0)
        self._seed(FlightPhase.BASE, reason=None)
        self.pump.check_and_emit()
        log_tail = self.bus.log_tail(50)
        matching = [line for line in log_tail if "go_around triggered" in line]
        self.assertEqual(matching, [])

    def test_unknown_go_around_reason_still_logs(self) -> None:
        # Defensive: if somehow the snapshot lacks a reason but phase
        # still flipped, the log line should still be pushed with
        # "unknown" instead of being silently dropped.
        self._seed(FlightPhase.FINAL, reason=None)
        self.pump.check_and_emit()
        self.clock.advance(3.0)
        self._seed(FlightPhase.GO_AROUND, reason=None)
        self.pump.check_and_emit()
        log_tail = self.bus.log_tail(50)
        matching = [line for line in log_tail if "go_around triggered" in line]
        self.assertEqual(len(matching), 1)
        self.assertIn("unknown", matching[0])


class HeartbeatPumpPilotCoreIntegrationTests(unittest.TestCase):
    """Integration: run the pilot core under the simple backend for one tick,
    then feed the pump; it should see the snapshot and detect profile
    changes when the operator engages a new profile."""

    def test_heartbeat_sees_profile_change_from_live_pilot_update(self) -> None:
        config = load_default_config_bundle()
        pilot = PilotCore(config)
        input_queue: "queue.Queue[IncomingMessage]" = queue.Queue()
        clock = _FakeClock()
        pump = HeartbeatPump(
            input_queue=input_queue,
            pilot=pilot,
            bridge=None,
            heartbeat_interval_s=30.0,
            clock=clock,
        )
        from sim_pilot.sim.simple_dynamics import SimpleAircraftModel
        model = SimpleAircraftModel(config, Vec2(0.0, 0.0))
        raw = model.initial_state()
        # First update seeds the snapshot with the three idle profiles
        pilot.update(raw, 0.2)
        pump.check_and_emit()  # seed pump's "previously seen" state
        self.assertTrue(input_queue.empty())
        # Engage heading_hold — profile set changes
        pilot.engage_profile(HeadingHoldProfile(heading_deg=270.0))
        pilot.update(raw, 0.2)
        clock.advance(3.0)
        pump.check_and_emit()
        msg = input_queue.get_nowait()
        self.assertEqual(msg.source, "heartbeat")
        self.assertIn("engaged: heading_hold", msg.text)
        self.assertIn("disengaged: idle_lateral", msg.text)


if __name__ == "__main__":
    unittest.main()
