from __future__ import annotations

import io
import threading
import unittest
from unittest.mock import patch

from sim_pilot.bus import SimBus


class SimBusBasicsTests(unittest.TestCase):
    def test_push_status_updates_latest_status_line(self) -> None:
        bus = SimBus(echo=False)
        bus.push_status("first")
        bus.push_status("second")
        status, _, _ = bus.snapshot()
        self.assertEqual(status, "second")

    def test_push_log_appends_to_log_buffer(self) -> None:
        bus = SimBus(echo=False)
        bus.push_log("a")
        bus.push_log("b")
        _, logs, _ = bus.snapshot()
        self.assertEqual(logs, ["a", "b"])

    def test_push_radio_appends_to_radio_buffer(self) -> None:
        bus = SimBus(echo=False)
        bus.push_radio("com1 hello")
        _, _, radio = bus.snapshot()
        self.assertEqual(radio, ["com1 hello"])

    def test_log_tail_returns_last_n_items(self) -> None:
        bus = SimBus(echo=False)
        for i in range(10):
            bus.push_log(f"line{i}")
        tail = bus.log_tail(3)
        self.assertEqual(tail, ["line7", "line8", "line9"])

    def test_log_buffer_has_upper_bound(self) -> None:
        bus = SimBus(echo=False)
        # Push more than the default cap (1000)
        for i in range(1200):
            bus.push_log(f"line{i}")
        _, logs, _ = bus.snapshot()
        self.assertEqual(len(logs), 1000)
        self.assertEqual(logs[0], "line200")
        self.assertEqual(logs[-1], "line1199")


class SimBusEchoTests(unittest.TestCase):
    def test_echo_true_prints_each_push_to_stdout(self) -> None:
        buf = io.StringIO()
        bus = SimBus(echo=True)
        with patch("sys.stdout", new=buf):
            bus.push_status("s")
            bus.push_log("l")
            bus.push_radio("r")
        output = buf.getvalue()
        self.assertIn("s", output)
        self.assertIn("l", output)
        self.assertIn("r", output)

    def test_echo_false_does_not_print(self) -> None:
        buf = io.StringIO()
        bus = SimBus(echo=False)
        with patch("sys.stdout", new=buf):
            bus.push_status("s")
            bus.push_log("l")
            bus.push_radio("r")
        self.assertEqual(buf.getvalue(), "")


class SimBusThreadingTests(unittest.TestCase):
    def test_concurrent_push_log_from_many_threads(self) -> None:
        bus = SimBus(echo=False)
        n_threads = 8
        n_per_thread = 50

        def worker(tag: int) -> None:
            for i in range(n_per_thread):
                bus.push_log(f"t{tag}-{i}")

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        _, logs, _ = bus.snapshot()
        self.assertEqual(len(logs), n_threads * n_per_thread)


if __name__ == "__main__":
    unittest.main()
