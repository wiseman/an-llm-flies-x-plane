from __future__ import annotations

import io
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from sim_pilot.bus import FileLog, SimBus


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


class FileLogTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.log_path = Path(self._tmp.name) / "run.log"

    def test_write_and_read_back(self) -> None:
        log = FileLog(self.log_path)
        log.write("log", "bridge connected")
        log.write("radio", "[BROADCAST com1] tower, cessna 123")
        log.close()
        content = self.log_path.read_text()
        self.assertIn("[log] bridge connected", content)
        self.assertIn("[radio] [BROADCAST com1] tower, cessna 123", content)

    def test_timestamp_iso_format(self) -> None:
        import re
        log = FileLog(self.log_path)
        log.write("status", "hello")
        log.close()
        content = self.log_path.read_text()
        # ISO 8601 seconds precision
        self.assertRegex(content, r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2} \[status\] hello")

    def test_multi_line_text_keeps_continuation_lines_aligned(self) -> None:
        log = FileLog(self.log_path)
        log.write("status", "line1\nline2\nline3")
        log.close()
        lines = self.log_path.read_text().splitlines()
        self.assertEqual(len(lines), 3)
        # All three lines should include [status]
        for line in lines:
            self.assertIn("[status]", line)
        # Continuation lines should start with whitespace (no timestamp)
        self.assertTrue(lines[1].startswith(" "))
        self.assertTrue(lines[2].startswith(" "))

    def test_parent_directory_created_if_missing(self) -> None:
        nested_path = Path(self._tmp.name) / "a" / "b" / "c" / "run.log"
        log = FileLog(nested_path)
        log.write("log", "ok")
        log.close()
        self.assertTrue(nested_path.exists())

    def test_close_is_idempotent(self) -> None:
        log = FileLog(self.log_path)
        log.write("log", "one")
        log.close()
        log.close()  # should not raise

    def test_write_after_close_is_noop(self) -> None:
        log = FileLog(self.log_path)
        log.write("log", "before close")
        log.close()
        log.write("log", "after close")  # should not raise or crash
        content = self.log_path.read_text()
        self.assertIn("before close", content)
        self.assertNotIn("after close", content)


class SimBusFileSinkTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.log_path = Path(self._tmp.name) / "bus.log"

    def test_all_channels_written_to_file(self) -> None:
        bus = SimBus(echo=False, file_log=FileLog(self.log_path))
        bus.push_status("status line")
        bus.push_log("log line")
        bus.push_radio("radio line")
        bus.close()
        content = self.log_path.read_text()
        self.assertIn("[status] status line", content)
        self.assertIn("[log] log line", content)
        self.assertIn("[radio] radio line", content)

    def test_no_file_log_behaves_like_before(self) -> None:
        # Doesn't crash, doesn't try to open a file
        bus = SimBus(echo=False)
        bus.push_status("s")
        bus.push_log("l")
        bus.push_radio("r")
        bus.close()
        status, logs, radio = bus.snapshot()
        self.assertEqual(status, "s")
        self.assertEqual(logs, ["l"])
        self.assertEqual(radio, ["r"])

    def test_file_sink_survives_concurrent_pushes(self) -> None:
        bus = SimBus(echo=False, file_log=FileLog(self.log_path))
        n_threads = 4
        n_per_thread = 50

        def worker(tag: int) -> None:
            for i in range(n_per_thread):
                bus.push_log(f"t{tag}-{i}")

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        bus.close()
        content = self.log_path.read_text()
        # All lines present; no line is torn or missing
        total = sum(1 for _ in content.splitlines())
        self.assertEqual(total, n_threads * n_per_thread)


if __name__ == "__main__":
    unittest.main()
