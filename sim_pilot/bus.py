from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
import threading
import time
from typing import TextIO


LOG_MAX_LINES = 1000
RADIO_MAX_LINES = 200


class FileLog:
    """Line-buffered append-only file sink with channel tags and timestamps.

    Thread-safe on its own (internal lock). The typical ``SimBus`` usage
    holds ``SimBus._lock`` around the file write so the in-memory deques
    and the file stay in the same order, but ``FileLog`` is independently
    callable for unit tests.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file: TextIO | None = self.path.open("a", encoding="utf-8", buffering=1)
        self._lock = threading.Lock()

    def write(self, channel: str, text: str) -> None:
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")
        with self._lock:
            if self._file is None:
                return
            lines = text.splitlines() or [""]
            for i, line in enumerate(lines):
                prefix = timestamp if i == 0 else " " * len(timestamp)
                self._file.write(f"{prefix} [{channel}] {line}\n")
            self._file.flush()

    def close(self) -> None:
        with self._lock:
            if self._file is not None:
                try:
                    self._file.close()
                finally:
                    self._file = None


@dataclass
class SimBus:
    """Thread-safe output channel for status, log, and radio regions.

    Set ``echo=True`` (default) to also print to stdout; set ``echo=False``
    when the TUI is rendering so stdout writes don't corrupt the display.

    If ``file_log`` is set, every push is also written to the backing file
    with a timestamp and channel tag (``status`` / ``log`` / ``radio``).
    The file write happens under ``SimBus._lock`` so ordering between the
    in-memory deques and the file is consistent under concurrent pushes.
    """

    echo: bool = True
    file_log: FileLog | None = None
    _status_line: str = ""
    _log_buffer: deque[str] = field(default_factory=lambda: deque(maxlen=LOG_MAX_LINES))
    _radio_buffer: deque[str] = field(default_factory=lambda: deque(maxlen=RADIO_MAX_LINES))
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def push_status(self, text: str) -> None:
        with self._lock:
            self._status_line = text
            if self.file_log is not None:
                self.file_log.write("status", text)
        if self.echo:
            print(text, flush=True)

    def push_log(self, text: str) -> None:
        with self._lock:
            self._log_buffer.append(text)
            if self.file_log is not None:
                self.file_log.write("log", text)
        if self.echo:
            print(text, flush=True)

    def push_radio(self, text: str) -> None:
        with self._lock:
            self._radio_buffer.append(text)
            if self.file_log is not None:
                self.file_log.write("radio", text)
        if self.echo:
            print(text, flush=True)

    def snapshot(self) -> tuple[str, list[str], list[str]]:
        with self._lock:
            return self._status_line, list(self._log_buffer), list(self._radio_buffer)

    def log_tail(self, n: int) -> list[str]:
        with self._lock:
            if n >= len(self._log_buffer):
                return list(self._log_buffer)
            return list(self._log_buffer)[-n:]

    def radio_tail(self, n: int) -> list[str]:
        with self._lock:
            if n >= len(self._radio_buffer):
                return list(self._radio_buffer)
            return list(self._radio_buffer)[-n:]

    def close(self) -> None:
        with self._lock:
            if self.file_log is not None:
                self.file_log.close()
