from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import threading


LOG_MAX_LINES = 1000
RADIO_MAX_LINES = 200


@dataclass
class SimBus:
    """Thread-safe output channel for status, log, and radio regions.

    Set ``echo=True`` (default) to also print to stdout; set ``echo=False``
    when the TUI is rendering so stdout writes don't corrupt the display.
    """

    echo: bool = True
    _status_line: str = ""
    _log_buffer: deque[str] = field(default_factory=lambda: deque(maxlen=LOG_MAX_LINES))
    _radio_buffer: deque[str] = field(default_factory=lambda: deque(maxlen=RADIO_MAX_LINES))
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def push_status(self, text: str) -> None:
        with self._lock:
            self._status_line = text
        if self.echo:
            print(text, flush=True)

    def push_log(self, text: str) -> None:
        with self._lock:
            self._log_buffer.append(text)
        if self.echo:
            print(text, flush=True)

    def push_radio(self, text: str) -> None:
        with self._lock:
            self._radio_buffer.append(text)
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
