from __future__ import annotations

from dataclasses import dataclass, field
import json
import queue
import threading
import time
from typing import Any, Literal

from sim_pilot.llm.responses_client import ResponsesClient
from sim_pilot.llm.tools import TOOL_SCHEMAS, ToolContext, dispatch_tool


SYSTEM_PROMPT = """\
You are the autonomous pilot agent for a Cessna 172 in X-Plane 12. A deterministic
pilot core runs at ~10 Hz in the background and executes profile-based guidance; your
job is to engage the right profiles and respond to operator and ATC messages.

Profiles you can engage (via the engage_* tools):
- heading_hold: lateral-axis heading hold.
- altitude_hold: vertical-axis altitude hold (TECS).
- speed_hold: speed axis target.
- pattern_fly: full deterministic mission pilot, takeoff through landing, using the
  configured airport's phase machine. Engage this when the operator wants you to
  actually fly a complete flight.
- approach_runway: (stub, not yet implemented).
- route_follow: (stub, not yet implemented).

Incoming messages are tagged:
  [OPERATOR] ... — the human operator you are flying for.
  [ATC] ...      — air traffic control.

Use get_status() to check current aircraft state. Call sleep() to explicitly end your
turn; the control loop keeps flying the active profiles while you wait for the next
message. Use the pattern-event tools (extend_downwind, turn_base_now, go_around,
cleared_to_land, join_pattern) only when pattern_fly is active.

Do not fabricate actions. Every change to the flight state must go through a tool
call. Reply in plain text to acknowledge actions or explain what you are doing.
"""


MAX_INPUT_CHARS = 60_000
DEFAULT_PER_REQUEST_TIMEOUT_S = 60.0
DEFAULT_TOTAL_WALL_BUDGET_S = 120.0


@dataclass(slots=True, frozen=True)
class IncomingMessage:
    source: Literal["operator", "atc"]
    text: str


def _operator_item(text: str) -> dict[str, Any]:
    return {
        "role": "user",
        "content": [{"type": "input_text", "text": f"[OPERATOR] {text}"}],
    }


def _atc_item(text: str) -> dict[str, Any]:
    return {
        "role": "user",
        "content": [{"type": "input_text", "text": f"[ATC] {text}"}],
    }


def _system_item(text: str) -> dict[str, Any]:
    return {
        "role": "system",
        "content": [{"type": "input_text", "text": text}],
    }


@dataclass
class Conversation:
    system_prompt: str
    pinned_items: list[dict[str, Any]] = field(default_factory=list)
    rotating_items: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.pinned_items:
            self.pinned_items = [_system_item(self.system_prompt)]

    def append_operator_message(self, text: str) -> None:
        self.rotating_items.append(_operator_item(text))

    def append_atc_message(self, text: str) -> None:
        self.rotating_items.append(_atc_item(text))

    def append_response_items(self, output_items: list[dict[str, Any]]) -> None:
        for item in output_items:
            if isinstance(item, dict):
                self.rotating_items.append(item)

    def append_function_call_output(self, call_id: str, output: str) -> None:
        if not call_id:
            raise ValueError("function_call_output requires a call_id")
        self.rotating_items.append(
            {
                "type": "function_call_output",
                "call_id": call_id,
                "output": output,
            }
        )

    def build_input(self, active_profiles_summary: str) -> list[dict[str, Any]]:
        summary_item = _system_item(
            f"Active profiles: {active_profiles_summary}" if active_profiles_summary else "Active profiles: (none)"
        )
        return list(self.pinned_items) + [summary_item] + list(self.rotating_items)

    def total_char_count(self) -> int:
        return sum(
            len(json.dumps(item, separators=(",", ":")))
            for item in (self.pinned_items + self.rotating_items)
        )

    def compact_if_needed(self, threshold_chars: int = MAX_INPUT_CHARS) -> int:
        """Drop the oldest full turn until under threshold. A 'turn' spans from one
        user message (operator or ATC) up to but not including the next one."""
        dropped = 0
        while self.total_char_count() > threshold_chars:
            turn_end = self._find_first_full_turn_end()
            if turn_end is None:
                break
            del self.rotating_items[0:turn_end]
            dropped += turn_end
        return dropped

    def _find_first_full_turn_end(self) -> int | None:
        """Return the index just past the end of the oldest full turn in rotating_items,
        or None if no full turn is present (we won't compact a partial in-progress turn)."""
        first_user = self._first_user_message_index()
        if first_user is None:
            return None
        next_user = self._next_user_message_index(first_user + 1)
        if next_user is None:
            return None
        return next_user

    def _first_user_message_index(self) -> int | None:
        for i, item in enumerate(self.rotating_items):
            if item.get("role") == "user":
                return i
        return None

    def _next_user_message_index(self, start: int) -> int | None:
        for i in range(start, len(self.rotating_items)):
            item = self.rotating_items[i]
            if item.get("role") == "user":
                return i
        return None


def _function_calls_from_output(output_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [item for item in output_items if isinstance(item, dict) and item.get("type") == "function_call"]


def _format_profiles_summary(names: list[str]) -> str:
    if not names:
        return "(none)"
    return ", ".join(names)


def run_conversation_loop(
    *,
    client: ResponsesClient,
    tool_context: ToolContext,
    input_queue: "queue.Queue[IncomingMessage]",
    stop_event: threading.Event | None = None,
    per_request_timeout_s: float = DEFAULT_PER_REQUEST_TIMEOUT_S,
    total_wall_budget_s: float = DEFAULT_TOTAL_WALL_BUDGET_S,
) -> None:
    conversation = Conversation(system_prompt=SYSTEM_PROMPT)
    while stop_event is None or not stop_event.is_set():
        try:
            message = input_queue.get(timeout=0.5)
        except queue.Empty:
            continue
        try:
            _handle_message(
                conversation=conversation,
                client=client,
                tool_context=tool_context,
                message=message,
                per_request_timeout_s=per_request_timeout_s,
                total_wall_budget_s=total_wall_budget_s,
            )
        except Exception as exc:
            print(f"[llm-worker] error handling message: {exc!r}", flush=True)


def _handle_message(
    *,
    conversation: Conversation,
    client: ResponsesClient,
    tool_context: ToolContext,
    message: IncomingMessage,
    per_request_timeout_s: float,
    total_wall_budget_s: float,
) -> None:
    if message.source == "operator":
        conversation.append_operator_message(message.text)
    else:
        conversation.append_atc_message(message.text)
    conversation.compact_if_needed()

    deadline = time.monotonic() + total_wall_budget_s
    while True:
        if time.monotonic() >= deadline:
            print("[llm-worker] wall-clock budget exceeded; ending turn", flush=True)
            return
        profiles_summary = _format_profiles_summary(tool_context.pilot.list_profile_names())
        input_items = conversation.build_input(active_profiles_summary=profiles_summary)
        remaining = deadline - time.monotonic()
        request_timeout = min(per_request_timeout_s, remaining)
        response = client.create_response(
            input_items=input_items,
            tools=TOOL_SCHEMAS,
            timeout_s=request_timeout,
        )
        output_items = response.get("output", [])
        if not isinstance(output_items, list):
            print(f"[llm-worker] unexpected output shape: {response!r}", flush=True)
            return
        conversation.append_response_items(output_items)
        function_calls = _function_calls_from_output(output_items)
        if not function_calls:
            _print_assistant_text(output_items)
            return
        slept = False
        for call in function_calls:
            call_id = call.get("call_id") or call.get("id") or ""
            if not isinstance(call_id, str) or not call_id:
                print(f"[llm-worker] function_call missing call_id: {call!r}", flush=True)
                continue
            name = call.get("name", "?")
            if name == "sleep":
                slept = True
                result = dispatch_tool(call, tool_context)
            else:
                result = dispatch_tool(call, tool_context)
            print(f"[llm-worker] tool {name} -> {result}", flush=True)
            conversation.append_function_call_output(call_id, result)
        if slept:
            return


def _print_assistant_text(output_items: list[dict[str, Any]]) -> None:
    for item in output_items:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "message":
            content = item.get("content")
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "output_text":
                        text = part.get("text")
                        if isinstance(text, str) and text.strip():
                            print(f"[llm] {text}", flush=True)
