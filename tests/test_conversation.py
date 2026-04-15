from __future__ import annotations

import json
import queue
import threading
import unittest
from typing import Any

from sim_pilot.core.config import load_default_config_bundle
from sim_pilot.core.mission_manager import PilotCore
from sim_pilot.llm.conversation import (
    Conversation,
    IncomingMessage,
    SYSTEM_PROMPT,
    run_conversation_loop,
)
from sim_pilot.llm.tools import ToolContext


class StubClient:
    """Responses API stub — returns pre-scripted response payloads in order."""

    def __init__(self, scripted: list[dict[str, Any]]) -> None:
        self._scripted = list(scripted)
        self.calls: list[dict[str, Any]] = []

    def create_response(
        self,
        *,
        input_items: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        timeout_s: float,
    ) -> dict[str, Any]:
        self.calls.append({"input_items": list(input_items), "tools": tools, "timeout_s": timeout_s})
        if not self._scripted:
            return {"output": []}
        return self._scripted.pop(0)


def _make_ctx() -> ToolContext:
    config = load_default_config_bundle()
    pilot = PilotCore(config)
    return ToolContext(pilot=pilot, bridge=None, config=config, recent_broadcasts=[])


def _function_call(name: str, arguments: dict[str, Any], call_id: str = "call_1") -> dict[str, Any]:
    return {
        "type": "function_call",
        "call_id": call_id,
        "name": name,
        "arguments": json.dumps(arguments),
    }


def _text_message(text: str) -> dict[str, Any]:
    return {
        "type": "message",
        "role": "assistant",
        "content": [{"type": "output_text", "text": text}],
    }


class ConversationStateTests(unittest.TestCase):
    def test_append_operator_message_adds_tagged_user_item(self) -> None:
        conv = Conversation(system_prompt="sys")
        conv.append_operator_message("take off")
        self.assertEqual(len(conv.rotating_items), 1)
        item = conv.rotating_items[0]
        self.assertEqual(item["role"], "user")
        self.assertIn("[OPERATOR]", item["content"][0]["text"])

    def test_append_atc_message_adds_tagged_user_item(self) -> None:
        conv = Conversation(system_prompt="sys")
        conv.append_atc_message("squawk 1234")
        self.assertIn("[ATC]", conv.rotating_items[0]["content"][0]["text"])

    def test_append_function_call_output_requires_call_id(self) -> None:
        conv = Conversation(system_prompt="sys")
        with self.assertRaises(ValueError):
            conv.append_function_call_output("", "result")

    def test_build_input_includes_pinned_profiles_summary_and_rotating(self) -> None:
        conv = Conversation(system_prompt="sys")
        conv.append_operator_message("hi")
        items = conv.build_input(active_profiles_summary="heading_hold, idle_vertical, idle_speed")
        self.assertEqual(items[0]["role"], "system")
        self.assertEqual(items[1]["role"], "system")
        self.assertIn("Active profiles", items[1]["content"][0]["text"])
        self.assertIn("heading_hold", items[1]["content"][0]["text"])
        self.assertEqual(items[2]["role"], "user")


class CompactionTests(unittest.TestCase):
    def test_compaction_drops_oldest_full_turn(self) -> None:
        conv = Conversation(system_prompt="sys")
        # Two turns, each a user msg + one function_call + output
        for turn in range(2):
            conv.append_operator_message(f"msg{turn}")
            conv.rotating_items.append(_function_call("noop", {}, call_id=f"c{turn}"))
            conv.append_function_call_output(f"c{turn}", "ok")
        self.assertEqual(len([i for i in conv.rotating_items if i.get("role") == "user"]), 2)
        # Force compaction by using a very small threshold
        dropped = conv.compact_if_needed(threshold_chars=50)
        self.assertGreater(dropped, 0)
        # The oldest full turn should be gone; only the second turn's items remain.
        user_items = [i for i in conv.rotating_items if i.get("role") == "user"]
        self.assertEqual(len(user_items), 1)
        self.assertIn("msg1", user_items[0]["content"][0]["text"])

    def test_compaction_leaves_partial_turn_alone(self) -> None:
        conv = Conversation(system_prompt="sys")
        conv.append_operator_message("only")
        conv.rotating_items.append(_function_call("noop", {}, call_id="c0"))
        conv.append_function_call_output("c0", "ok")
        # No second user message — there's no 'full turn' to drop.
        dropped = conv.compact_if_needed(threshold_chars=10)
        self.assertEqual(dropped, 0)


class RunConversationLoopTests(unittest.TestCase):
    def test_single_tool_call_then_text_ends_turn(self) -> None:
        ctx = _make_ctx()
        script = [
            {"output": [_function_call("engage_heading_hold", {"heading_deg": 270.0})]},
            {"output": [_text_message("heading 270 engaged")]},
        ]
        client = StubClient(script)
        q: "queue.Queue[IncomingMessage]" = queue.Queue()
        q.put(IncomingMessage(source="operator", text="fly heading 270"))
        stop = threading.Event()

        worker = threading.Thread(
            target=run_conversation_loop,
            kwargs={
                "client": client,
                "tool_context": ctx,
                "input_queue": q,
                "stop_event": stop,
            },
            daemon=True,
        )
        worker.start()
        # Wait for script to drain
        for _ in range(50):
            if len(client.calls) >= 2:
                break
            threading.Event().wait(0.05)
        stop.set()
        worker.join(timeout=2.0)

        self.assertEqual(len(client.calls), 2)
        self.assertIn("heading_hold", ctx.pilot.list_profile_names())

    def test_sleep_tool_ends_turn_without_second_post(self) -> None:
        ctx = _make_ctx()
        script = [
            {"output": [_function_call("sleep", {})]},
        ]
        client = StubClient(script)
        q: "queue.Queue[IncomingMessage]" = queue.Queue()
        q.put(IncomingMessage(source="operator", text="standby"))
        stop = threading.Event()

        worker = threading.Thread(
            target=run_conversation_loop,
            kwargs={
                "client": client,
                "tool_context": ctx,
                "input_queue": q,
                "stop_event": stop,
            },
            daemon=True,
        )
        worker.start()
        for _ in range(50):
            if len(client.calls) >= 1:
                break
            threading.Event().wait(0.05)
        # Give it a moment to confirm no second call
        threading.Event().wait(0.2)
        stop.set()
        worker.join(timeout=2.0)

        self.assertEqual(len(client.calls), 1)

    def test_text_only_first_response_ends_turn_immediately(self) -> None:
        ctx = _make_ctx()
        script = [{"output": [_text_message("acknowledged")]}]
        client = StubClient(script)
        q: "queue.Queue[IncomingMessage]" = queue.Queue()
        q.put(IncomingMessage(source="atc", text="maintain altitude"))
        stop = threading.Event()

        worker = threading.Thread(
            target=run_conversation_loop,
            kwargs={
                "client": client,
                "tool_context": ctx,
                "input_queue": q,
                "stop_event": stop,
            },
            daemon=True,
        )
        worker.start()
        for _ in range(50):
            if len(client.calls) >= 1:
                break
            threading.Event().wait(0.05)
        stop.set()
        worker.join(timeout=2.0)

        self.assertEqual(len(client.calls), 1)

    def test_system_prompt_is_pinned_at_head_of_input(self) -> None:
        ctx = _make_ctx()
        script = [{"output": [_text_message("ok")]}]
        client = StubClient(script)
        q: "queue.Queue[IncomingMessage]" = queue.Queue()
        q.put(IncomingMessage(source="operator", text="status?"))
        stop = threading.Event()

        worker = threading.Thread(
            target=run_conversation_loop,
            kwargs={
                "client": client,
                "tool_context": ctx,
                "input_queue": q,
                "stop_event": stop,
            },
            daemon=True,
        )
        worker.start()
        for _ in range(50):
            if len(client.calls) >= 1:
                break
            threading.Event().wait(0.05)
        stop.set()
        worker.join(timeout=2.0)

        first_call = client.calls[0]
        items = first_call["input_items"]
        self.assertEqual(items[0]["role"], "system")
        self.assertIn(SYSTEM_PROMPT[:40], items[0]["content"][0]["text"])


if __name__ == "__main__":
    unittest.main()
