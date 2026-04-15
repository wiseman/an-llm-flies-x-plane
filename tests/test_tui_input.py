from __future__ import annotations

import unittest

from sim_pilot.tui import parse_input_source


class ParseInputSourceTests(unittest.TestCase):
    def test_plain_text_defaults_to_operator(self) -> None:
        self.assertEqual(parse_input_source("take off and fly west"), ("operator", "take off and fly west"))

    def test_atc_colon_prefix(self) -> None:
        self.assertEqual(
            parse_input_source("atc: N1234, what runway are you on?"),
            ("atc", "N1234, what runway are you on?"),
        )

    def test_atc_bracket_prefix(self) -> None:
        self.assertEqual(
            parse_input_source("[ATC] cleared for takeoff runway 16L"),
            ("atc", "cleared for takeoff runway 16L"),
        )

    def test_operator_colon_prefix(self) -> None:
        self.assertEqual(
            parse_input_source("operator: standby"),
            ("operator", "standby"),
        )

    def test_operator_bracket_prefix(self) -> None:
        self.assertEqual(
            parse_input_source("[operator] ready"),
            ("operator", "ready"),
        )

    def test_case_insensitive_prefix_detection(self) -> None:
        self.assertEqual(parse_input_source("AtC: hello"), ("atc", "hello"))
        self.assertEqual(parse_input_source("[Operator] hi"), ("operator", "hi"))

    def test_whitespace_stripped_after_prefix(self) -> None:
        self.assertEqual(parse_input_source("atc:    spaced out"), ("atc", "spaced out"))


if __name__ == "__main__":
    unittest.main()
