"""Part 1 routing. No model, no key."""
import pytest

from helpdesk.part1_graph import run_with_trace

CASES = [
    # message, original route, route with clarification
    ("I cannot connect to Wi-Fi.", "wifi_help", "wifi_help"),
    ("I forgot my PASSWORD.", "password_help", "password_help"),
    ("My laptop screen is broken.", "other_help", "other_help"),
    ("My Wi-Fi password does not work.", "wifi_help", "clarify_problem"),
    ("WIFI is slow", "wifi_help", "wifi_help"),
]


@pytest.mark.parametrize("message,original,_", CASES)
def test_original_graph_routes(message, original, _):
    # Catches: wrong rule order, forgetting lower(), "wi-fi" spelling not matched.
    trace = run_with_trace(message, "original")
    assert trace["route"] == ["START", "read_request", original, "END"]


@pytest.mark.parametrize("message,_,changed", CASES)
def test_changed_graph_routes(message, _, changed):
    # Catches: the clarify branch missing or firing for single-topic messages.
    trace = run_with_trace(message, "with_clarification")
    assert trace["route"] == ["START", "read_request", changed, "END"]


def test_state_after_each_node():
    # Catches: a node overwriting `message`, or `reply` set before a help node runs.
    trace = run_with_trace("I cannot connect to Wi-Fi.")
    first, second = trace["steps"]
    assert first["state_after"]["message"] == "I cannot connect to Wi-Fi."
    assert first["state_after"]["category"] == "wifi"
    assert "reply" not in first["state_after"]
    assert "GUIDE-WIFI" in second["state_after"]["reply"]


def test_other_help_asks_question():
    reply = run_with_trace("My laptop screen is broken.")["final_state"]["reply"]
    assert "Wi-Fi" in reply and "password" in reply and "?" in reply
