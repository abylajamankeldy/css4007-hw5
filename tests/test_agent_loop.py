"""Part 2 graph with the FAKE model (helpdesk/fake_model.py, keyword rules, not an LLM).

These tests check the graph plumbing - routing between agent and tools, the
loop limit, and that tools use the session's snapshot - not the quality of a
real model's decisions.
"""
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

from helpdesk.agent import MAX_TOOL_ROUNDS, build_agent_graph, run_turn
from helpdesk.fake_model import FakeHelpdeskModel

REQUEST = ("I enabled Wi-Fi and forgot the saved network, but Campus-Demo still will not "
           "connect. Check whether Wi-Fi is down. If it is operating, prepare a ticket about "
           "this problem.")


def _run(profile, tmp_path, message=REQUEST, model=None):
    app = build_agent_graph(model or FakeHelpdeskModel(), InMemorySaver(), InMemoryStore())
    return run_turn(app, message, user_id="demo-user-a", thread_id="t",
                    status_profile=profile, ticket_dir=str(tmp_path))


def _tools_called(result):
    return [c["name"] for step in result["trace"] for c in step.get("tool_requests", [])]


def test_operational_leads_to_ticket(tmp_path):
    # Catches: the tools node not looping back to the agent after a result.
    r = _run("normal", tmp_path)
    assert _tools_called(r) == ["check_service_status", "prepare_ticket"]
    assert r["ticket_file"] and len(list(tmp_path.iterdir())) == 1


def test_outage_leads_to_no_ticket(tmp_path):
    # Catches: tools using a hard-coded snapshot instead of the session's one.
    r = _run("wifi_outage", tmp_path)
    assert _tools_called(r) == ["check_service_status"]
    assert r["ticket_file"] is None and not list(tmp_path.iterdir())


def test_unknown_status_asks_user(tmp_path):
    r = _run("status_unavailable", tmp_path)
    assert r["ticket_file"] is None and "?" in r["reply"]


def test_ticket_records_checked_status_from_program(tmp_path):
    # Catches: service_status in the ticket coming from the model instead of the tool result.
    r = _run("normal", tmp_path)
    result = next(s["tool_results"]["prepare_ticket"] for s in r["trace"]
                  if "prepare_ticket" in s.get("tool_results", {}))
    assert result["ticket"]["service_status"] == "operational"
    assert result["ticket"]["user_id"] == "demo-user-a"


def test_loop_limit_stops_endless_tool_requests(tmp_path):
    # Catches: a model that always asks for tools running forever.
    r = _run("normal", tmp_path, "wifi", FakeHelpdeskModel(always_call_tools=True))
    assert r["loop_limit_reached"] is True
    assert r["tool_rounds"] == MAX_TOOL_ROUNDS
    assert r["trace"][-1]["node"] == "loop_limit"
