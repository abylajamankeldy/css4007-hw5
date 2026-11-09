"""Loading the supplied fixtures from data/.

Every other module reads the JSON files through these functions, so there is
exactly one place that knows where the files live.
"""
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"


def _load(name: str) -> dict:
    with open(DATA_DIR / name, encoding="utf-8") as f:
        return json.load(f)


def helpdesk() -> dict:
    """data/helpdesk.json: instructions, topic aliases, topic -> service map, ticket policy."""
    return _load("helpdesk.json")


def service_status() -> dict:
    """data/service_status.json: four named status snapshots (profiles)."""
    return _load("service_status.json")


def ticket_template() -> dict:
    """data/ticket_template.json: the fields of a local ticket draft."""
    return _load("ticket_template.json")


def scenarios() -> list[dict]:
    """data/scenarios.json: the fourteen demonstration scenarios S01-S14."""
    return _load("scenarios.json")["scenarios"]


def format_instructions(topic: str) -> str:
    """Turn the instructions of one topic ('wifi' or 'password') into readable text."""
    entry = helpdesk()["instructions"][topic]
    lines = [f"{entry['title']} ({entry['instruction_id']}):"]
    lines += [f"{i}. {step}" for i, step in enumerate(entry["steps"], start=1)]
    return "\n".join(lines)
