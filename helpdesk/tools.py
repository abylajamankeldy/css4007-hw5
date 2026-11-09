"""Part 2 — the helpdesk tools as ordinary Python functions.

Each tool works WITHOUT the agent: you can import it and call it directly
(see `python -m helpdesk.tools`). The agent graph (helpdesk/agent.py) only
decides WHEN to call them and with WHICH arguments.

Some arguments are never chosen by the model. They are filled in by our own
program from the session (the person who started the program chose them):
  * status_profile  - which snapshot of service_status.json is active
  * user_id, thread_id - who is talking and in which conversation
  * store           - the persistent user-memory store
This way the model cannot load another user's data or switch the snapshot.
"""
import json
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

from helpdesk.data import REPO_ROOT, helpdesk, service_status, ticket_template

DEFAULT_TICKET_DIR = REPO_ROOT / "outputs" / "tickets"

# Words that suggest the user typed an actual secret into the ticket text.
SECRET_PATTERN = re.compile(r"(password|passcode|recovery code|api key)\s*(is|:|=)\s*\S+", re.I)


def _normalise_topic(topic: str) -> str | None:
    """'Wi-Fi', 'wifi', 'WIFI' -> 'wifi'; 'password', 'account_recovery' -> 'password'."""
    text = (topic or "").lower().strip()
    if text in ("account_recovery", "account recovery"):
        return "password"
    for name, aliases in helpdesk()["topic_aliases"].items():
        if text == name or any(alias in text for alias in aliases):
            return name
    return None


# ------------------------------------------------------------------ tool 1

def find_instructions(topic: str) -> dict:
    """Look up the Wi-Fi or password instructions in data/helpdesk.json (retrieval)."""
    name = _normalise_topic(topic)
    if name is None:
        return {"found": False, "topic": topic,
                "available_topics": list(helpdesk()["instructions"]),
                "source": "data/helpdesk.json"}
    entry = helpdesk()["instructions"][name]
    return {"found": True, "topic": name, "instruction_id": entry["instruction_id"],
            "title": entry["title"], "steps": entry["steps"],
            "source": "data/helpdesk.json"}


# ------------------------------------------------------------------ tool 2

def check_service_status(service: str, status_profile: str) -> dict:
    """Look up one service in the selected snapshot of data/service_status.json.

    `service` may be a service name ("wifi", "account_recovery") or a topic
    ("password" -> "account_recovery"). status "unknown" means the status could
    NOT be determined: it is neither "operating" nor a confirmed outage.
    """
    topic = _normalise_topic(service)
    if topic is None:
        return {"service": service, "status": "unknown", "known_service": False,
                "message": "This service is not in the helpdesk data.",
                "status_profile": status_profile, "source": "data/service_status.json"}
    service_name = helpdesk()["topic_services"][topic]
    profiles = service_status()["profiles"]
    if status_profile not in profiles:
        raise ValueError(f"unknown status profile {status_profile!r}; "
                         f"choose one of {list(profiles)}")
    entry = profiles[status_profile][service_name]
    return {"service": service_name, "status": entry["status"], "known_service": True,
            "message": entry["message"], "status_profile": status_profile,
            "source": "data/service_status.json"}


# ------------------------------------------------------------------ tool 3

def prepare_ticket(category: str, summary: str, steps_tried: list[str], *,
                   user_id: str, thread_id: str, status_profile: str,
                   service_status: str = "not_checked",
                   ticket_dir: str | Path | None = None) -> dict:
    """Write a LOCAL ticket draft (JSON file). Nothing is sent anywhere.

    The draft follows data/ticket_template.json. If the problem description or
    the steps already tried are missing, NO file is written: the tool returns
    what is missing so the assistant asks the user instead of inventing it.
    `service_status` is filled by our program from an earlier
    check_service_status result, or "not_checked" - never by the model.
    """
    topic = _normalise_topic(category)
    steps = [s.strip() for s in (steps_tried or []) if s and s.strip()]
    missing = []
    if topic is None:
        missing.append("which problem (Wi-Fi or password)")
    if not (summary or "").strip():
        missing.append("problem description")
    if not steps:
        missing.append("steps already tried")
    if missing:
        return {"created": False, "missing": missing,
                "advice": helpdesk()["ticket_policy"]["unknown_information"]}

    if SECRET_PATTERN.search(summary + " " + " ".join(steps)):
        return {"created": False, "missing": [],
                "advice": helpdesk()["ticket_policy"]["secret_information"]
                + " Remove the secret and describe only the error."}

    ticket = json.loads(json.dumps(ticket_template()["ticket"]))  # fresh copy of the template
    ticket.update({
        "ticket_id": "DRAFT-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
                     + "-" + uuid.uuid4().hex[:4].upper(),
        "status": "draft",
        "user_id": user_id,
        "thread_id": thread_id,
        "category": topic,
        "summary": summary.strip(),
        "steps_tried": steps,
        "service_status": service_status,
        "status_profile": status_profile,
        "reference_ids": [helpdesk()["instructions"][topic]["instruction_id"]],
    })

    folder = Path(ticket_dir or os.getenv("HELPDESK_TICKET_DIR") or DEFAULT_TICKET_DIR)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{ticket['ticket_id']}.json"
    path.write_text(json.dumps(ticket, indent=2, ensure_ascii=False), encoding="utf-8")
    try:
        shown = str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        shown = str(path)
    return {"created": True, "ticket_id": ticket["ticket_id"], "file": shown,
            "ticket": ticket, "note": "Local draft only. Nothing was sent to a helpdesk."}


# ------------------------------------------------------------------ memory tool (Part 3)

def remember_preference(language: str, *, user_id: str, store) -> dict:
    """Save the user's preferred reply language in USER memory (persistent store).

    The namespace is ("users", user_id). user_id comes from the session, so a
    model cannot write into another user's memory.
    """
    language = (language or "").strip()
    if not language:
        return {"saved": False, "reason": "no language given"}
    namespace = ("users", user_id)
    current = store.get(namespace, "preferences")
    prefs = dict(current.value) if current else {}
    prefs["reply_language"] = language
    store.put(namespace, "preferences", prefs)
    return {"saved": True, "user_id": user_id, "preferences": prefs,
            "storage": "LangGraph SqliteStore, namespace ('users', user_id)"}


def read_preferences(user_id: str, store) -> dict:
    """Read user memory. Returns {} if nothing has been saved for this user."""
    item = store.get(("users", user_id), "preferences") if store else None
    return dict(item.value) if item else {}


if __name__ == "__main__":
    # Direct calls, no agent and no model involved.
    import tempfile
    print("find_instructions('Wi-Fi') ->")
    print(json.dumps(find_instructions("Wi-Fi"), indent=2))
    for profile in ("normal", "wifi_outage", "status_unavailable"):
        print(f"\ncheck_service_status('wifi', {profile!r}) ->")
        print(json.dumps(check_service_status("wifi", profile), indent=2))
    print("\ncheck_service_status('password', 'recovery_outage') ->")
    print(json.dumps(check_service_status("password", "recovery_outage"), indent=2))
    print("\nprepare_ticket(...) with no steps ->")
    print(json.dumps(prepare_ticket("wifi", "Cannot connect", [], user_id="demo-user-a",
                                    thread_id="direct", status_profile="normal"), indent=2))
    with tempfile.TemporaryDirectory() as tmp:
        print("\nprepare_ticket(...) complete ->")
        print(json.dumps(prepare_ticket(
            "wifi", "Campus-Demo will not connect.",
            ["Enabled Wi-Fi", "Forgot the saved network and reconnected"],
            user_id="demo-user-a", thread_id="direct", status_profile="normal",
            service_status="operational", ticket_dir=tmp), indent=2))
