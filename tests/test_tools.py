"""The three tools called DIRECTLY, without any agent or model."""
import json

import pytest

from helpdesk.tools import check_service_status, find_instructions, prepare_ticket


def test_find_instructions_wifi_and_password():
    # Catches: wrong file/keys read, or topic spelling "Wi-Fi" not normalised.
    assert find_instructions("Wi-Fi")["instruction_id"] == "GUIDE-WIFI"
    assert find_instructions("password")["instruction_id"] == "GUIDE-PASSWORD"


def test_find_instructions_unknown_topic():
    # Catches: inventing instructions for a topic the guide does not cover.
    out = find_instructions("printer")
    assert out["found"] is False and "steps" not in out


@pytest.mark.parametrize("profile,expected", [
    ("normal", "operational"), ("wifi_outage", "outage"), ("status_unavailable", "unknown"),
])
def test_status_depends_on_snapshot(profile, expected):
    # Catches: ignoring the selected snapshot, or treating "unknown" as operational.
    assert check_service_status("wifi", profile)["status"] == expected


def test_password_topic_maps_to_account_recovery():
    # Catches: looking up "password" instead of the account_recovery service.
    out = check_service_status("password", "recovery_outage")
    assert out["service"] == "account_recovery" and out["status"] == "outage"


def test_prepare_ticket_writes_template_fields(tmp_path):
    # Catches: missing template fields, ticket not saved, invented status.
    out = prepare_ticket("wifi", "Campus-Demo will not connect", ["Enabled Wi-Fi"],
                         user_id="demo-user-a", thread_id="t1", status_profile="normal",
                         service_status="operational", ticket_dir=tmp_path)
    assert out["created"]
    saved = json.loads((tmp_path / f"{out['ticket_id']}.json").read_text(encoding="utf-8"))
    assert set(saved) == {"ticket_id", "status", "user_id", "thread_id", "category", "summary",
                          "steps_tried", "service_status", "status_profile", "reference_ids"}
    assert saved["status"] == "draft" and saved["steps_tried"] == ["Enabled Wi-Fi"]
    assert saved["reference_ids"] == ["GUIDE-WIFI"]


def test_prepare_ticket_refuses_missing_details(tmp_path):
    # Catches: a ticket created with empty steps (details must be asked, not invented).
    out = prepare_ticket("wifi", "", [], user_id="u", thread_id="t",
                         status_profile="normal", ticket_dir=tmp_path)
    assert out["created"] is False
    assert "steps already tried" in out["missing"] and not list(tmp_path.iterdir())


def test_prepare_ticket_refuses_secrets(tmp_path):
    # Catches: storing a password in a ticket.
    out = prepare_ticket("password", "my password is hunter2", ["tried reset"], user_id="u",
                         thread_id="t", status_profile="normal", ticket_dir=tmp_path)
    assert out["created"] is False and not list(tmp_path.iterdir())
