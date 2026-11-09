"""Part 3 memory with the FAKE model and real SQLite files in a temporary folder."""
import os
import subprocess
import sys

from helpdesk.agent import build_agent_graph, run_turn
from helpdesk.data import REPO_ROOT
from helpdesk.fake_model import FakeHelpdeskModel
from helpdesk.storage import open_storage
from helpdesk.tools import read_preferences


def _say(storage, user, thread, message):
    """Runs the CLI in a NEW Python process - a real restart."""
    cmd = [sys.executable, "-m", "helpdesk.cli", "say", "--model", "fake", "--storage",
           str(storage), "--user", user, "--thread", thread, "--profile", "normal", message]
    out = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, encoding="utf-8",
                         env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    assert out.returncode == 0, out.stderr
    return out.stdout


def test_conversation_survives_process_restart(tmp_path):
    # Catches: an in-memory checkpointer, or a thread_id that changes between runs.
    _say(tmp_path, "demo-user-a", "restart-a",
         "My issue is Wi-Fi. I enabled it and tried reconnecting, but it still fails.")
    second = _say(tmp_path, "demo-user-a", "restart-a", "Which problem and steps did we discuss?")
    assert "tried reconnecting" in second
    # 4 = user message, model tool request, tool result, model answer from the first process
    assert "4 earlier message(s) loaded" in second


def test_preference_survives_restart_in_new_thread(tmp_path):
    # Catches: preference stored in the thread state instead of user memory.
    _say(tmp_path, "demo-user-a", "pref-1", "Please remember that I prefer replies in Kazakh.")
    second = _say(tmp_path, "demo-user-a", "pref-2", "Which language did I ask you to use?")
    assert "reply in Kazakh" in second
    assert "0 earlier message(s)" in second            # the new thread starts empty


def test_users_are_separated(tmp_path):
    # Catches: a store namespace that ignores user_id (memory leaking between users).
    with open_storage(tmp_path) as (checkpointer, store):
        app = build_agent_graph(FakeHelpdeskModel(), checkpointer, store)
        run_turn(app, "Please remember that I prefer replies in Kazakh. I have a Wi-Fi problem.",
                 user_id="demo-user-a", thread_id="isolation-a", status_profile="normal",
                 ticket_dir=str(tmp_path))
        r = run_turn(app, "Have I told you a preferred language or a previous problem?",
                     user_id="demo-user-b", thread_id="isolation-b", status_profile="normal",
                     ticket_dir=str(tmp_path))
        assert read_preferences("demo-user-a", store) == {"reply_language": "Kazakh"}
        assert read_preferences("demo-user-b", store) == {}
        assert "Kazakh" not in r["reply"] and "Wi-Fi" not in r["reply"]
        assert r["messages_in_thread"] == 2           # only user b's own two messages


def test_threads_are_separated(tmp_path):
    # Catches: all conversations written into one thread.
    with open_storage(tmp_path) as (checkpointer, store):
        app = build_agent_graph(FakeHelpdeskModel(), checkpointer, store)
        kw = dict(user_id="demo-user-a", status_profile="normal", ticket_dir=str(tmp_path))
        run_turn(app, "I cannot connect to Wi-Fi.", thread_id="t1", **kw)
        r = run_turn(app, "Which steps did I already try?", thread_id="t2", **kw)
        assert "Nothing has been recorded" in r["reply"]
