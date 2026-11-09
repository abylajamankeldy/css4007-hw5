"""Part 3 — where the two kinds of memory live on disk.

    local-storage/conversations.sqlite   CONVERSATION memory
        LangGraph SqliteSaver (checkpointer). After every graph step it saves the
        whole AgentState of one thread (messages, service_status_seen, ...),
        keyed by thread_id. Same thread_id after a restart -> same conversation.

    local-storage/user_memory.sqlite     USER memory
        LangGraph SqliteStore. Key-value items in namespace ("users", user_id),
        key "preferences" -> {"reply_language": "Kazakh"}. Shared by all threads
        of that user, invisible to other users.

Both are ordinary SQLite files, so they survive closing the program.
Delete the folder (python -m helpdesk.cli reset) to start again.
"""
import os
import shutil
import sqlite3
from contextlib import contextmanager
from pathlib import Path

from dotenv import load_dotenv
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.store.sqlite import SqliteStore

from helpdesk.data import REPO_ROOT


def storage_dir(path: str | os.PathLike | None = None) -> Path:
    load_dotenv()
    folder = Path(path or os.getenv("HELPDESK_STORAGE_DIR") or "local-storage")
    return folder if folder.is_absolute() else REPO_ROOT / folder


@contextmanager
def open_storage(path=None):
    """Yields (checkpointer, store). Closes both SQLite connections afterwards."""
    folder = storage_dir(path)
    folder.mkdir(parents=True, exist_ok=True)
    conv = sqlite3.connect(folder / "conversations.sqlite", check_same_thread=False)
    # SqliteStore opens its own transactions, so the connection must be in autocommit mode.
    users = sqlite3.connect(folder / "user_memory.sqlite", check_same_thread=False,
                            isolation_level=None)
    try:
        checkpointer = SqliteSaver(conv)
        checkpointer.setup()
        store = SqliteStore(users)
        store.setup()
        yield checkpointer, store
    finally:
        conv.close()
        users.close()


def reset_storage(path=None) -> Path:
    """Delete all conversation and user memory."""
    folder = storage_dir(path)
    if folder.exists():
        shutil.rmtree(folder)
    return folder
