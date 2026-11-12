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
import os                                   # os.getenv: чтение переменных окружения
import shutil                               # shutil.rmtree: удалить папку целиком
import sqlite3                              # встроенная в Python база данных в одном файле
from contextlib import contextmanager       # превращает функцию с yield в блок "with"
from pathlib import Path

from dotenv import load_dotenv              # читает файл .env в переменные окружения
from langgraph.checkpoint.sqlite import SqliteSaver   # checkpointer: память РАЗГОВОРА
from langgraph.store.sqlite import SqliteStore        # store: память ПОЛЬЗОВАТЕЛЯ

from helpdesk.data import REPO_ROOT


def storage_dir(path: str | os.PathLike | None = None) -> Path:
    load_dotenv()   # чтобы HELPDESK_STORAGE_DIR из .env был виден через os.getenv
    # Порядок выбора: явно переданный путь -> переменная окружения -> "local-storage".
    folder = Path(path or os.getenv("HELPDESK_STORAGE_DIR") or "local-storage")
    # Относительный путь считаем от корня репозитория, а не от текущей папки.
    return folder if folder.is_absolute() else REPO_ROOT / folder


@contextmanager
def open_storage(path=None):
    """Yields (checkpointer, store). Closes both SQLite connections afterwards."""
    folder = storage_dir(path)
    folder.mkdir(parents=True, exist_ok=True)   # создать папку, если её нет
    # check_same_thread=False: LangGraph может обращаться к базе из другого потока.
    conv = sqlite3.connect(folder / "conversations.sqlite", check_same_thread=False)
    # SqliteStore opens its own transactions, so the connection must be in autocommit mode.
    # (Без isolation_level=None была ошибка "cannot start a transaction within a transaction".)
    users = sqlite3.connect(folder / "user_memory.sqlite", check_same_thread=False,
                            isolation_level=None)
    try:
        checkpointer = SqliteSaver(conv)   # история тредов, ключ - thread_id
        checkpointer.setup()               # создать таблицы, если их ещё нет
        store = SqliteStore(users)         # настройки пользователей, ключ - ("users", user_id)
        store.setup()
        yield checkpointer, store          # отдаём обе памяти в блок "with open_storage() as (...)"
    finally:
        # finally выполняется всегда, даже если внутри блока with случилась ошибка.
        conv.close()
        users.close()


def reset_storage(path=None) -> Path:
    """Delete all conversation and user memory."""
    folder = storage_dir(path)
    if folder.exists():
        shutil.rmtree(folder)   # удаляем папку вместе с обоими файлами .sqlite
    return folder
