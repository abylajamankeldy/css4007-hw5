"""Loading the supplied fixtures from data/.

Every other module reads the JSON files through these functions, so there is
exactly one place that knows where the files live.
"""
import json                 # json: превращает текст JSON в словари/списки Python
from pathlib import Path    # Path: путь к файлу, одинаково работает на Windows и Linux

# __file__ = путь к ЭТОМУ файлу (helpdesk/data.py).
# .resolve() делает путь полным, .parent.parent поднимается на 2 уровня вверх:
# helpdesk/data.py -> helpdesk/ -> корень репозитория.
REPO_ROOT = Path(__file__).resolve().parent.parent
# Оператор "/" у Path склеивает части пути: <корень>/data
DATA_DIR = REPO_ROOT / "data"


def _load(name: str) -> dict:
    # "_" в начале имени = внутренняя функция, снаружи модуля её не вызывают.
    # with ... as f: файл гарантированно закроется после блока, даже при ошибке.
    # encoding="utf-8" обязателен: на Windows по умолчанию другая кодировка.
    with open(DATA_DIR / name, encoding="utf-8") as f:
        return json.load(f)  # читает весь файл и возвращает словарь


def helpdesk() -> dict:
    """data/helpdesk.json: instructions, topic aliases, topic -> service map, ticket policy."""
    # Справочник: инструкции Wi-Fi и пароля, слова-синонимы тем,
    # какая тема к какому сервису относится, правила для тикетов.
    return _load("helpdesk.json")


def service_status() -> dict:
    """data/service_status.json: four named status snapshots (profiles)."""
    # Четыре снимка статуса: normal, wifi_outage, recovery_outage, status_unavailable.
    return _load("service_status.json")


def ticket_template() -> dict:
    """data/ticket_template.json: the fields of a local ticket draft."""
    # Шаблон черновика тикета: какие поля в нём должны быть.
    return _load("ticket_template.json")


def scenarios() -> list[dict]:
    """data/scenarios.json: the fourteen demonstration scenarios S01-S14."""
    # В файле словарь {"description": ..., "scenarios": [...]}; берём только список.
    return _load("scenarios.json")["scenarios"]


def format_instructions(topic: str) -> str:
    """Turn the instructions of one topic ('wifi' or 'password') into readable text."""
    entry = helpdesk()["instructions"][topic]   # запись для темы: title, steps, instruction_id
    # Первая строка: "Заголовок (GUIDE-WIFI):"
    lines = [f"{entry['title']} ({entry['instruction_id']}):"]
    # enumerate(..., start=1) даёт пары (номер, шаг), нумерация с 1: "1. ...", "2. ..."
    lines += [f"{i}. {step}" for i, step in enumerate(entry["steps"], start=1)]
    return "\n".join(lines)  # склеиваем строки через перевод строки
