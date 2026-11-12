"""Run S01-S14 from data/scenarios.json and write one log per scenario to outputs/.

  python -m helpdesk.run_scenarios                 # all, real model from .env
  python -m helpdesk.run_scenarios S06 S07 S08     # some
  python -m helpdesk.run_scenarios --model fake    # offline dry run -> outputs/fake-model/

How the data/README.md rules are followed:
  * reset_before: true  -> the memory folder is deleted before the first event.
  * restart_process     -> the events are split into segments at every restart.
                           EACH SEGMENT RUNS IN A NEW PYTHON PROCESS (subprocess)
                           that opens the same SQLite files. The log records the
                           process id (pid) of every segment, so you can see it changed.
  * S04 uses the original Part 1 graph, S05 the changed one.
"""
import argparse
import json
import os
import subprocess      # запуск НОВОГО процесса Python (настоящий перезапуск)
import sys             # sys.executable: путь к текущему python.exe
import time
from pathlib import Path

from helpdesk.data import REPO_ROOT, scenarios

# Сценарии хранят память отдельно от ручных чатов.
SCENARIO_STORAGE = "local-storage/scenarios"


def _segments(events):
    # Режем список событий на куски по "restart_process".
    # S11: [msg, restart, msg] -> [[msg], [msg]]; S10: [msg, msg] -> [[msg, msg]]
    segments, current = [], []
    for event in events:
        if event["action"] == "restart_process":
            segments.append(current)   # закончить текущий кусок
            current = []               # начать новый
        else:
            current.append(event)
    segments.append(current)           # последний кусок
    return segments


# ------------------------------------------------------------------ child process

def run_child(scenario_id: str, segment: int, model_kind: str, ticket_dir: str):
    """Runs ONE segment in this (new) process and prints the results as JSON."""
    from helpdesk.agent import build_agent_graph, make_model, run_turn
    from helpdesk.storage import open_storage
    scenario = next(s for s in scenarios() if s["id"] == scenario_id)   # найти сценарий по id
    events = _segments(scenario["events"])[segment]                     # свой кусок событий
    out = []
    # Открываем ТЕ ЖЕ файлы, что и предыдущий процесс: память берётся с диска.
    with open_storage(SCENARIO_STORAGE) as (checkpointer, store):
        app = build_agent_graph(make_model(model_kind), checkpointer, store)
        for event in events:
            result = run_turn(app, event["message"], user_id=event["user_id"],
                              thread_id=event["thread_id"],
                              status_profile=scenario["status_profile"], ticket_dir=ticket_dir)
            # os.getpid() = номер ЭТОГО процесса; разные pid в логе = разные процессы.
            result.update(user_id=event["user_id"], thread_id=event["thread_id"], pid=os.getpid())
            out.append(result)
    # Одна строка JSON в stdout - её прочитает родительский процесс.
    # ensure_ascii=True: казахские буквы как \uXXXX, чтобы консоль Windows их не испортила.
    print(json.dumps(out, ensure_ascii=True))


# ------------------------------------------------------------------ parent process

def run_part1(scenario: dict) -> list[str]:
    # Часть 1 не нужна модель: выполняем прямо в этом процессе.
    from helpdesk.part1_graph import run_with_trace
    variant = scenario.get("graph_variant", "original")   # S04 - original, S05 - with_clarification
    lines = [f"- graph variant: `{variant}` (Part 1, no model)", ""]
    for event in scenario["events"]:
        r = run_with_trace(event["message"], variant)
        lines += [f"**User** ({event['user_id']}, thread {event['thread_id']}): {event['message']}", "",
                  f"Route: `{' -> '.join(r['route'])}`", "", "| step | node | update | state after node |",
                  "| --- | --- | --- | --- |"]
        for i, step in enumerate(r["steps"], 1):
            lines.append(f"| {i} | {step['node']} | `{_short(step['update'])}` | "
                         f"`{_short(step['state_after'])}` |")
        lines += ["", "**Final reply:**", "", "```", r["final_state"]["reply"], "```", ""]
    return lines


def _short(obj, limit=400) -> str:
    # JSON в одну строку: "|" экранируем (иначе сломается таблица Markdown), обрезаем длинное.
    text = json.dumps(obj, ensure_ascii=False).replace("|", "\\|").replace("\n", " ")
    return text if len(text) <= limit else text[:limit] + "…"


def run_agent_scenario(scenario: dict, model_kind: str, ticket_dir: str) -> list[str]:
    from helpdesk.storage import reset_storage
    lines = []
    if scenario.get("reset_before"):
        reset_storage(SCENARIO_STORAGE)              # стереть память перед сценарием
        lines.append(f"- memory reset before first event: deleted `{SCENARIO_STORAGE}/`")
    for index, segment in enumerate(_segments(scenario["events"])):
        if index > 0:
            lines += ["", "---", "### restart_process: previous process exited; "
                          "a NEW process opens the same storage", ""]
        # Команда для ребёнка: python -m helpdesk.run_scenarios --child S11 1 --model real ...
        cmd = [sys.executable, "-m", "helpdesk.run_scenarios", "--child", scenario["id"],
               str(index), "--model", model_kind, "--ticket-dir", ticket_dir]
        # subprocess.run запускает новый процесс и ЖДЁТ его завершения.
        proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True,
                              encoding="utf-8", env={**os.environ, "PYTHONIOENCODING": "utf-8"})
        if proc.returncode != 0:                     # ребёнок упал -> записать ошибку в лог
            lines += ["```", "CHILD PROCESS FAILED", proc.stderr[-3000:], "```"]
            return lines
        # Последняя строка вывода ребёнка = JSON с результатами его ходов.
        for result in json.loads(proc.stdout.strip().splitlines()[-1]):
            lines += _turn_lines(result)
    return lines


def _turn_lines(r: dict) -> list[str]:
    # Формат одного хода в outputs/S06-S14.md
    lines = ["", f"**User** (`{r['user_id']}`, thread `{r['thread_id']}`, process pid {r['pid']}): "
                 f"{r['message']}", "", "Trace:", ""]
    for i, step in enumerate(r["trace"], 1):
        parts = [f"{i}. `{step['node']}`"]
        if "memory_read" in step:
            parts.append("read " + "; ".join(step["memory_read"]))
        if "tool_requests" in step:
            parts.append("model requests " + ", ".join(
                f"`{c['name']}({_short(c['args'], 300)})`" for c in step["tool_requests"]))
        if "tool_results" in step:
            parts.append("results " + ", ".join(f"`{k}` → `{_short(v, 350)}`"
                                                for k, v in step["tool_results"].items()))
        if "state_update" in step:
            parts.append(f"state `{_short(step['state_update'], 200)}`")
        if "model_text" in step and "tool_requests" not in step:
            parts.append("model answers (no tool requested) → END")
        lines.append(" — ".join(parts))
    lines += ["", "Sources used in this turn:", ""] + [f"- {s}" for s in r["sources"]]
    lines += ["", "**Final reply:**", "", "```", r["reply"], "```", "",
              f"- ticket draft: `{r['ticket_file']}`" if r["ticket_file"] else "- ticket draft: none",
              f"- tool rounds: {r['tool_rounds']}; loop limit reached: {r['loop_limit_reached']}",
              f"- messages now saved in this thread: {r['messages_in_thread']}"]
    return lines


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")     # чтобы печать работала в консоли Windows
        except AttributeError:
            pass
    p = argparse.ArgumentParser()
    p.add_argument("ids", nargs="*")                 # S06 S07 ... (пусто = все)
    p.add_argument("--model", default="real", choices=["real", "fake"])
    p.add_argument("--out", default=None)
    p.add_argument("--child", nargs=2, metavar=("SCENARIO", "SEGMENT"))   # скрытый режим ребёнка
    p.add_argument("--ticket-dir", default=None)
    args = p.parse_args(argv)

    if args.child:
        # Нас запустил родитель: выполнить один кусок и выйти.
        run_child(args.child[0], int(args.child[1]), args.model, args.ticket_dir)
        return

    from helpdesk.agent import model_label
    # Настоящая модель -> outputs/, фальшивая -> outputs/fake-model/ (не попадает в git).
    out = Path(args.out or ("outputs/fake-model" if args.model == "fake" else "outputs"))
    out = out if out.is_absolute() else REPO_ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    ticket_dir = str(out / "tickets")
    chosen = [s for s in scenarios() if not args.ids or s["id"] in args.ids]
    for scenario in chosen:
        started = time.strftime("%Y-%m-%d %H:%M:%S")
        command = f"python -m helpdesk.run_scenarios {scenario['id']}" + (
            " --model fake" if args.model == "fake" else "")
        # Шапка лога: id, часть, снимок, команда, время (+ модель для частей 2-3)
        header = [f"# {scenario['id']} — {scenario['title']}", "",
                  f"- part: {scenario['part']}", f"- status snapshot: `{scenario['status_profile']}`",
                  f"- command: `{command}`", f"- run at: {started}"]
        if scenario["part"] == 1:
            body = run_part1(scenario)
        else:
            header.append(f"- model: {model_label(args.model)}")
            body = run_agent_scenario(scenario, args.model, ticket_dir)
        path = out / f"{scenario['id']}.md"
        path.write_text("\n".join(header + body) + "\n", encoding="utf-8")
        print(f"{scenario['id']}: wrote {path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
