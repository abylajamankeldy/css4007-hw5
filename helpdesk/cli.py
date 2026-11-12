"""Terminal entry point.

  python -m helpdesk.cli part1 "My Wi-Fi password does not work."            # Part 1, no model
  python -m helpdesk.cli part1 --variant with_clarification "My Wi-Fi password does not work."
  python -m helpdesk.cli chat --user demo-user-a --thread t1 --profile normal  # interactive
  python -m helpdesk.cli say  --user demo-user-a --thread t1 --profile normal "message" ["message" ...]
  python -m helpdesk.cli show --user demo-user-a --thread t1                   # what is stored
  python -m helpdesk.cli reset                                                 # delete memory

Add --model fake to any chat/say command to use the labelled offline test double.
The user id, thread id and status profile are chosen HERE, by the person who
starts the program - never by the model.
"""
import argparse     # разбор аргументов командной строки (--user, --thread, ...)
import json
import os           # os.getpid(): номер текущего процесса
import sys

from helpdesk.data import service_status


def _utf8_console():
    # Windows consoles default to a code page that cannot print Kazakh letters.
    # Без этого print("Сәлем") упал бы с UnicodeEncodeError.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except AttributeError:
            pass        # у некоторых потоков нет reconfigure - тогда просто пропускаем


def print_turn(result: dict, show_trace: bool = True):
    # Печать одного хода: трасса узлов, ответ, путь тикета.
    if show_trace:
        for step in result["trace"]:
            line = f"  [{step['node']}]"                   # имя узла
            if "memory_read" in step:                      # что прочитано из памяти
                line += " memory: " + " | ".join(step["memory_read"])
            if "tool_requests" in step:                    # что модель попросила
                line += " requests " + ", ".join(f"{c['name']}({json.dumps(c['args'], ensure_ascii=False)})"
                                                 for c in step["tool_requests"])
            if "tool_results" in step:                     # что вернул инструмент (обрезано)
                line += " results " + json.dumps(step["tool_results"], ensure_ascii=False)[:300]
            print(line)
        if result["loop_limit_reached"]:
            print("  !! loop limit reached")
    print(f"Helpdesk: {result['reply']}")
    if result["ticket_file"]:
        print(f"  (local ticket draft: {result['ticket_file']})")


def main(argv=None):
    _utf8_console()
    profiles = list(service_status()["profiles"])          # допустимые снимки статуса
    p = argparse.ArgumentParser(prog="python -m helpdesk.cli")
    sub = p.add_subparsers(dest="command", required=True)  # подкоманды: part1, chat, say...

    # part1: граф без модели
    p1 = sub.add_parser("part1", help="Part 1 graph, no model")
    p1.add_argument("--variant", default="original", choices=["original", "with_clarification"])
    p1.add_argument("message")

    # chat и say: одинаковые флаги. user/thread обязательны - их выбирает ЧЕЛОВЕК.
    for name in ("chat", "say"):
        s = sub.add_parser(name)
        s.add_argument("--user", required=True)
        s.add_argument("--thread", required=True)
        s.add_argument("--profile", default="normal", choices=profiles)   # только реальные снимки
        s.add_argument("--model", default="real", choices=["real", "fake"])
        s.add_argument("--storage", default=None, help="memory folder (default local-storage/)")
        s.add_argument("--quiet", action="store_true", help="hide the node/tool trace")
        if name == "say":
            s.add_argument("messages", nargs="+")    # одно или несколько сообщений

    sh = sub.add_parser("show", help="print stored user memory and thread messages")
    sh.add_argument("--user")
    sh.add_argument("--thread")
    sh.add_argument("--storage", default=None)

    r = sub.add_parser("reset", help="delete local memory")
    r.add_argument("--storage", default=None)

    args = p.parse_args(argv)                        # разобрать то, что ввёл пользователь

    if args.command == "part1":
        from helpdesk.part1_graph import run_with_trace
        print(json.dumps(run_with_trace(args.message, args.variant), indent=2, ensure_ascii=False))
        return

    from helpdesk.storage import open_storage, reset_storage
    if args.command == "reset":
        print(f"deleted {reset_storage(args.storage)}")   # удалить папку с обеими базами
        return

    if args.command == "show":
        from helpdesk.agent import build_agent_graph
        from helpdesk.fake_model import FakeHelpdeskModel
        from helpdesk.tools import read_preferences
        with open_storage(args.storage) as (checkpointer, store):
            if args.user:
                print(f"user memory for {args.user}: {read_preferences(args.user, store) or 'nothing saved'}")
            if args.thread:
                # Чтобы прочитать state треда, нужен скомпилированный граф с checkpointer.
                app = build_agent_graph(FakeHelpdeskModel(), checkpointer, store)  # model unused
                state = app.get_state({"configurable": {"thread_id": args.thread}}).values
                for m in state.get("messages", []):
                    print(f"  {m.type}: {str(m.content)[:200] or m.tool_calls}")
                if not state:
                    print(f"thread {args.thread}: no saved conversation")
        return

    # chat / say: настоящий агент
    from helpdesk.agent import build_agent_graph, make_model, model_label, run_turn
    model = make_model(args.model)                   # Groq из .env или фальшивая
    # pid печатается, чтобы было видно: каждая команда say - отдельный процесс.
    print(f"[pid {os.getpid()}] model: {model_label(args.model)} | user={args.user} "
          f"thread={args.thread} profile={args.profile}")
    with open_storage(args.storage) as (checkpointer, store):
        app = build_agent_graph(model, checkpointer, store)
        # session: то, что выбрал человек; передаётся в run_turn -> config -> узлы
        session = dict(user_id=args.user, thread_id=args.thread, status_profile=args.profile)
        if args.command == "say":
            for message in args.messages:
                print(f"\nUser: {message}")
                print_turn(run_turn(app, message, **session), not args.quiet)
            return
        print("Type a message. Commands: /quit")
        while True:                                  # интерактивный чат
            try:
                message = input("\nUser: ").strip()
            except (EOFError, KeyboardInterrupt):    # Ctrl+Z / Ctrl+C -> выход
                break
            if message in ("/quit", "/exit"):
                break
            if message:
                print_turn(run_turn(app, message, **session), not args.quiet)


if __name__ == "__main__":
    main()
