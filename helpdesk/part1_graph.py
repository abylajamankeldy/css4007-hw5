"""Part 1 — a helpdesk graph WITHOUT a language model.

Original graph:

    START -> read_request -> route_request() --+-> wifi_help     -> END
                                               +-> password_help -> END
                                               +-> other_help    -> END

Changed graph (variant "with_clarification") adds one more branch:

                                               +-> clarify_problem -> END

Who decides what:
  * Nodes (read_request, wifi_help, ...) are ordinary Python functions. Each one
    receives the CURRENT graph state and returns a dict: an UPDATE to the state.
  * The conditional edge (route_request / route_request_with_clarification) is
    also a Python function. It returns the NAME of the next node. It does not
    change the state.
  * LangGraph runs the nodes in order, merges each update into the state, calls
    the routing function and moves to the node it names.

Run it:  python -m helpdesk.part1_graph "My Wi-Fi password does not work."
"""
import json                      # для печати результата в формате JSON
import sys                       # sys.argv: аргументы командной строки
from typing import TypedDict     # TypedDict: описание словаря с известными ключами

# StateGraph - конструктор графа; START и END - специальные имена входа и выхода.
from langgraph.graph import END, START, StateGraph

# format_instructions - текст инструкции; helpdesk - справочник helpdesk.json.
from helpdesk.data import format_instructions, helpdesk


# Схема СОСТОЯНИЯ (state) графа. LangGraph передаёт этот словарь от узла к узлу.
# total=False: не все ключи обязаны быть сразу. На входе есть только "message",
# остальные ключи появляются, когда узлы их возвращают.
class Part1State(TypedDict, total=False):
    message: str             # the user's original message (never changed by a node)
    category: str            # set by read_request: "wifi" | "password" | "other"
    topics_found: list[str]  # every topic keyword present, in rule order
    reply: str               # set by exactly one help node


# ---------------------------------------------------------------- nodes
# УЗЕЛ = обычная функция. Получает ВЕСЬ текущий state, возвращает словарь-ОБНОВЛЕНИЕ
# (только изменённые ключи). LangGraph сам вливает обновление в state.

def read_request(state: Part1State) -> dict:
    """Reads `message`. Writes `category` and `topics_found`.

    The rules are applied in order, so the FIRST matching rule wins:
      1. contains "wifi" or "wi-fi"  -> wifi
      2. otherwise contains "password" -> password
      3. otherwise                     -> other
    Lower-casing the message makes the rules case-insensitive.
    """
    # Нижний регистр: "PASSWORD" и "password" становятся одинаковыми (S02).
    text = state["message"].lower()
    # Слова берутся из справочника, а не пишутся в коде.
    aliases = helpdesk()["topic_aliases"]  # {"wifi": ["wifi", "wi-fi"], "password": ["password"]}

    # Для каждой темы: есть ли в тексте ХОТЯ БЫ ОДНО её слово (any).
    # "My Wi-Fi password..." -> ["wifi", "password"]
    topics_found = [topic for topic, words in aliases.items()
                    if any(word in text for word in words)]

    # Правила проверяются ПО ПОРЯДКУ: первое совпавшее побеждает.
    # Поэтому при двух темах выигрывает Wi-Fi (S04).
    if "wifi" in topics_found:          # rule 1
        category = "wifi"
    elif "password" in topics_found:    # rule 2
        category = "password"
    else:                               # rule 3
        category = "other"
    # Возвращаем ОБНОВЛЕНИЕ, а не весь state: "message" останется на месте.
    return {"category": category, "topics_found": topics_found}


def wifi_help(state: Part1State) -> dict:
    """Reads nothing but the reference file. Writes `reply`."""
    # state не используется: ответ берётся из справочника helpdesk.json.
    return {"reply": format_instructions("wifi")}


def password_help(state: Part1State) -> dict:
    """Reads nothing but the reference file. Writes `reply`."""
    return {"reply": format_instructions("password")}


def other_help(state: Part1State) -> dict:
    """Unsupported request: ask which of the two supported problems it is."""
    # Правило 3 из задания: спросить, Wi-Fi это или пароль.
    return {"reply": "I can only help with Wi-Fi or password problems. "
                     "Does your problem concern Wi-Fi or a password?"}


def clarify_problem(state: Part1State) -> dict:
    """Only in the changed graph: the message mentions BOTH topics."""
    # Новый узел для S05. Это ДРУГОЙ узел в графе, а не другая фраза в старом.
    return {"category": "needs_clarification",
            "reply": "Your message mentions both Wi-Fi and a password. "
                     "Is the problem connecting to Wi-Fi, or recovering your account "
                     "password? Which one should we solve first?"}


# ---------------------------------------------------------------- conditional edges
# УСЛОВНОЕ РЕБРО = функция, которая смотрит на state и возвращает ИМЯ следующего узла
# (строку). State она НЕ меняет.

def route_request(state: Part1State) -> str:
    """Original routing: the next node is chosen from `category` only."""
    # Словарь "категория -> узел". Если категории нет в словаре -> "other_help".
    return {"wifi": "wifi_help", "password": "password_help"}.get(state["category"], "other_help")


def route_request_with_clarification(state: Part1State) -> str:
    """Changed routing: if two topics were found, ask first; otherwise as before."""
    # Сначала новая проверка: найдено больше одной темы -> уточняем.
    if len(state.get("topics_found", [])) > 1:
        return "clarify_problem"
    # Иначе поведение как в исходном графе.
    return route_request(state)


# ---------------------------------------------------------------- graph

def build_part1_graph(variant: str = "original"):
    """variant = "original" (S01-S04) or "with_clarification" (S05)."""
    graph = StateGraph(Part1State)                 # пустой граф со схемой состояния
    # add_node("имя", функция): регистрируем узлы. По имени на узел ссылаются рёбра.
    graph.add_node("read_request", read_request)
    graph.add_node("wifi_help", wifi_help)
    graph.add_node("password_help", password_help)
    graph.add_node("other_help", other_help)

    targets = ["wifi_help", "password_help", "other_help"]  # куда может вести условное ребро
    router = route_request                                   # какая функция выбирает путь
    if variant == "with_clarification":
        # Изменённый граф: +1 узел, +1 возможная цель, другая функция-маршрутизатор.
        graph.add_node("clarify_problem", clarify_problem)
        targets.append("clarify_problem")
        router = route_request_with_clarification
    elif variant != "original":
        # Неизвестный вариант: лучше сразу упасть с понятной ошибкой.
        raise ValueError(f"unknown graph variant: {variant}")

    # Обычное ребро: всегда начинаем с read_request.
    graph.add_edge(START, "read_request")
    # Условное ребро: после read_request вызвать router(state) и перейти туда, что он вернёт.
    # targets нужен LangGraph, чтобы заранее знать все возможные пути.
    graph.add_conditional_edges("read_request", router, targets)
    for node in targets:
        graph.add_edge(node, END)                  # из каждого help-узла -> конец
    # compile(): checks the graph (every edge points to an existing node, START is
    # connected) and turns the description into a runnable object with .invoke()/.stream().
    return graph.compile()


def run_with_trace(message: str, variant: str = "original") -> dict:
    """Runs the graph and records the node order and the full state after each node."""
    app = build_part1_graph(variant)
    state: dict = {"message": message}   # моя копия state, чтобы записать его после каждого узла
    steps = []
    # stream_mode="updates" yields {node_name: update_returned_by_that_node} per step.
    for chunk in app.stream({"message": message}, stream_mode="updates"):
        for node, update in chunk.items():
            # Слияние словарей: ключи из update перезаписывают старые.
            state = {**state, **update}          # same merge LangGraph does (no reducers here)
            steps.append({"node": node, "update": update, "state_after": dict(state)})
    # route = START + узлы по порядку + END (это попадает в логи S01-S05)
    return {"variant": variant, "message": message,
            "route": ["START"] + [s["node"] for s in steps] + ["END"],
            "steps": steps, "final_state": state}


def mermaid(variant: str = "original") -> str:
    # LangGraph сам рисует граф в текстовом формате Mermaid (для проверки и SUBMISSION).
    return build_part1_graph(variant).get_graph().draw_mermaid()


# Блок срабатывает только при запуске файла как программы, не при импорте.
if __name__ == "__main__":
    args = sys.argv[1:]                    # всё, что написано после имени модуля
    variant = "original"
    if args and args[0] in ("--clarify", "--with-clarification"):
        variant, args = "with_clarification", args[1:]   # флаг включает изменённый граф
    msg = " ".join(args) or "I cannot connect to Wi-Fi."   # сообщение по умолчанию
    print(json.dumps(run_with_trace(msg, variant), indent=2, ensure_ascii=False))
