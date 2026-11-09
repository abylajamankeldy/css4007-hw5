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
import json
import sys
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from helpdesk.data import format_instructions, helpdesk


class Part1State(TypedDict, total=False):
    message: str             # the user's original message (never changed by a node)
    category: str            # set by read_request: "wifi" | "password" | "other"
    topics_found: list[str]  # every topic keyword present, in rule order
    reply: str               # set by exactly one help node


# ---------------------------------------------------------------- nodes

def read_request(state: Part1State) -> dict:
    """Reads `message`. Writes `category` and `topics_found`.

    The rules are applied in order, so the FIRST matching rule wins:
      1. contains "wifi" or "wi-fi"  -> wifi
      2. otherwise contains "password" -> password
      3. otherwise                     -> other
    Lower-casing the message makes the rules case-insensitive.
    """
    text = state["message"].lower()
    aliases = helpdesk()["topic_aliases"]  # {"wifi": ["wifi", "wi-fi"], "password": ["password"]}

    topics_found = [topic for topic, words in aliases.items()
                    if any(word in text for word in words)]

    if "wifi" in topics_found:          # rule 1
        category = "wifi"
    elif "password" in topics_found:    # rule 2
        category = "password"
    else:                               # rule 3
        category = "other"
    return {"category": category, "topics_found": topics_found}


def wifi_help(state: Part1State) -> dict:
    """Reads nothing but the reference file. Writes `reply`."""
    return {"reply": format_instructions("wifi")}


def password_help(state: Part1State) -> dict:
    """Reads nothing but the reference file. Writes `reply`."""
    return {"reply": format_instructions("password")}


def other_help(state: Part1State) -> dict:
    """Unsupported request: ask which of the two supported problems it is."""
    return {"reply": "I can only help with Wi-Fi or password problems. "
                     "Does your problem concern Wi-Fi or a password?"}


def clarify_problem(state: Part1State) -> dict:
    """Only in the changed graph: the message mentions BOTH topics."""
    return {"category": "needs_clarification",
            "reply": "Your message mentions both Wi-Fi and a password. "
                     "Is the problem connecting to Wi-Fi, or recovering your account "
                     "password? Which one should we solve first?"}


# ---------------------------------------------------------------- conditional edges

def route_request(state: Part1State) -> str:
    """Original routing: the next node is chosen from `category` only."""
    return {"wifi": "wifi_help", "password": "password_help"}.get(state["category"], "other_help")


def route_request_with_clarification(state: Part1State) -> str:
    """Changed routing: if two topics were found, ask first; otherwise as before."""
    if len(state.get("topics_found", [])) > 1:
        return "clarify_problem"
    return route_request(state)


# ---------------------------------------------------------------- graph

def build_part1_graph(variant: str = "original"):
    """variant = "original" (S01-S04) or "with_clarification" (S05)."""
    graph = StateGraph(Part1State)
    graph.add_node("read_request", read_request)
    graph.add_node("wifi_help", wifi_help)
    graph.add_node("password_help", password_help)
    graph.add_node("other_help", other_help)

    targets = ["wifi_help", "password_help", "other_help"]
    router = route_request
    if variant == "with_clarification":
        graph.add_node("clarify_problem", clarify_problem)
        targets.append("clarify_problem")
        router = route_request_with_clarification
    elif variant != "original":
        raise ValueError(f"unknown graph variant: {variant}")

    graph.add_edge(START, "read_request")
    graph.add_conditional_edges("read_request", router, targets)
    for node in targets:
        graph.add_edge(node, END)
    # compile(): checks the graph (every edge points to an existing node, START is
    # connected) and turns the description into a runnable object with .invoke()/.stream().
    return graph.compile()


def run_with_trace(message: str, variant: str = "original") -> dict:
    """Runs the graph and records the node order and the full state after each node."""
    app = build_part1_graph(variant)
    state: dict = {"message": message}
    steps = []
    # stream_mode="updates" yields {node_name: update_returned_by_that_node} per step.
    for chunk in app.stream({"message": message}, stream_mode="updates"):
        for node, update in chunk.items():
            state = {**state, **update}          # same merge LangGraph does (no reducers here)
            steps.append({"node": node, "update": update, "state_after": dict(state)})
    return {"variant": variant, "message": message,
            "route": ["START"] + [s["node"] for s in steps] + ["END"],
            "steps": steps, "final_state": state}


def mermaid(variant: str = "original") -> str:
    return build_part1_graph(variant).get_graph().draw_mermaid()


if __name__ == "__main__":
    args = sys.argv[1:]
    variant = "original"
    if args and args[0] in ("--clarify", "--with-clarification"):
        variant, args = "with_clarification", args[1:]
    msg = " ".join(args) or "I cannot connect to Wi-Fi."
    print(json.dumps(run_with_trace(msg, variant), indent=2, ensure_ascii=False))
