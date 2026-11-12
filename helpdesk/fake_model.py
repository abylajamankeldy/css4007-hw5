"""FAKE MODEL — a test double, NOT a language model.

Used by the tests (GitHub Actions has no API key) and by `--model fake` for an
offline dry run. It imitates what a tool-calling model sends back:
an AIMessage that either contains `tool_calls` or a final text answer.
Its "decisions" are fixed keyword rules written below, so every log produced
with it says "FAKE model". The real demonstration uses a real model.

It still goes through the SAME graph: the same agent node, the same router,
the same tools node, checkpointer and store. That is what the tests check.
"""
import json
import re
import uuid

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

# Слова, по которым фальшивка понимает: пользователь описывает уже сделанное действие.
STEP_WORDS = ("enabled", "forgot", "reconnect", "tried", "restarted")


def _call(name: str, **args) -> AIMessage:
    # Создаёт "просьбу инструмента": пустой текст + один tool_call.
    # Ровно такой объект возвращает настоящая модель, когда просит инструмент.
    return AIMessage(content="", tool_calls=[{"name": name, "args": args,
                                              "id": "fake_" + uuid.uuid4().hex[:8]}])


def _topic(text: str):
    # Тема по ключевым словам (как read_request в Части 1).
    text = text.lower()
    if "wifi" in text or "wi-fi" in text:
        return "wifi"
    if "password" in text:
        return "password"
    return None


def _steps(texts: list[str]) -> list[str]:
    """Sentences in which the user describes something they already did."""
    steps = []
    for text in texts:
        # Режем на предложения по . ! ? (с пробелом после).
        for sentence in re.split(r"(?<=[.!?])\s+", text):
            # Оставляем предложения со словом-шагом и без вопроса.
            if any(w in sentence.lower() for w in STEP_WORDS) and "?" not in sentence:
                steps.append(sentence.strip())
    return steps


class FakeHelpdeskModel:
    """Labelled fake. `always_call_tools=True` makes it request a tool forever
    (used to prove the loop limit stops it)."""

    label = "FAKE model (keyword rules, not an LLM)"

    def __init__(self, always_call_tools: bool = False):
        self.always_call_tools = always_call_tools   # режим для теста предела петли

    def bind_tools(self, tools):
        # Граф вызывает bind_tools у любой модели. Фальшивке схемы не нужны.
        return self

    def invoke(self, messages) -> AIMessage:
        # На входе тот же список, что у настоящей модели: SystemMessage + история.
        system = messages[0].content if isinstance(messages[0], SystemMessage) else ""
        humans = [m.content for m in messages if isinstance(m, HumanMessage)]   # все сообщения пользователя
        # Индекс последнего сообщения пользователя = начало текущего хода.
        last_human_index = max(i for i, m in enumerate(messages) if isinstance(m, HumanMessage))
        this_turn = messages[last_human_index + 1:]   # всё, что было после него в этом ходе
        # Результаты инструментов в ЭТОМ ходе: {"check_service_status": {...}, ...}
        results = {m.name: json.loads(m.content) for m in this_turn if isinstance(m, ToolMessage)}
        text = humans[-1]                             # текущее сообщение
        low = text.lower()
        earlier = humans[:-1]                         # прошлые сообщения (из checkpointer)

        if self.always_call_tools:
            return _call("find_instructions", topic="wifi")   # всегда просит -> проверка предела

        # Память пользователя фальшивка "видит" так же, как настоящая модель: в system prompt.
        lang = re.search(r"preferred reply language is (\w+)", system)
        prefix = f"[{lang.group(1)} reply requested] " if lang else ""

        # 1. "Please remember that I prefer replies in Kazakh"
        m = re.search(r"prefer (?:replies|answers) in (\w+)", text, re.I)
        if m and "remember_preference" not in results:
            return _call("remember_preference", language=m.group(1))
        if "remember_preference" in results:
            prefix = f"[{results['remember_preference']['preferences']['reply_language']} reply requested] "

        # 2. Questions about what is remembered
        if "language" in low and "?" in text:
            answer = (f"You asked me to reply in {lang.group(1)}." if lang
                      else "No preferred language has been recorded for you.")
            if "previous problem" in low:
                answer += (" Earlier in this conversation you wrote: " + " | ".join(earlier)
                           if earlier else " No previous problem has been recorded for you.")
            if "help with" in low:
                answer += " I can help with Wi-Fi, password recovery, service status and local ticket drafts."
            return AIMessage(content=prefix + answer)
        # "Which steps did I try?" -> пересказ прошлых сообщений из истории треда
        if re.search(r"which (steps|problem)|did we discuss|already try", low):
            if not earlier:
                return AIMessage(content=prefix + "Nothing has been recorded in this conversation yet.")
            return AIMessage(content=prefix + "From this conversation: you wrote: "
                             + " | ".join(earlier) + " Steps you tried: " + "; ".join(_steps(earlier)))

        # Тема: из текущего сообщения или из последнего прошлого, где она была.
        topic = _topic(text) or next((_topic(t) for t in reversed(earlier) if _topic(t)), None)
        wants_status = any(w in low for w in ("down", "status", "known problem", "outage"))
        # Тикет просили сейчас ИЛИ в прошлом сообщении (S09: детали пришли следующим сообщением).
        ticket_pending = "ticket" in low or (earlier and "ticket" in earlier[-1].lower()
                                             and "?" not in earlier[-1])
        conditional = "if it is operating" in low or "if operating" in low   # условие из S06-S08

        # 3. Status check
        if wants_status and topic and "check_service_status" not in results:
            return _call("check_service_status",
                         service="wifi" if topic == "wifi" else "account_recovery")
        status = results.get("check_service_status", {})

        # 4. Ticket
        if ticket_pending:
            if conditional and status and status["status"] != "operational":
                if status["status"] == "outage":       # авария -> тикет не делаем (S07)
                    return AIMessage(content=prefix + f"Status source: {status['message']} "
                                     "You asked for a ticket only if the service is operating, "
                                     "so I did not prepare one. The data gives no repair time.")
                # unknown -> не можем подтвердить, спрашиваем (S08)
                return AIMessage(content=prefix + f"Status source: {status['message']} I cannot "
                                 "confirm the service is operating. Would you like a ticket draft "
                                 "explaining that the status could not be checked?")
            steps = _steps(humans)
            if not topic or not steps:                 # нет деталей -> спросить (S09, ход 1)
                return AIMessage(content=prefix + "What problem are you having, "
                                 "and what have you already tried?")
            if "prepare_ticket" not in results:
                return _call("prepare_ticket", category=topic,
                             summary=f"{topic} problem: " + humans[-1][:120], steps_tried=steps)
            ticket = results["prepare_ticket"]
            if ticket.get("created"):
                return AIMessage(content=prefix + f"{status.get('message', 'Status not checked.')} "
                                 f"I prepared a local draft {ticket['file']}. Nothing has been sent.")
            return AIMessage(content=prefix + "I could not prepare the draft: "
                             + ", ".join(ticket.get("missing", [])))

        if status:                                     # статус проверили, тикет не просили (S14)
            return AIMessage(content=prefix + f"Status source: {status['message']}")

        # 5. Instructions
        if topic and "find_instructions" not in results:
            return _call("find_instructions", topic=topic)
        if "find_instructions" in results:
            r = results["find_instructions"]
            return AIMessage(content=prefix + f"{r['title']} ({r['instruction_id']}): "
                             + " ".join(f"{i}. {s}" for i, s in enumerate(r["steps"], 1)))

        # Ничего не подошло -> общий ответ
        return AIMessage(content=prefix + "I can help with Wi-Fi, password recovery, service "
                         "status and local ticket drafts. What is the problem?")
