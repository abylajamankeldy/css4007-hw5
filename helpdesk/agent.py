"""Parts 2 and 3 — the model chooses tools; the graph runs them and keeps memory.

    START -> load_memory -> agent --route_after_agent()--+-> tools ------> agent  (loop)
                                                         +-> loop_limit -> END
                                                         +-> END  (no tool requested)

  * load_memory : reads USER memory (store, by user_id) and reports how many
                  earlier messages the CHECKPOINT gave us for this thread_id.
  * agent       : calls the model once. The model either answers with text or
                  REQUESTS tools (AIMessage.tool_calls). It never runs them.
  * tools       : OUR code executes every requested tool and appends the
                  results as ToolMessages; increments tool_rounds.
  * loop_limit  : reached when the model still wants tools after MAX_TOOL_ROUNDS.
                  Stops the loop with an honest message.

The routing function route_after_agent() is where the loop limit is enforced.
LangGraph's recursion_limit (set in run_turn) is a second safety net.
"""
import json                       # результаты инструментов передаются модели как JSON-текст
import os                         # os.getenv: ключ и имя модели из .env
from typing import Optional       # Optional[str] = строка или None

# Типы сообщений LangChain:
#   AIMessage     - ответ модели (текст или tool_calls)
#   SystemMessage - системная инструкция (идёт первой)
#   ToolMessage   - результат инструмента
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig   # тип config: внутри user_id, thread_id...
# MessagesState - готовая схема state с полем messages и редьюсером add_messages.
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.store.base import BaseStore           # общий тип store (SQLite или в памяти)
from pydantic import BaseModel, Field                # описание схем инструментов для модели

from helpdesk import tools as T                      # мои функции-инструменты под именем T
from helpdesk.data import helpdesk

MAX_TOOL_ROUNDS = 4        # model -> tools -> model ... at most 4 tool rounds per user message
RECURSION_LIMIT = 25       # LangGraph's own hard stop on the number of steps


class AgentState(MessagesState):
    """`messages` (from MessagesState) uses the add_messages reducer: updates are
    APPENDED, not overwritten. Everything here is saved by the checkpointer per thread."""
    # messages: list - уже есть в MessagesState; новые сообщения ДОПИСЫВАЮТСЯ в конец.
    tool_rounds: int                    # tool rounds in the CURRENT user turn (reset each turn)
    loop_limit_reached: bool            # True, если сработал предел петли
    user_language: Optional[str]        # copied from user memory by load_memory
    service_status_seen: dict           # {"wifi": "operational"} - results of check_service_status
    last_ticket_file: Optional[str]     # путь к последнему созданному тикету
    sources: list                       # what this turn read: user memory / conversation / reference file


# ------------------------------------------------------------------ tool schemas shown to the model
# Only the arguments the MODEL may choose. user_id, thread_id, status_profile and
# the store are added by tools_node from the session.
# Имя класса = имя инструмента, докстринг = описание для модели,
# поля = аргументы, Field(description=...) = подсказка модели, что передавать.

class find_instructions(BaseModel):
    """Look up the official troubleshooting steps for 'wifi' or 'password' in the local helpdesk guide."""
    topic: str = Field(description="'wifi' or 'password'")


class check_service_status(BaseModel):
    """Check whether a service is operating, has a known outage, or its status is unknown."""
    service: str = Field(description="'wifi' or 'account_recovery' (password recovery)")
    # НЕТ поля status_profile: снимок модель выбрать не может.


class prepare_ticket(BaseModel):
    """Save a LOCAL support-ticket draft (nothing is sent). Use only details the user actually gave."""
    category: str = Field(description="'wifi' or 'password'")
    summary: str = Field(description="Short description of the problem, in the user's terms")
    steps_tried: list[str] = Field(description="Steps the user said they already tried; do not add any")
    # НЕТ полей user_id, thread_id, service_status: их подставляет программа.


class remember_preference(BaseModel):
    """Save the user's preferred reply language so it is available in later conversations."""
    language: str = Field(description="e.g. 'Kazakh', 'Russian', 'English'")
    # НЕТ поля user_id: модель не может записать язык другому пользователю.


# Этот список прикрепляется к модели через bind_tools: LangChain превращает каждый
# класс в JSON-описание {"type": "function", "function": {"name": ..., "parameters": ...}}.
TOOL_SCHEMAS = [find_instructions, check_service_status, prepare_ticket, remember_preference]


def system_prompt(user_language: Optional[str]) -> str:
    # Системная инструкция строится заново при КАЖДОМ вызове модели и не хранится в истории.
    data = helpdesk()
    # Сюда попадает ПАМЯТЬ ПОЛЬЗОВАТЕЛЯ: язык из store или "ничего не сохранено".
    memory_line = (f"USER MEMORY: this user's saved preferred reply language is {user_language}. "
                   f"Write your replies in {user_language}."
                   if user_language else
                   "USER MEMORY: nothing is saved for this user (no preferred language, "
                   "no earlier problems). Reply in the language the user writes in.")
    # Правила - это текст для модели, а не закон. Важное дополнительно проверяет код
    # (пустые шаги и пароли в prepare_ticket, статус в тикете - в tools_node).
    return f"""You are a university IT helpdesk assistant in a practice exercise with fictional data.
Scope: {", ".join(data["scope"])}.

{memory_line}

Rules:
1. Get instructions only from the find_instructions tool and service status only from
   check_service_status. Never invent steps, outages, repair times or real university facts.
2. Status "unknown" means the status could not be determined: it is NOT "operating" and
   NOT a confirmed outage. Say that you cannot confirm it.
3. Respect the user's conditions. If they say "if it is operating, prepare a ticket", call
   prepare_ticket only when check_service_status returned "operational". Otherwise explain
   why you did not prepare it and, for "unknown", ask whether they still want a draft.
4. A ticket needs the problem and the steps the user already tried, in the user's own words
   from this conversation. If either is missing, ASK; never add steps the user did not say.
   A ticket is only a local draft: never say it was sent. Give its file name.
5. Never ask for or record passwords, recovery codes or API keys.
6. If the user asks you to remember a reply language, call remember_preference.
7. Use the earlier messages of this conversation to understand references such as "that"
   or "the steps I tried". If something was not said in this conversation and is not in
   user memory, say that it has not been recorded.
"""


# ------------------------------------------------------------------ graph

def build_agent_graph(model, checkpointer=None, store=None):
    """model: a chat model with .bind_tools(), or the FakeHelpdeskModel used in tests."""
    # "Модель с инструментами": к каждому запросу будет приложен список TOOL_SCHEMAS.
    model_with_tools = model.bind_tools(TOOL_SCHEMAS)
    # Узлы ниже объявлены ВНУТРИ функции, чтобы видеть model_with_tools (замыкание).

    # ---- УЗЕЛ 1: load_memory ------------------------------------------------
    # LangGraph смотрит на имена параметров: config и store передаёт сам.
    def load_memory(state: AgentState, config: RunnableConfig, *, store: BaseStore = None) -> dict:
        cfg = config["configurable"]                     # user_id, thread_id, status_profile из run_turn
        prefs = T.read_preferences(cfg["user_id"], store)   # ЧТЕНИЕ памяти пользователя (store)
        language = prefs.get("reply_language")           # "Kazakh" или None
        # К этому моменту LangGraph УЖЕ загрузил историю треда из checkpointer
        # и дописал новое сообщение. Минус 1 = без нового сообщения.
        earlier = len(state["messages"]) - 1          # the last one is the new user message
        # Две строки для лога: что прочитано из каждой памяти.
        sources = [
            f"user memory (store namespace ('users', '{cfg['user_id']}')): "
            + (f"reply_language={language}" if language else "nothing saved"),
            f"conversation memory (checkpoint thread_id='{cfg['thread_id']}'): "
            f"{earlier} earlier message(s) loaded",
        ]
        return {"user_language": language, "sources": sources}

    # ---- УЗЕЛ 2: agent - ЕДИНСТВЕННОЕ место, где вызывается модель ----------
    def agent(state: AgentState) -> dict:
        # Запрос к модели = системная инструкция + ВСЯ история треда.
        prompt = [SystemMessage(system_prompt(state.get("user_language")))] + state["messages"]
        # HTTP-запрос к Groq. Ответ - AIMessage: либо текст, либо tool_calls (просьба).
        response = model_with_tools.invoke(prompt)
        # Благодаря редьюсеру add_messages ответ ДОПИШЕТСЯ к истории.
        return {"messages": [response]}

    # ---- УЗЕЛ 3: tools_node - здесь МОЙ КОД выполняет инструменты -----------
    def tools_node(state: AgentState, config: RunnableConfig, *, store: BaseStore = None) -> dict:
        cfg = config["configurable"]
        # Копии полей state, которые будем дополнять (dict()/list() = копия, не ссылка).
        seen = dict(state.get("service_status_seen") or {})
        sources = list(state.get("sources") or [])
        ticket_file = state.get("last_ticket_file")
        language = state.get("user_language")
        results = []                                     # сюда сложим ToolMessage-ответы
        # Последнее сообщение - ответ модели. tool_calls = список просьб:
        # [{"name": "check_service_status", "args": {"service": "wifi"}, "id": "call_..."}]
        for call in state["messages"][-1].tool_calls:
            name, args = call["name"], call["args"]
            try:
                if name == "find_instructions":
                    # args.get(...) берёт только известный аргумент; лишнее игнорируется.
                    out = T.find_instructions(args.get("topic", ""))
                    if out["found"]:
                        sources.append(f"reference file data/helpdesk.json: {out['instruction_id']}")
                elif name == "check_service_status":
                    # Снимок статуса берётся из сессии (cfg), а НЕ от модели.
                    out = T.check_service_status(args.get("service", ""), cfg["status_profile"])
                    seen[out["service"]] = out["status"]   # запоминаем: {"wifi": "operational"}
                    sources.append(f"reference file data/service_status.json "
                                   f"[{cfg['status_profile']}]: {out['service']}={out['status']}")
                elif name == "prepare_ticket":
                    # Тема -> сервис, чтобы найти результат проверки статуса этого сервиса.
                    topic = T._normalise_topic(args.get("category", ""))
                    service = helpdesk()["topic_services"].get(topic or "", "")
                    out = T.prepare_ticket(
                        # 3 аргумента ОТ МОДЕЛИ:
                        args.get("category", ""), args.get("summary", ""),
                        args.get("steps_tried", []),
                        # остальное ПОДСТАВЛЯЕТ ПРОГРАММА:
                        user_id=cfg["user_id"], thread_id=cfg["thread_id"],
                        status_profile=cfg["status_profile"],
                        # статус - результат настоящей проверки или "not_checked"
                        service_status=seen.get(service, "not_checked"),
                        ticket_dir=cfg.get("ticket_dir"))
                    if out.get("created"):
                        ticket_file = out["file"]
                elif name == "remember_preference":
                    # ЗАПИСЬ в память пользователя; user_id - из сессии.
                    out = T.remember_preference(args.get("language", ""),
                                                user_id=cfg["user_id"], store=store)
                    if out.get("saved"):
                        # Сразу обновляем язык в state: следующий вызов модели в этом же
                        # ходе уже получит "отвечай по-казахски" (поэтому S11 сразу на казахском).
                        language = out["preferences"]["reply_language"]
                        sources.append(f"user memory WRITE ('users', '{cfg['user_id']}'): "
                                       f"reply_language={language}")
                else:
                    out = {"error": f"unknown tool {name}"}   # модель придумала несуществующий инструмент
            except Exception as exc:                     # a broken tool must not crash the chat
                out = {"error": f"{type(exc).__name__}: {exc}"}
            # Результат -> ToolMessage. tool_call_id связывает его с конкретной просьбой.
            results.append(ToolMessage(content=json.dumps(out, ensure_ascii=False),
                                       name=name, tool_call_id=call["id"]))
        # Обновление state: ToolMessage допишутся к истории, счётчик раундов +1.
        return {"messages": results, "tool_rounds": state.get("tool_rounds", 0) + 1,
                "service_status_seen": seen, "last_ticket_file": ticket_file,
                "user_language": language, "sources": sources}

    # ---- УЗЕЛ 4: loop_limit - сюда попадаем только после 4 раундов ---------
    def loop_limit(state: AgentState) -> dict:
        # Answer the pending tool calls (APIs require a result for every call),
        # then end the turn with an explicit message.
        # (Иначе в истории треда осталась бы просьба без ответа, и следующий запрос упал бы.)
        pending = [ToolMessage(content=json.dumps({"error": "not executed: loop limit reached"}),
                               name=c["name"], tool_call_id=c["id"])
                   for c in state["messages"][-1].tool_calls]
        final = AIMessage(content=f"I stopped after {MAX_TOOL_ROUNDS} tool rounds without "
                                  "reaching an answer. Please rephrase or give more details.")
        return {"messages": pending + [final], "loop_limit_reached": True}

    # ---- УСЛОВНОЕ РЕБРО после agent ----------------------------------------
    # Возвращает ИМЯ следующего узла; state не меняет.
    def route_after_agent(state: AgentState) -> str:
        last = state["messages"][-1]                     # ответ модели
        # getattr(..., None): если атрибута нет, вернётся None, ошибки не будет.
        if not getattr(last, "tool_calls", None):
            return END                                   # plain answer -> finish the turn
        if state.get("tool_rounds", 0) >= MAX_TOOL_ROUNDS:
            return "loop_limit"                          # LOOP LIMIT enforced here
        return "tools"                                   # модель просит инструмент -> выполнить

    # ---- Сборка графа -------------------------------------------------------
    graph = StateGraph(AgentState)
    graph.add_node("load_memory", load_memory)
    graph.add_node("agent", agent)
    graph.add_node("tools", tools_node)
    graph.add_node("loop_limit", loop_limit)
    graph.add_edge(START, "load_memory")                 # сначала всегда читаем память
    graph.add_edge("load_memory", "agent")               # потом зовём модель
    # После agent: в tools, в loop_limit или в END - решает route_after_agent.
    graph.add_conditional_edges("agent", route_after_agent, ["tools", "loop_limit", END])
    graph.add_edge("tools", "agent")                     # ПЕТЛЯ: результат обратно модели
    graph.add_edge("loop_limit", END)
    # checkpointer: LangGraph сам грузит/сохраняет state треда по thread_id.
    # store: передаётся в узлы параметром store.
    return graph.compile(checkpointer=checkpointer, store=store)


def run_turn(app, message: str, *, user_id: str, thread_id: str, status_profile: str,
             ticket_dir: str | None = None) -> dict:
    """Send ONE user message and collect a trace of everything that happened."""
    # config["configurable"] видят все узлы. thread_id - ключ истории в checkpointer.
    config = {"configurable": {"user_id": user_id, "thread_id": thread_id,
                               "status_profile": status_profile, "ticket_dir": ticket_dir},
              "recursion_limit": RECURSION_LIMIT}         # вторая страховка от бесконечной петли
    # Вход хода: новое сообщение (допишется к истории) + обнуление счётчиков на этот ход.
    turn_input = {"messages": [("user", message)], "tool_rounds": 0,
                  "loop_limit_reached": False, "sources": []}
    trace = []
    # stream(..., "updates"): после каждого узла -> {имя_узла: его обновление}.
    for chunk in app.stream(turn_input, config, stream_mode="updates"):
        for node, update in chunk.items():
            trace.append(describe_update(node, update or {}))
    state = app.get_state(config).values                 # итоговый state треда из checkpointer
    last = state["messages"][-1]                         # последнее сообщение = ответ модели
    # Тикеты, созданные именно в ЭТОМ ходе (а не раньше в треде).
    tickets = [t["tool_results"]["prepare_ticket"]["file"] for t in trace
               if t.get("tool_results", {}).get("prepare_ticket", {}).get("created")]
    return {"message": message, "trace": trace, "reply": _text(last),
            "sources": state.get("sources", []),
            "ticket_file": tickets[-1] if tickets else None,   # only a ticket made in THIS turn
            "loop_limit_reached": state.get("loop_limit_reached", False),
            "tool_rounds": state.get("tool_rounds", 0),
            "messages_in_thread": len(state["messages"])}


def _text(message) -> str:
    # Достаёт текст из ответа. Некоторые провайдеры возвращают content списком блоков.
    content = message.content
    if isinstance(content, list):   # some providers return content blocks
        content = "".join(part.get("text", "") if isinstance(part, dict) else str(part)
                          for part in content)
    return content


def describe_update(node: str, update: dict) -> dict:
    """Small, readable summary of one node's state update (for logs)."""
    entry = {"node": node}
    msgs = update.get("messages", [])
    # Просьбы модели (tool_calls) -> "tool_requests" в логе
    calls = [{"name": c["name"], "args": c["args"]}
             for m in msgs for c in (getattr(m, "tool_calls", None) or [])]
    if calls:
        entry["tool_requests"] = calls
    # Результаты инструментов (ToolMessage) -> "tool_results" в логе
    results = {m.name: json.loads(m.content) for m in msgs if isinstance(m, ToolMessage)}
    if results:
        entry["tool_results"] = results
    # Текст модели, если она ответила словами
    texts = [_text(m) for m in msgs if isinstance(m, AIMessage) and _text(m)]
    if texts:
        entry["model_text"] = texts[-1]
    # Изменения ключевых полей state
    for key in ("user_language", "tool_rounds", "service_status_seen",
                "last_ticket_file", "loop_limit_reached"):
        if key in update:
            entry.setdefault("state_update", {})[key] = update[key]
    if node == "load_memory":
        entry["memory_read"] = update.get("sources", [])   # что прочитано из памяти
    return entry


# ------------------------------------------------------------------ model

def make_model(kind: str = "real"):
    """'real' -> OpenAI-compatible chat model from .env; 'fake' -> labelled test double."""
    if kind == "fake":
        from helpdesk.fake_model import FakeHelpdeskModel
        return FakeHelpdeskModel()                       # для тестов, без ключа
    # Импорты здесь, а не наверху: тестам langchain_openai не нужен.
    from dotenv import load_dotenv
    from langchain_openai import ChatOpenAI
    load_dotenv()                                        # .env -> переменные окружения
    key, name = os.getenv("OPENAI_API_KEY"), os.getenv("MODEL_NAME")
    if not key or not name:
        raise SystemExit("Set OPENAI_API_KEY and MODEL_NAME in .env (see README), "
                         "or use --model fake for the offline test double.")
    # base_url = адрес Groq -> класс для OpenAI работает с Groq (формат API одинаковый).
    # temperature=0: минимум случайности. max_retries=3: повтор при сетевой ошибке/лимите.
    return ChatOpenAI(model=name, api_key=key, base_url=os.getenv("OPENAI_BASE_URL") or None,
                      temperature=0, max_retries=3)


def model_label(kind: str) -> str:
    # Подпись модели для шапки логов: "openai/gpt-oss-120b via https://api.groq.com/openai/v1"
    if kind == "fake":
        return "FAKE model (helpdesk/fake_model.py, keyword rules, NOT an LLM)"
    from dotenv import load_dotenv
    load_dotenv()
    base = os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1"
    return f"{os.getenv('MODEL_NAME')} via {base}"
