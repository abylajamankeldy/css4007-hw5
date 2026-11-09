# HW5 — University IT helpdesk with LangGraph

CSS-4007 Artificial Intelligence. The assignment text is in [ASSIGNMENT.md](ASSIGNMENT.md),
my answers and results are in [SUBMISSION.md](SUBMISSION.md), run logs are in [outputs/](outputs/).

All data is fictional (`data/`). Nothing connects to real university systems; tickets are
local JSON drafts.

## Tested environment

- Python **3.13** (CI also runs on Windows and Ubuntu with Python 3.13)
- Exact package versions: [requirements.txt](requirements.txt) (direct dependencies, pinned),
  [requirements-lock.txt](requirements-lock.txt) (full `pip freeze`)
- LangGraph 1.2.20, langgraph-checkpoint-sqlite 3.1.3, langchain-openai 1.7.4

## 1. Install

```bash
git clone <your fork URL>
cd css4007-hw5
python -m venv .venv
# Windows PowerShell:  .venv\Scripts\Activate.ps1
# macOS / Linux:       source .venv/bin/activate
python -m pip install -r requirements.txt
```

## 2. Configure a model (only for Parts 2–3)

```bash
cp .env.example .env        # Windows: copy .env.example .env
```

Fill in `.env`. Any OpenAI-compatible chat model that supports tool calling works:

| Provider | `OPENAI_API_KEY` | `MODEL_NAME` | `OPENAI_BASE_URL` |
| --- | --- | --- | --- |
| OpenAI | `sk-...` | a tool-capable model, e.g. `gpt-5.6-luna` | *(empty)* |
| Groq (free key) | `gsk_...` | `openai/gpt-oss-120b` | `https://api.groq.com/openai/v1` |

`.env` is in `.gitignore`. Part 1 and the tests need no key.

## 3. Run

```bash
# Part 1 - graph without a model
python -m helpdesk.cli part1 "My Wi-Fi password does not work."
python -m helpdesk.cli part1 --variant with_clarification "My Wi-Fi password does not work."

# Part 2 - tools called directly, without the agent
python -m helpdesk.tools

# Parts 2-3 - interactive chat (user, thread and status snapshot are chosen here)
python -m helpdesk.cli chat --user demo-user-a --thread t1 --profile normal

# One-shot messages: every command is a separate process (useful to show a restart)
python -m helpdesk.cli say --user demo-user-a --thread t1 --profile normal "I cannot connect to Wi-Fi."
python -m helpdesk.cli say --user demo-user-a --thread t1 --profile normal "Which steps did I already try?"

# What is stored for a user / a thread
python -m helpdesk.cli show --user demo-user-a --thread t1
```

Status snapshots (`--profile`): `normal`, `wifi_outage`, `recovery_outage`, `status_unavailable`.
Add `--model fake` to `chat`/`say` to use the offline test double instead of a real model.

## 4. Run the supplied scenarios S01–S14

```bash
python -m helpdesk.run_scenarios            # all fourteen, real model from .env
python -m helpdesk.run_scenarios S06 S07 S08
python -m helpdesk.run_scenarios --model fake   # offline dry run -> outputs/fake-model/ (ignored by git)
```

The runner writes `outputs/S01.md` … `outputs/S14.md` and ticket drafts to
`outputs/tickets/`. It follows `data/README.md`: memory is deleted before a scenario
with `reset_before: true`, and at every `restart_process` event the next messages run in a
**new Python process** (the log shows the different process ids) against the same files.

## 5. Where memory is saved and how to clear it

| File | Kind of memory | Key |
| --- | --- | --- |
| `local-storage/conversations.sqlite` | conversation (LangGraph `SqliteSaver` checkpointer) | `thread_id` |
| `local-storage/user_memory.sqlite` | user preferences (LangGraph `SqliteStore`) | namespace `("users", user_id)` |
| `local-storage/scenarios/` | the same two files, used only by the scenario runner | |

The folder can be changed with `HELPDESK_STORAGE_DIR` or `--storage`. It is in `.gitignore`.

```bash
python -m helpdesk.cli reset                                # deletes local-storage/
python -m helpdesk.cli reset --storage local-storage/scenarios
```

## 6. Tests

```bash
python -m pytest -v
```

| File | Checks | Would catch |
| --- | --- | --- |
| `tests/test_routing.py` | Part 1 routes for all four messages, both graph variants, state after each node | wrong rule order, case sensitivity, clarify branch missing |
| `tests/test_tools.py` | each tool called directly | wrong snapshot, `unknown` treated as operational, ticket missing fields, ticket with invented/missing steps or a password |
| `tests/test_agent_loop.py` | agent ↔ tools loop with the **fake model** | loop not returning to the model, hard-coded snapshot, endless tool loop (limit) |
| `tests/test_memory.py` | SQLite memory, restart via a new process, user and thread separation | in-memory checkpointer, preference saved in the thread, memory shared between users |

The agent tests use `helpdesk/fake_model.py`, a clearly labelled keyword-rule fake, so they run
without a key. GitHub Actions ([.github/workflows/tests.yml](.github/workflows/tests.yml)) runs them
on every push. The real-model demonstration is separate (`python -m helpdesk.run_scenarios`).

## Code map

| File | What it contains |
| --- | --- |
| `helpdesk/data.py` | loads the JSON fixtures |
| `helpdesk/part1_graph.py` | Part 1 graph, routing rules, trace |
| `helpdesk/tools.py` | the tools as plain functions (+ `remember_preference`) |
| `helpdesk/agent.py` | Parts 2–3 graph: `load_memory → agent ⇄ tools`, loop limit, model setup |
| `helpdesk/fake_model.py` | fake model for tests (not an LLM) |
| `helpdesk/storage.py` | opens the two SQLite files |
| `helpdesk/cli.py` | terminal program |
| `helpdesk/run_scenarios.py` | S01–S14 runner and log writer |
