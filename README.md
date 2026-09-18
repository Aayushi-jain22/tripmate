# TripMate — Agentic AI Travel Assistant (LangGraph edition)

TripMate is the agentic core of an AI travel assistant: given a natural-language
question about a destination, it dynamically decides which tool(s) it needs
(a destination-knowledge RAG lookup, a weather forecast lookup, both, or
neither), calls them, and synthesizes one coherent answer — with a full,
inspectable reasoning trace.

This is the **framework-based** version of the assistant, built on
**LangGraph**. The initial submission used a hand-rolled tool-use loop
(~150 lines in a single `orchestrator.py`); this version keeps the same
tools and behavior but restructures the control flow as an explicit,
declarative **agent graph**, with tool selection delegated entirely to
the LLM via standard LangChain tool-calling — see
[Design decisions](#design-decisions--assumptions) for why LangGraph, and
[What changed](#what-changed-from-the-custom-implementation) for a
side-by-side of old vs. new.

## Contents

- [Setup & run instructions](#setup--run-instructions)
- [Architecture](#architecture)
- [What changed from the custom implementation](#what-changed-from-the-custom-implementation)
- [Tools given to the LLM](#tools-given-to-the-llm)
- [Example runs](#example-runs)
- [Design decisions & assumptions](#design-decisions--assumptions)
- [Known limitations](#known-limitations)
- [Scalability considerations](#scalability-considerations-discussion-only)
- [Future improvements](#future-improvements)

---

## Setup & run instructions

**Requirements:** Python 3.10+, and an LLM provider — either an Anthropic API key, or a local Ollama installation (free, no key needed). The agent is provider-agnostic; `TRIPMATE_PROVIDER` in `.env` switches between them with no code changes.

```bash
git clone <your-repo-url>
cd tripmate
pip install -r requirements.txt
cp .env.example .env
```

### Ollama (free, local, no API key)

**1. Install Ollama** — https://ollama.com/, then verify:

```bash
ollama --version
```

**2. Start the Ollama server** (keep this running in its own terminal):

```bash
ollama serve
```

**3. Pull the model** (one-time):

```bash
ollama pull llama3.1:8b
```

**4. Verify**, and optionally chat with it directly:

```bash
ollama list
ollama run llama3.1:8b
```

**5. Configure the provider** — in `.env`:

```text
TRIPMATE_PROVIDER=ollama
```

### Anthropic (hosted, needs an API key)

```text
TRIPMATE_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-...
```

### Run TripMate

```bash
python main.py --query "What should I pack for Bangkok in July?" --verbose
# or interactively:
python main.py
```

### Run the tests

```bash
python -m pytest tests/ -v
```

The LLM's tool-selection *decisions* are scripted via `FakeChatModel`
(`tests/fakes.py`) in every test, while the real `rag_tool.py` /
`weather_tool.py` execute for real underneath LangGraph's `ToolNode` in
every test that scripts a tool call — including the integration test.
Only the live CLI (`main.py`) needs a real provider (Ollama or
Anthropic) configured.

---

## Architecture

See [`architecture.svg`](architecture.svg) for the diagram. Flow:

```
User query
   │
   ▼
LangGraph StateGraph (agent/graph.py)      state = {"messages": [...]}
   │
   ├─▶ "agent" node ──▶ llm.bind_tools(TOOLS).invoke(system_prompt + messages)
   │        │                  (agent/llm_factory.py picks Ollama or Anthropic)
   │        │
   │        ▼
   │   tools_condition (LangGraph prebuilt)
   │        │
   │        ├─ AIMessage has tool_calls ──▶ "tools" node
   │        │        │
   │        │        ▼
   │        │   ToolNode(TOOLS)  — runs every requested tool call
   │        │     ├── search_destination_guide → tools/rag_tool.py     (TF-IDF over destination_data.json)
   │        │     └── get_weather_forecast      → tools/weather_tool.py (mock seasonal lookup table)
   │        │        │
   │        │        ▼
   │        │   ToolMessage(s) appended to state ──▶ back to "agent" node
   │        │
   │        └─ AIMessage has no tool_calls ──▶ END
   ▼
Final answer  →  returned to User
```

`agent/graph.py` only depends on the standard LangChain `BaseChatModel`
interface (`.bind_tools(...)` → `.invoke(messages)`); `agent/llm_factory.py`
is the only file that picks a concrete provider (`ChatOllama` or
`ChatAnthropic`) based on `TRIPMATE_PROVIDER` — the graph, tool dispatch,
and logging code are identical either way.

Every step (user query, each tool call + args + result, each LLM turn,
and any error) is still emitted as a structured JSON log line
(`agent/logging_setup.py`), which doubles as the "visible reasoning
trace" required by the assessment — this is unchanged by the framework
swap, since the log events now live in the tool wrappers
(`tools/langchain_tools.py`) instead of the old orchestrator.


## Tools given to the LLM

From `tools/langchain_tools.py` (schemas below are auto-generated from
these docstrings — this is what the LLM actually sees):

**`search_destination_guide(query: str, city_filter: str | None = None)`**
> Search TripMate's destination knowledge base for visa/entry requirements,
> best time to visit, local customs, packing advice, and safety notes.
> Covers: Tokyo, Reykjavik, Bangkok, Barcelona. Not for live weather forecasts.

**`get_weather_forecast(city: str, date_or_month: str)`**
> Get expected weather conditions and temperature range for a city during a
> given month/date. Use for weather/temperature/season questions, or when
> packing advice needs to reflect actual seasonal conditions.

The system prompt (`agent/tool_schemas.py`) additionally instructs the
model to: decide tool need per-query rather than always calling
something; use **both** tools together for packing questions; refuse
out-of-scope actions (bookings etc.) instead of fabricating a result;
and surface "no info found" honestly rather than guessing. None of this
routing is hardcoded — the LLM reads the tool docstrings + system prompt
and chooses, per query, whether to call zero, one, or both tools.

---

## Example runs

Same four scenarios as the original submission, now flowing through the
graph. Tool execution is real; the LLM's phrasing will vary slightly run
to run.

### 1. Single-tool — visa question

```
Query: "Do I need a visa to visit Japan?"
```
```json
{"event": "tool_call_started", "tool": "search_destination_guide", "arguments": {"query": "do I need a visa to visit Japan", "city_filter": "Tokyo"}}
{"event": "tool_call_succeeded", "tool": "search_destination_guide", "result": ["[Tokyo — visa] Many nationalities -- including most EU countries, the UK, the US, Canada, and Australia -- can enter Japan visa-free for short tourist stays (typically up to 90 days), though exact allowances vary by passport. ..."]}
{"event": "agent_final_response", "turns": 2}
```
> **TripMate:** Most nationalities (EU, UK, US, Canada, Australia) can
> enter Japan visa-free for short tourist stays up to 90 days, but exact
> allowances vary by passport — make sure your passport has enough
> remaining validity and you can show a return/onward ticket.

### 2. Multi-tool — packing question (both tools requested in one turn)

```
Query: "What should I pack for Bangkok in July?"
```
```json
{"event": "tool_call_started", "tool": "search_destination_guide", "arguments": {"query": "packing tips", "city_filter": "Bangkok"}}
{"event": "tool_call_succeeded", "tool": "search_destination_guide", "result": ["[Bangkok — packing_tips] Lightweight, breathable clothing is recommended given the heat and humidity. ..."]}
{"event": "tool_call_started", "tool": "get_weather_forecast", "arguments": {"city": "Bangkok", "date_or_month": "July"}}
{"event": "tool_call_succeeded", "tool": "get_weather_forecast", "result": {"city": "Bangkok", "month": "July", "conditions": "hot, frequent afternoon monsoon showers", "temp_range_c": [25, 33], "source": "mock_lookup_table"}}
```
> **TripMate:** Pack lightweight, breathable clothing — Bangkok in July
> runs 25-33°C with frequent afternoon monsoon showers, so bring a
> compact umbrella or rain jacket, and a modest cover-up layer if you're
> visiting any temples.

With the graph, the LLM can request both tool calls on the *same*
`AIMessage`; `ToolNode` runs both before routing back to the `agent`
node, so this is one round trip through "tools", not two.

### 3. Out-of-scope request

```
Query: "Can you book my flight to Barcelona?"
```
No tool calls made — `tools_condition` routes straight to `END`.

> **TripMate:** I can't book flights or make any real-world bookings — I
> can only help with trip-planning information like visas, weather,
> customs, packing, and safety. Want me to check visa requirements or
> weather for Barcelona instead?

### 4. Tool error handling — unsupported city

```
Query: "What is the weather like in Rome in June?"
```
```json
{"event": "tool_call_started", "tool": "get_weather_forecast", "arguments": {"city": "Rome", "date_or_month": "June"}}
{"event": "tool_call_error", "tool": "get_weather_forecast", "error": "No mock weather data available for 'Rome' in June. Supported cities: ['bangkok', 'barcelona', 'reykjavik', 'tokyo']."}
```
> **TripMate:** I don't have weather data for Rome — it's not one of the
> destinations in my current database (Tokyo, Reykjavik, Bangkok,
> Barcelona). Could you check the city name or ask about one of those?

The tool wrapper catches the typed `WeatherToolError` and returns an
`"ERROR: ..."` string instead of raising; `ToolNode` feeds that back to
the LLM as the tool's result, so the graph never crashes — the LLM
explains the gap honestly instead of inventing weather data.

---

## Design decisions & assumptions

- **LangGraph over LangChain's higher-level `AgentExecutor` or CrewAI/AutoGen.**
  The assessment specifically calls out "structuring the agent,
  skills/tools, state, and workflow" — LangGraph's `StateGraph` makes
  each of those an explicit, named artifact (`state.py`, the node
  functions in `graph.py`, the conditional edge) rather than hiding them
  inside a single executor abstraction. It also composes cleanly with a
  provider-agnostic `BaseChatModel`, keeping the Ollama-first setup from
  the original submission working with a one-line `.env` change to
  Anthropic.
- **Dynamic tool selection is entirely LLM-driven.** `agent/graph.py`
  contains no `if`/keyword routing; the LLM decides per query, via
  standard tool-calling, whether to answer directly or call one/both
  tools, and `tools_condition` (a LangGraph prebuilt) just checks whether
  the latest message carries `tool_calls`.
- **Retrieval/weather logic (`tools/rag_tool.py`, `tools/weather_tool.py`)
  is unchanged from the original submission.** Only a thin `@tool`
  adapter layer (`tools/langchain_tools.py`) was added on top, so the
  reasoning/decisions documented in the first submission about TF-IDF vs.
  a hosted embedding model, and a mock weather table vs. a live API,
  still apply unchanged.
- **Structured JSON logging as the reasoning trace** moved from the
  orchestrator into the tool wrappers themselves, since that's now where
  tool execution actually happens; the trace format and CLI
  (`--verbose`) output are unaffected.
- **`TripMateAgent` (agent/orchestrator.py) is kept as a thin
  compatibility layer** over the compiled graph, rather than having
  `main.py` call the graph directly, so the CLI, the `AgentResponse`/
  `ToolCallRecord` trace shape, and all existing call sites didn't need
  to change shape — only what's *underneath* `TripMateAgent.run()`
  changed.

## Known limitations

- Tool *selection* correctness still depends on the live LLM's judgment
  call each time — this is a property of any tool-calling agent
  (custom or framework-based) and is more pronounced with the local
  `llama3.1:8b` model than a larger hosted model.
- Two pre-existing test failures carried over from the original
  submission are **not** fixed by this refactor (out of scope — this
  pass only restructures the workflow, per the assessment's brief):
  `test_get_weather_month_number` asks for Paris, which was never in the
  mock weather table (only Tokyo/Barcelona/Bangkok/Reykjavik), and
  `test_search_returns_relevant_chunk_for_packing_query`'s TF-IDF match
  for the Reykjavik packing chunk doesn't clear the relevance threshold
  with the current stemmer. Both are data/test mismatches in
  `tools/`, not in the agentic workflow.
- The RAG tool only knows 4 cities; TF-IDF has no deep semantic
  understanding of synonyms far outside the corpus vocabulary.
- The weather tool returns one static seasonal range per month, not a
  real day-level forecast.
- No multi-turn conversation memory across separate `--query`
  invocations (LangGraph supports this via a checkpointer/thread id if
  needed later — see Future improvements).

## Scalability considerations (discussion only)

**RAG tool at hundreds of cities:** move the corpus to a proper vector
store (FAISS or Chroma) with a real sentence-embedding model and ANN
indexing, with metadata filtering by city/country. This swap only
touches `tools/rag_tool.py`'s internals — the `@tool` wrapper and the
graph don't need to change.

**Avoiding redundant tool/LLM calls:** Cache RAG results and weather
lookups keyed on `(normalized_query_or_city, month)`, e.g. Redis in front
of both tools; a semantic cache (embed the query, check similarity
against recent cached queries) would also catch near-duplicate
phrasings an exact-match cache misses.

**Reducing LLM cost/compute at volume:** cache final answers for
identical/near-identical `(query, resolved-tool-args)` pairs; use a
smaller/cheaper model for simple single-tool or no-tool queries; batch
non-interactive workloads where latency isn't critical.

**Keeping tool-selection latency low as tools grow:** beyond a handful
of tools, sending every schema on every call adds latency and cost. With
LangGraph this becomes a routing concern that can live in its own node —
e.g. a cheap pre-routing node that narrows `TOOLS` to a relevant subset
before binding them to the main model — without touching the rest of the
graph.

## Future improvements

- Swap TF-IDF for sentence embeddings + FAISS/Chroma once the knowledge
  base grows past a handful of cities.
- Replace the mock weather table with a real Open-Meteo integration
  behind the same function signature, with retry/backoff and a fallback
  to the mock table on API failure.
- Add multi-turn conversation memory using LangGraph's checkpointing
  (e.g. `MemorySaver` keyed on a conversation/thread id), so follow-up
  questions ("what about in July instead?") resolve without repeating
  the city — this is now a natural extension of the existing graph
  rather than a structural change.
- Add a lightweight tool-selection eval harness that runs a fixed set of
  labeled queries against the live model periodically, to catch drift in
  tool-selection behavior across model/provider swaps.
- If the tool count grows meaningfully, add an explicit pre-routing node
  to the graph that narrows the bound toolset per query (see
  Scalability above).
