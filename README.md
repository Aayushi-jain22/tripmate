# TripMate — Agentic AI Travel Assistant

TripMate is the agentic core of an AI travel assistant: given a natural-language
question about a destination, it dynamically decides which tool(s) it needs
(a destination-knowledge RAG lookup, a weather forecast lookup, both, or
neither), calls them, and synthesizes one coherent answer — with a full,
inspectable reasoning trace and no fixed keyword-routing script.

## Contents

- [Setup & run instructions](#setup--run-instructions)
- [Architecture](#architecture)
- [Tool schemas given to the LLM](#tool-schemas-given-to-the-llm)
- [Example runs](#example-runs)
- [Design decisions & assumptions](#design-decisions--assumptions)
- [Known limitations](#known-limitations)
- [Scalability considerations](#scalability-considerations-discussion-only)
- [Future improvements](#future-improvements)

---

## Setup & run instructions

**Requirements:** Python 3.10+, and an LLM provider — either an Anthropic
API key, or a local [Ollama](https://ollama.com) install (free, no key
needed). The agent is provider-agnostic; `TRIPMATE_PROVIDER` in `.env`
switches between them with no code changes.

```bash
git clone <your-repo-url>
cd tripmate
pip install -r requirements.txt
cp .env.example .env
```

**Option A — Ollama (free, local, no API key):**
```bash
ollama serve                    # in a separate terminal, keep it running
ollama pull llama3.1:8b
```
In `.env`, leave `TRIPMATE_PROVIDER=ollama` (the default).

**Option B — Anthropic (hosted, needs billing/credits):**
In `.env`, set `TRIPMATE_PROVIDER=anthropic` and `ANTHROPIC_API_KEY=sk-ant-...`.

Run a single query:

```bash
python main.py --query "What should I pack for Bangkok in July?" --verbose
```

Or run interactively:

```bash
python main.py
```

Run the tests (no API key or Ollama needed for this — see below):

```bash
python -m pytest tests/ -v
```

The LLM's tool-selection *decisions* are scripted via a fake client
(`tests/fakes.py`) in every test, while the real `rag_tool.py` /
`weather_tool.py` execute for real in the integration test. Only the live
CLI (`main.py`) needs a real provider (Ollama or Anthropic) configured.

---

## Architecture

See [`architecture.svg`](architecture.svg) for the diagram. Flow:

```
User query
   │
   ▼
Agent / Orchestrator (agent/orchestrator.py)
   │  sends query + tool schemas
   ▼
LLM Provider — local Ollama (agent/ollama_client.py adapter)
   │  returns either final text OR a tool_use request
   ▼
Orchestrator dispatches to the real tool:
   ├── RAG Tool          → tools/rag_tool.py      (TF-IDF over destination_data.json)
   └── Weather Tool       → tools/weather_tool.py  (mock seasonal lookup table)
   │  tool result(s) fed back to the LLM
   ▼
LLM synthesizes one final answer  →  returned to User
```

`agent/orchestrator.py` only depends on a provider-agnostic client
interface (`.messages.create(...)` returning `.content` blocks + a
`.stop_reason`). `main.py` picks which concrete client to build
(`anthropic.Anthropic()` or `agent.ollama_client.OllamaClient`) based on
`TRIPMATE_PROVIDER` — the orchestration, tool dispatch, and logging code
is identical either way.

Every step (user query, each tool call + args + result, each LLM turn, and
any error) is emitted as a structured JSON log line (`agent/logging_setup.py`),
which doubles as the "visible reasoning trace" required by the assessment.

---

## Tool schemas given to the LLM

From `agent/tool_schemas.py` (abbreviated — see the file for full text):

**`search_destination_guide(query: str, city_filter?: str)`**
> Search TripMate's destination knowledge base for visa/entry requirements,
> best time to visit, local customs, packing advice, and safety notes.
> Covers: Tokyo, Reykjavik, Bangkok, Barcelona. Not for live weather forecasts.

**`get_weather_forecast(city: str, date_or_month: str)`**
> Get expected weather conditions and temperature range for a city during a
> given month/date. Use for weather/temperature/season questions, or when
> packing advice needs to reflect actual seasonal conditions.

The system prompt additionally instructs the model to: decide tool need
per-query rather than always calling something; use **both** tools together
for packing questions; refuse out-of-scope actions (bookings etc.) instead
of fabricating a result; and surface "no info found" honestly rather than
guessing.

---

## Example runs

The destination content below is the real company-provided data pack
(Tokyo, Reykjavik, Bangkok, Barcelona) ingested as-is into the RAG tool.
Tool execution in these traces is real; running live against Ollama
(`llama3.1:8b`) reproduces the same shape of trace, with natural-language
phrasing varying slightly run to run since the LLM's exact wording isn't
deterministic.

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

### 2. Multi-tool — packing question (RAG + Weather together)

```
Query: "What should I pack for Bangkok in July?"
```
```json
{"event": "tool_call_started", "tool": "search_destination_guide", "arguments": {"query": "packing tips", "city_filter": "Bangkok"}}
{"event": "tool_call_succeeded", "tool": "search_destination_guide", "result": ["[Bangkok — packing_tips] Lightweight, breathable clothing is recommended given the heat and humidity. A modest cover-up layer ... A compact umbrella or rain jacket is handy during the rainy season."]}
{"event": "tool_call_started", "tool": "get_weather_forecast", "arguments": {"city": "Bangkok", "date_or_month": "July"}}
{"event": "tool_call_succeeded", "tool": "get_weather_forecast", "result": {"city": "Bangkok", "month": "July", "conditions": "hot, frequent afternoon monsoon showers", "temp_range_c": [25, 33], "source": "mock_lookup_table"}}
```
> **TripMate:** Pack lightweight, breathable clothing — Bangkok in July
> runs 25-33°C with frequent afternoon monsoon showers, so bring a
> compact umbrella or rain jacket, and a modest cover-up layer if you're
> visiting any temples.

### 3. Out-of-scope request

```
Query: "Can you book my flight to Barcelona?"
```
No tool calls made — the model recognized this requires a real-world
action it cannot perform.

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

The tool raised a typed error (`WeatherToolError`); the orchestrator caught
it, logged it, and fed it back to the LLM as an `is_error` tool result
rather than crashing — the LLM then explained the gap honestly instead of
inventing weather data.

---

## Design decisions & assumptions

- **Provider-agnostic client interface, defaulting to local Ollama.**
  `agent/orchestrator.py` depends only on a `.messages.create(...)`
  interface returning normalized content blocks + a stop reason. This is
  satisfied both by the real `anthropic.Anthropic()` SDK client and by
  `agent/ollama_client.py`, a thin adapter that converts to/from Ollama's
  local `/api/chat` tool-calling format. `main.py` picks the concrete
  client based on `TRIPMATE_PROVIDER`, so the orchestration/logging code
  never needs to know which provider is active. Ollama was used for the
  primary demo to avoid requiring paid API credits; Anthropic is a
  one-line `.env` change away, per the assessment's "any LLM provider"
  allowance.
- **No agent framework (LangChain/LangGraph/CrewAI).** A raw tool-use
  loop (`agent/orchestrator.py`) is ~150 lines and keeps every step of
  the control flow (turn loop, dispatch, error handling, message
  threading) fully visible and unit-testable without framework internals
  getting in the way.
- **TF-IDF instead of a hosted embedding model for RAG**, with a small
  built-in suffix-stripping stemmer (`tools/rag_tool.py`). The corpus is
  ~20 short chunks across 4 cities. Deterministic, zero external API
  dependency at query time, and trivial for a reviewer to run offline.
  Chunk text is indexed together with its city/country/section labels so
  a query naming a destination or country (e.g. "visa for Japan" when
  the chunk is filed under city "Tokyo") still matches. The stemmer
  handles common inflection mismatches between how a user phrases a
  question and how the source document is worded (e.g. "pack" vs.
  "packing", "custom" vs. "customs") — this was tuned against the real
  company-provided documents, not synthetic test data.
- **`city_filter` matches on city OR country** — a query like "visa for
  Japan" should resolve to the Tokyo chunk even though the model may
  reasonably pass either "Japan" or "Tokyo" as the filter argument.
- **Mock weather lookup table (Option B) instead of a live API.** Keeps
  the demo deterministic, network-independent, and focused on agentic
  reasoning rather than third-party API integration/rate limits.
- **Chunking strategy:** one chunk per (city, topic) pair — visa,
  best-time-to-visit, customs, packing tips, safety — matching the
  section structure of the provided source documents directly, rather
  than splitting by fixed token windows.
- **Reasoning trace as structured JSON logs** rather than free-text
  narration, so the trace is both human-readable (via `--verbose`) and
  machine-parseable for future observability tooling.
- **Scope awareness is prompt-based, not a separate classifier tool** —
  the system prompt explicitly instructs the model to decline
  out-of-scope real-world actions rather than fabricate a result.

## Known limitations

- Tool *selection* correctness depends on the live LLM's judgment call
  each time. This is more pronounced with the local `llama3.1:8b` model
  used for the free demo path than it would be with a larger hosted
  model (Claude/GPT-4-class) — smaller models occasionally skip a tool
  call it should have made, or pass slightly malformed arguments.
  Automated tests cover the orchestrator's handling of a *given*
  decision correctly; they don't guarantee the live model always makes
  the ideal decision.
- The RAG tool only knows 4 cities; TF-IDF (even with basic stemming)
  has no deep semantic understanding of synonyms far outside the corpus
  vocabulary — it can under-retrieve on very differently worded queries.
- The weather tool returns one static seasonal range per month, not a
  real day-level forecast.
- No multi-turn conversation memory across separate `--query` invocations.
- No caching/rate-limiting layer yet (see Scalability below).

## Scalability considerations (discussion only)

**RAG tool at hundreds of cities:** TF-IDF's in-memory cosine similarity
is O(n) per query and fine into the low thousands of chunks, but at
"several hundred cities" the corpus should move to a proper vector store
(FAISS or Chroma) with a real sentence-embedding model, ANN indexing
(HNSW/IVF), and metadata filtering by city/country so retrieval stays
sub-second. Chunking would also move from hand-authored sections to an
automated pipeline (source docs → chunker → embedder → indexer) run as a
batch job whenever the knowledge base updates.

**Avoiding redundant tool/LLM calls:** Cache RAG results and weather
lookups keyed on `(normalized_query_or_city, month)` with a short TTL for
weather (it changes) and a longer TTL for destination-guide content
(mostly static) — e.g. Redis in front of both tools. A semantic cache
(embed the query, check similarity against recent cached queries) would
also catch near-duplicate phrasings an exact-match cache misses.

**Reducing LLM cost/compute at volume:** Cache final answers for
identical or near-identical (query, resolved-tool-args) pairs; use a
smaller/cheaper model for simple single-tool or no-tool queries and
reserve a larger model for multi-tool synthesis; batch non-interactive
workloads where latency isn't critical; trim the system prompt/tool
schemas to only relevant tools if the tool count grows large.

**Keeping tool-selection latency low as tools grow:** Beyond a handful of
tools, sending every schema on every call adds latency and cost. A cheap
pre-routing step (embedding-based intent classification, or a small/fast
model) can narrow the tool schemas sent to the main model to a relevant
subset per query, rather than always sending the full tool catalog.

## Future improvements

- Swap TF-IDF for sentence embeddings + FAISS/Chroma once the knowledge
  base grows past a handful of cities.
- Replace the mock weather table with a real Open-Meteo integration
  behind the same function signature, with retry/backoff and a fallback
  to the mock table on API failure.
- Add multi-turn conversation memory so follow-up questions
  ("what about in July instead?") resolve without repeating the city.
- Add a lightweight tool-selection eval harness that runs a fixed set of
  labeled queries against the live model periodically, to catch drift in
  tool-selection behavior across model/provider swaps (e.g. Ollama model
  upgrades, or switching from Ollama to Anthropic).
- Consider LangGraph if the tool count grows enough to need explicit
  branching/parallel-tool-call graphs rather than the current linear
  turn loop.
