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

**Requirements:** Python 3.10+, an Anthropic API key.

```bash
git clone <your-repo-url>
cd tripmate
pip install -r requirements.txt

cp .env.example .env
# then edit .env and set ANTHROPIC_API_KEY=sk-ant-...
```

Run a single query:

```bash
python main.py --query "What should I pack for Tokyo in December?" --verbose
```

Or run interactively:

```bash
python main.py
```

Run the tests:

```bash
python -m pytest tests/ -v
```

No API key is needed to run the tests — the LLM's *decisions* are scripted
via fake Anthropic client responses (see `tests/fakes.py`), while the real
`rag_tool.py` / `weather_tool.py` execute for real in the integration test.
Only the live CLI (`main.py`) needs a real key.

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
LLM Provider (Anthropic Claude, tool-use API)
   │  returns either final text OR a tool_use request
   ▼
Orchestrator dispatches to the real tool:
   ├── RAG Tool          → tools/rag_tool.py      (TF-IDF over destination_data.json)
   └── Weather Tool       → tools/weather_tool.py  (mock seasonal lookup table)
   │  tool result(s) fed back to the LLM
   ▼
LLM synthesizes one final answer  →  returned to User
```

Every step (user query, each tool call + args + result, each LLM turn, and
any error) is emitted as a structured JSON log line (`agent/logging_setup.py`),
which doubles as the "visible reasoning trace" required by the assessment.

---

## Tool schemas given to the LLM

From `agent/tool_schemas.py` (abbreviated — see the file for full text):

**`search_destination_guide(query: str, city_filter?: str)`**
> Search TripMate's destination knowledge base for visa/entry requirements,
> best time to visit, local customs, packing advice, and safety notes.
> Covers: Tokyo, Paris, Bangkok, Reykjavik. Not for live weather forecasts.

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

The following are real outputs from the orchestrator (tool execution is
real; the LLM's tool-selection decision was scripted for these four
snapshots so they're reproducible without a live API key — see
`tests/test_tool_selection.py` and `tests/test_integration.py` for the
same flows verified as tests). Logs are trimmed to the relevant lines.

### 1. Single-tool — visa question

```
Query: "Do I need a visa to visit Tokyo as an Indian citizen?"
```
```json
{"event": "tool_call_started", "tool": "search_destination_guide", "arguments": {"query": "do I need a visa to visit Japan as an Indian citizen", "city_filter": "Tokyo"}}
{"event": "tool_call_succeeded", "tool": "search_destination_guide", "result": ["[Tokyo — visa] Indian passport holders need a visa to enter Japan for tourism; apply through a VFS-authorized center with proof of itinerary, hotel bookings, and bank statements. Processing typically takes 5-7 working days. ..."]}
{"event": "agent_final_response", "turns": 2}
```
> **TripMate:** Yes — as an Indian passport holder you need a visa for Japan.
> Apply through a VFS-authorized center with proof of itinerary, hotel
> bookings, and bank statements; processing takes about 5-7 working days.

### 2. Multi-tool — packing question (RAG + Weather together)

```
Query: "What should I pack for a trip to Reykjavik in January?"
```
```json
{"event": "tool_call_started", "tool": "search_destination_guide", "arguments": {"query": "packing tips", "city_filter": "Reykjavik"}}
{"event": "tool_call_succeeded", "tool": "search_destination_guide", "result": ["[Reykjavik — packing_tips] Pack windproof and waterproof outer layers regardless of season... In winter, bring thermal wear, waterproof boots with good grip for ice, and a heavy coat."]}
{"event": "tool_call_started", "tool": "get_weather_forecast", "arguments": {"city": "Reykjavik", "date_or_month": "January"}}
{"event": "tool_call_succeeded", "tool": "get_weather_forecast", "result": {"city": "Reykjavik", "month": "January", "conditions": "cold, windy, dark, Northern Lights visible", "temp_range_c": [-3, 3], "source": "mock_lookup_table"}}
```
> **TripMate:** Pack heavy thermal layers, a windproof/waterproof outer
> shell, and boots with good ice grip — Reykjavik in January runs about
> -3 to 3°C with wind and possible snow, and the destination guide
> recommends the same layered, weatherproof approach year-round.

### 3. Out-of-scope request

```
Query: "Can you book my flight to Paris for next month?"
```
No tool calls made — the model recognized this requires a real-world
action it cannot perform.

> **TripMate:** I can't book flights or make any real-world bookings — I
> can only help with trip-planning information like visas, weather,
> customs, packing, and safety. Want me to check visa requirements or
> weather for your trip instead?

### 4. Tool error handling — unsupported city

```
Query: "What is the weather like in Atlantis in June?"
```
```json
{"event": "tool_call_started", "tool": "get_weather_forecast", "arguments": {"city": "Atlantis", "date_or_month": "June"}}
{"event": "tool_call_error", "tool": "get_weather_forecast", "error": "No mock weather data available for 'Atlantis' in June. Supported cities: ['bangkok', 'paris', 'reykjavik', 'tokyo']."}
```
> **TripMate:** I don't have weather data for Atlantis — it's not one of
> the destinations in my current database (Tokyo, Paris, Bangkok,
> Reykjavik). Could you check the city name or ask about one of those?

The tool raised a typed error (`WeatherToolError`); the orchestrator caught
it, logged it, and fed it back to the LLM as an `is_error` tool result
rather than crashing — the LLM then explained the gap honestly instead of
inventing weather data.

---

## Design decisions & assumptions

- **No agent framework (LangChain/LangGraph/CrewAI).** A raw Anthropic
  tool-use loop (`agent/orchestrator.py`) is ~150 lines and keeps every
  step of the control flow (turn loop, dispatch, error handling, message
  threading) fully visible and unit-testable without framework internals
  getting in the way. For a 2-tool assistant this is the simpler and more
  debuggable choice; a larger tool count or need for graph-like branching
  would tip the balance toward LangGraph (see Future Improvements).
- **TF-IDF instead of a hosted embedding model for RAG.** The corpus is
  ~20 short chunks across 4 cities. TF-IDF + cosine similarity
  (`tools/rag_tool.py`) is deterministic, has no external API dependency
  at query time, and is trivial for a reviewer to run offline. Chunk text
  is indexed together with its city/country/section labels so a query
  that names a destination by name still matches even if the body text
  doesn't repeat that name.
- **Mock weather lookup table (Option B) instead of a live API.** Keeps
  the demo deterministic, network-independent, and focused on agentic
  reasoning rather than third-party API integration/rate limits. The
  `get_weather_forecast()` signature and output schema are what the agent
  depends on, so swapping in Open-Meteo later only means rewriting the
  function body.
- **Chunking strategy:** one chunk per (city, topic) pair — visa,
  best-time-to-visit, customs, packing tips, safety — rather than
  splitting by fixed token windows. Assessment source content is
  naturally structured this way, and topic-aligned chunks retrieve more
  cleanly than arbitrary windows for a corpus this small.
- **Reasoning trace as structured JSON logs** rather than a separate
  free-text "thinking" narration, so the trace is both human-readable
  (via `--verbose`) and machine-parseable for future observability
  tooling.
- **Scope awareness is prompt-based, not a separate classifier tool** —
  the system prompt explicitly instructs the model to decline
  out-of-scope real-world actions rather than fabricate a result. This is
  simpler than a dedicated "scope check" tool and, in testing, reliably
  caught the assessment's booking example.

## Known limitations

- Tool *selection* correctness depends on the live LLM's judgment call
  each time; it isn't hardcoded, so edge-case phrasing could occasionally
  pick zero/one/two tools differently than expected. Automated tests cover
  the orchestrator's handling of a given decision, not whether the live
  model always makes the "ideal" decision.
- The RAG tool only knows 4 cities; TF-IDF also has no semantic
  understanding of synonyms it hasn't seen in the corpus (e.g. "gear to
  bring" instead of "pack") — it can under-retrieve on very differently
  worded queries.
- The weather tool returns one static seasonal range per month, not a
  real day-level forecast.
- No conversation memory across CLI invocations in `--query` mode (the
  interactive REPL only keeps memory for the current process run, and
  each `agent.run()` call currently starts a fresh message list rather
  than carrying prior turns).
- No caching/rate-limiting layer yet (see Scalability below).

## Scalability considerations (discussion only)

**RAG tool at hundreds of cities:** TF-IDF's in-memory cosine similarity
is O(n) per query and fine into the low thousands of chunks, but at
"several hundred cities" the corpus should move to a proper vector store
(FAISS or Chroma) with a real sentence-embedding model, ANN indexing
(HNSW/IVF), and metadata filtering by city/country so retrieval stays
sub-second. Chunking would also need to move from hand-authored sections
to an automated pipeline (source docs → chunker → embedder → indexer) run
as a batch job whenever the knowledge base is updated, rather than
re-fitting one TF-IDF matrix on every process start.

**Avoiding redundant tool/LLM calls:** Cache RAG results and weather
lookups keyed on `(normalized_query_or_city, month)` with a short TTL for
weather (it changes) and a longer TTL for destination-guide content (it's
mostly static) — e.g. Redis in front of both tools. For near-duplicate
natural-language queries, a semantic cache (embed the query, check
cosine similarity against recent cached queries above a threshold) would
catch paraphrases that an exact-match cache misses.

**Reducing LLM API cost at volume:** Cache final answers for identical or
near-identical (query, resolved-tool-args) pairs; use a cheaper/smaller
model for simple single-tool or no-tool queries and reserve the larger
model for multi-tool synthesis; batch non-interactive workloads through
the Anthropic Batch API where latency isn't critical; and trim the system
prompt/tool schemas to only the tools relevant to a coarse query
classification if the tool count grows large enough that schema tokens
themselves become a meaningful cost.

**Keeping tool-selection latency low as tools grow:** Beyond a handful of
tools, sending every schema on every call adds both latency and cost.
A cheap pre-routing step (embedding-based intent classification, or a
small/fast model) can narrow the tool schemas sent to the main model down
to a relevant subset per query, rather than always sending the full tool
catalog.

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
  tool-selection behavior over model version upgrades.
- Consider LangGraph if the tool count grows enough to need explicit
  branching/parallel-tool-call graphs rather than the current linear
  turn loop.
