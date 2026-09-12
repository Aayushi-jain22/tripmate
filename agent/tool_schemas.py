"""
Tool schemas (as given to the LLM) for Anthropic's tool-use / function
calling API. These are the exact descriptions the model sees when
deciding which tool(s), if any, to call for a given query.
"""

TOOL_SCHEMAS = [
    {
        "name": "search_destination_guide",
        "description": (
            "Search TripMate's destination knowledge base for information about "
            "visa/entry requirements, best time to visit, local customs and etiquette, "
            "general packing advice, and safety notes for a destination. "
            "Currently covers: Tokyo, Paris, Bangkok, and Reykjavik. "
            "Use this for any question about visas, customs, general safety, or "
            "what a destination is generally like -- but NOT for the current/seasonal "
            "weather forecast itself (use get_weather_forecast for that)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The natural-language question or topic to search for, e.g. 'do I need a visa for Japan' or 'what to pack for Iceland'.",
                },
                "city_filter": {
                    "type": "string",
                    "description": "Optional. Restrict the search to a single known city (e.g. 'Tokyo') if the user's query names one.",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_weather_forecast",
        "description": (
            "Get the expected weather conditions and temperature range for a city "
            "during a given month or date. Use this whenever the user asks about "
            "weather, temperature, or season, or when packing advice needs to reflect "
            "actual seasonal conditions rather than general tips."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "City name, e.g. 'Tokyo'.",
                },
                "date_or_month": {
                    "type": "string",
                    "description": "A month name, month number, or date, e.g. 'December', '12', or '2026-12-05'.",
                },
            },
            "required": ["city", "date_or_month"],
        },
    },
]

SYSTEM_PROMPT = """You are TripMate, an AI travel assistant that helps users plan trips.

You have access to two tools:
- search_destination_guide: destination knowledge (visas, customs, packing tips, safety notes)
- get_weather_forecast: seasonal weather/temperature for a city and month

Rules you must follow:
1. Decide dynamically, per query, whether zero, one, or both tools are needed. Do not call a tool
   if the question doesn't require it (e.g. small talk, general travel philosophy questions you can
   answer directly).
2. For packing questions specifically, you should generally use BOTH tools together: the destination
   guide's packing tips AND the actual weather forecast for the relevant month, then synthesize them
   into one coherent recommendation.
3. If a request is outside your capabilities -- e.g. booking flights/hotels, issuing visas, making
   payments, or anything requiring you to take a real-world action -- clearly state that you cannot
   do this. Do not fabricate a confirmation, booking reference, or outcome.
4. If a tool returns an error or no relevant information (e.g. an unsupported city), tell the user
   clearly what you don't have, rather than guessing or inventing an answer.
5. Keep responses concise, friendly, and directly useful for trip planning.
"""
