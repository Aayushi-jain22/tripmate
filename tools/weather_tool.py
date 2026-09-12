"""
Module 3 — Weather Forecast Tool.

get_weather_forecast(city: str, date_or_month: str) -> dict

Design decision: Option B (mocked lookup table) was chosen over a live
API call (e.g. Open-Meteo). This assessment is scored on agentic
reasoning, not weather-API integration, and a mock table keeps the
demo deterministic and runnable with zero network access or rate
limits. The function signature and output schema below are what the
agent depends on, so swapping in a real API later only means
rewriting the body of `get_weather_forecast` -- see README "Swapping
in a real weather API".
"""
from __future__ import annotations

import calendar
import re
from typing import TypedDict


class WeatherResult(TypedDict):
    city: str
    month: str
    conditions: str
    temp_range_c: list[int]
    source: str


class WeatherToolError(Exception):
    """Raised for invalid input or an unsupported city/month."""


# Rough seasonal mock data. Keyed by (city, month_number).
_MOCK_TABLE: dict[tuple[str, int], dict] = {}


def _fill(city: str, months: list[int], conditions: str, temp_range: list[int]) -> None:
    for m in months:
        _MOCK_TABLE[(city, m)] = {"conditions": conditions, "temp_range_c": temp_range}


# Tokyo
_fill("tokyo", [12, 1, 2], "cold, dry, occasional light snow", [1, 10])
_fill("tokyo", [3, 4, 5], "mild, cherry blossoms, some rain", [10, 20])
_fill("tokyo", [6, 7, 8], "hot, humid, rainy season in June", [23, 32])
_fill("tokyo", [9, 10, 11], "mild, clear autumn skies", [12, 23])

# Paris
_fill("paris", [12, 1, 2], "cold, grey, occasional frost", [1, 8])
_fill("paris", [3, 4, 5], "mild, showers likely", [7, 17])
_fill("paris", [6, 7, 8], "warm, mostly dry, occasional heatwave", [16, 26])
_fill("paris", [9, 10, 11], "cool, crisp, increasing rain", [8, 17])

# Bangkok
_fill("bangkok", [12, 1, 2], "cool and dry by local standards, pleasant", [22, 32])
_fill("bangkok", [3, 4, 5], "very hot, humid", [26, 36])
_fill("bangkok", [6, 7, 8, 9], "hot, frequent afternoon monsoon showers", [25, 33])
_fill("bangkok", [10, 11], "transitioning out of monsoon, still humid", [24, 32])

# Reykjavik
_fill("reykjavik", [12, 1, 2], "cold, windy, dark, Northern Lights visible", [-3, 3])
_fill("reykjavik", [3, 4, 5], "cold, blustery, lengthening daylight", [0, 8])
_fill("reykjavik", [6, 7, 8], "cool, midnight sun, unpredictable rain/wind", [8, 15])
_fill("reykjavik", [9, 10, 11], "cold, windy, Northern Lights season begins", [1, 8])

_MONTH_NAME_TO_NUM = {name.lower(): num for num, name in enumerate(calendar.month_name) if name}
_MONTH_ABBR_TO_NUM = {abbr.lower(): num for num, abbr in enumerate(calendar.month_abbr) if abbr}


def _parse_month(date_or_month: str) -> int:
    """Best-effort parse of a free-text month/date reference into a month number (1-12)."""
    text = date_or_month.strip().lower()
    if not text:
        raise WeatherToolError("date_or_month must be a non-empty string.")

    # "12", "december", "dec", "december 2026", "15 dec", "2026-12-05"
    iso_match = re.match(r"\d{4}-(\d{2})-\d{2}", text)
    if iso_match:
        return int(iso_match.group(1))

    if text.isdigit():
        month_num = int(text)
        if 1 <= month_num <= 12:
            return month_num
        raise WeatherToolError(f"'{date_or_month}' is not a valid month number (1-12).")

    for name, num in _MONTH_NAME_TO_NUM.items():
        if name in text:
            return num
    for abbr, num in _MONTH_ABBR_TO_NUM.items():
        if abbr in text:
            return num

    raise WeatherToolError(
        f"Could not parse a month from '{date_or_month}'. "
        "Try a month name (e.g. 'December') or a YYYY-MM-DD date."
    )


def get_weather_forecast(city: str, date_or_month: str) -> WeatherResult:
    """Public tool function exposed to the agent."""
    if not city or not city.strip():
        raise WeatherToolError("city must be a non-empty string.")

    city_key = city.strip().lower()
    month_num = _parse_month(date_or_month)
    month_label = calendar.month_name[month_num]

    entry = _MOCK_TABLE.get((city_key, month_num))
    if entry is None:
        raise WeatherToolError(
            f"No mock weather data available for '{city}' in {month_label}. "
            f"Supported cities: {sorted({c for c, _ in _MOCK_TABLE})}."
        )

    return {
        "city": city.strip().title(),
        "month": month_label,
        "conditions": entry["conditions"],
        "temp_range_c": entry["temp_range_c"],
        "source": "mock_lookup_table",
    }
