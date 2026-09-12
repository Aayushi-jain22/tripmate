import pytest

from tools.weather_tool import get_weather_forecast, WeatherToolError


def test_get_weather_known_city_month_name():
    result = get_weather_forecast("Tokyo", "December")
    assert result["city"] == "Tokyo"
    assert result["month"] == "December"
    assert isinstance(result["temp_range_c"], list) and len(result["temp_range_c"]) == 2
    assert "snow" in result["conditions"].lower() or "cold" in result["conditions"].lower()


def test_get_weather_month_number():
    result = get_weather_forecast("Paris", "7")
    assert result["month"] == "July"
    assert "warm" in result["conditions"].lower() or "dry" in result["conditions"].lower()


def test_get_weather_iso_date():
    result = get_weather_forecast("Bangkok", "2026-01-15")
    assert result["month"] == "January"


def test_get_weather_case_insensitive_city():
    result = get_weather_forecast("REYKJAVIK", "june")
    assert result["city"] == "Reykjavik"


def test_get_weather_unknown_city_raises():
    with pytest.raises(WeatherToolError):
        get_weather_forecast("Atlantis", "June")


def test_get_weather_unparseable_month_raises():
    with pytest.raises(WeatherToolError):
        get_weather_forecast("Tokyo", "sometime soon")


def test_get_weather_empty_city_raises():
    with pytest.raises(WeatherToolError):
        get_weather_forecast("", "June")


def test_get_weather_invalid_month_number_raises():
    with pytest.raises(WeatherToolError):
        get_weather_forecast("Tokyo", "13")
