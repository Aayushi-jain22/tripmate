import pytest

from tools.rag_tool import search_destination_guide, known_cities, RAGToolError


def test_known_cities_loaded():
    cities = known_cities()
    assert "Tokyo" in cities
    assert "Paris" in cities
    assert "Bangkok" in cities
    assert "Reykjavik" in cities
    assert len(cities) == 4


def test_search_returns_relevant_chunk_for_visa_query():
    results = search_destination_guide("do I need a visa to visit Japan", city_filter="Tokyo")
    assert results, "expected at least one result"
    assert any("visa" in r.lower() for r in results)
    assert all(r.startswith("[Tokyo") for r in results)


def test_search_returns_relevant_chunk_for_packing_query():
    results = search_destination_guide("what should I pack for Iceland", city_filter="Reykjavik")
    assert results
    assert any("pack" in r.lower() or "layer" in r.lower() for r in results)


def test_search_without_city_filter_still_finds_something():
    results = search_destination_guide("local customs and etiquette in Thailand")
    assert results
    assert any("Bangkok" in r for r in results)


def test_search_unknown_city_filter_returns_empty_not_error():
    results = search_destination_guide("packing tips", city_filter="Atlantis")
    assert results == []


def test_search_empty_query_raises():
    with pytest.raises(RAGToolError):
        search_destination_guide("")


def test_search_whitespace_query_raises():
    with pytest.raises(RAGToolError):
        search_destination_guide("   ")
