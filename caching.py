import time

from typing import Any

from rag_core import fixed_collection, generate_grounded_answer

response_cache: dict[tuple[int, str, int, float], Any] = {}
call_counter = {"count": 0}


def normalize_query(query: str) -> str:
    """Normalize harmless whitespace/case differences."""
    return " ".join(query.strip().lower().split())


def generate_grounded_answer_cached(
    query: str, collection, top_k: int = 3, threshold: float = 0.30
) -> Any:
    """Return a cached result when the complete request configuration matches."""
    cache_key = (id(collection), normalize_query(query), top_k, threshold)

    if cache_key in response_cache:
        print(f"[CACHE HIT] Returning cached result for: '{query}'")
        return response_cache[cache_key]

    print(f"[CACHE MISS] Computing fresh result for: '{query}'")
    call_counter["count"] += 1
    result = generate_grounded_answer(query, collection, top_k, threshold)
    response_cache[cache_key] = result
    return result


def main():
    print("\n=== TASK 16: RESPONSE CACHING ===\n")
    test_query = "What is the return window for electronics?"

    start1 = time.perf_counter()
    result1 = generate_grounded_answer_cached(test_query, fixed_collection)
    time1 = time.perf_counter() - start1

    start2 = time.perf_counter()
    result2 = generate_grounded_answer_cached(test_query, fixed_collection)
    time2 = time.perf_counter() - start2

    print(f"\nFirst call time: {time1 * 1000:.2f} ms")
    print(f"Second call time: {time2 * 1000:.2f} ms")
    print(f"Total actual computation calls made: {call_counter['count']}")
    print(f"Results match: {result1 == result2}")


if __name__ == "__main__":
    main()
