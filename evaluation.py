from crew_setup import detect_prompt_injection
from rag_core import fixed_collection, generate_grounded_answer


def mock_llm_judge(query: str, answer: str, similarity: float) -> dict[str, float]:
    """Return deterministic proxy scores; this is not a real LLM judge."""
    del query
    accuracy = min(1.0, similarity + 0.2) if similarity > 0 else 0.3
    grounding = max(0.0, min(1.0, similarity))
    completeness = min(1.0, len(answer) / 200) if answer else 0.0
    safety = 0.0 if detect_prompt_injection(answer) else 1.0
    return {
        "accuracy": round(accuracy, 2),
        "grounding": round(grounding, 2),
        "completeness": round(completeness, 2),
        "safety": round(safety, 2),
    }


test_queries = [
    "What is the return window for electronics?",
    "How long does COD refund take?",
    "What is the delivery timeline for standard orders?",
    "Can I get a reverse pickup for my return?",
    "What is the warranty period for appliances?",
    "Can I cancel my order after placing it?",
    "How many loyalty points do I earn per purchase?",
    "What happens if my payment fails?",
    "Can I exchange my apparel for a different size?",
    "What should I do if I receive a damaged item?",
    "Does SwiftKart ship internationally?",
    "How are customer support tickets escalated?",
    "What is the return policy for beauty products?",
    "What is the capital of France?",
    "How do I bake a chocolate cake?",
]


def main():
    print("\n=== TASK 13: MOCK LLM-AS-JUDGE EVALUATION (15 QUERIES) ===\n")
    all_scores = []
    for i, query in enumerate(test_queries, 1):
        answer, similarity = generate_grounded_answer(query, fixed_collection)
        scores = mock_llm_judge(query, answer, similarity)
        all_scores.append(scores)
        print(f"Query {i}: {query}")
        print(
            f"  Accuracy: {scores['accuracy']}, Grounding: {scores['grounding']}, "
            f"Completeness: {scores['completeness']}, Safety: {scores['safety']}"
        )

    averages = {
        key: sum(score[key] for score in all_scores) / len(all_scores)
        for key in all_scores[0]
    }
    print("\n=== AVERAGE SCORES ACROSS 15 QUERIES ===")
    for key, value in averages.items():
        print(f"Avg {key.title()}: {value:.2f}")


if __name__ == "__main__":
    main()
