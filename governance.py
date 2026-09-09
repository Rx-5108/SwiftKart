from crew_setup import composer_agent, lookup_agent, retrieval_agent

MAX_TOKEN_BUDGET = 500


def estimate_tokens(text: str) -> int:
    """Rough token estimate using the ~4 characters/token heuristic."""
    return max(1, (len(text) + 3) // 4)


def check_cost_budget(query: str) -> dict[str, int | bool | str]:
    """Check whether a query stays within the configured input-token budget."""
    estimated_tokens = estimate_tokens(query)
    allowed = estimated_tokens <= MAX_TOKEN_BUDGET
    return {
        "allowed": allowed,
        "estimated_tokens": estimated_tokens,
        "budget": MAX_TOKEN_BUDGET,
        "message": (
            "Request within budget."
            if allowed
            else f"Request rejected: estimated {estimated_tokens} tokens exceeds budget of {MAX_TOKEN_BUDGET} tokens."
        ),
    }


def main():
    print("\n=== TASK 15a: LEAST-AUTONOMY ENFORCEMENT ===\n")
    for label, agent in (
        ("Retrieval Agent", retrieval_agent),
        ("Lookup Agent", lookup_agent),
        ("Composer Agent", composer_agent),
    ):
        tools = [tool.name for tool in agent.tools] if agent.tools else []
        print(f"Tools assigned to {label}: {tools}")

    print("\n=== TASK 15b: RISK CLASSIFICATION ===\n")
    risk_level = "Medium"
    justification = (
        "This system is classified as Medium risk because it provides customer support, "
        "order-status lookups, and policy guidance. It uses masked customer contact/payment "
        "information in guardrail tests, but it does not make medical, hiring, or high-stakes "
        "financial decisions."
    )
    print(f"Risk Level: {risk_level}")
    print(f"Justification: {justification}")

    print("\n=== TASK 15c: RUNTIME COST-BUDGET CAP ===\n")
    normal_query = "What is the return window for electronics?"
    oversized_query = normal_query * 100
    print("Normal query check:", check_cost_budget(normal_query))
    print("\nOversized query check:", check_cost_budget(oversized_query))
    print("\nRuntime enforcement: api.py rejects over-budget requests before CrewAI execution.")


if __name__ == "__main__":
    main()
