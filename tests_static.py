import ast
from pathlib import Path

ROOT = Path(__file__).parent
PY_FILES = sorted(ROOT.glob("*.py"))


def source(name):
    return (ROOT / name).read_text(encoding="utf-8")


def assert_syntax():
    for path in PY_FILES:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def assert_grounding_api():
    api = source("api.py")
    assert "groundedness_check(" in api
    assert "if not is_in_scope(request.query):" in api
    assert "if not is_in_scope(data):" in api
    assert "blocked_grounding" in api


def assert_memory_demo():
    crew = source("crew_setup.py")
    assert "What is the return window for electronics?" in crew
    assert "What about defective ones?" in crew
    assert "crew_with_memory.invoke" in crew


def assert_tool_routing():
    crew = source("crew_setup.py")
    assert "ORDER_ID_PATTERN" in crew
    assert "if not match:" in crew
    assert 'Action: {tool_name}' in crew


def assert_observation_safe_parser():
    crew = source("crew_setup.py")
    assert "result_match = re.search(" in crew
    assert "last_message" in crew


def assert_websocket_chunks():
    api = source("api.py")
    assert '"type": "chunk"' in api
    assert '"type": "done"' in api
    assert "groundedness_check" in api


def assert_governance_claims():
    api = source("api.py")
    governance = source("governance.py")
    assert "MAX_QUERY_TOKENS = 500" in api
    assert "MAX_TOKEN_BUDGET = 500" in governance
    assert "Runtime enforcement: api.py rejects over-budget requests" in governance


def assert_review_tests():
    review = source("review_stage.py")
    assert "verdict1 = extract_review_verdict(result1)" in review
    assert "verdict2 = extract_review_verdict(result2)" in review
    assert '"approved": False' in review
    assert '"approved": True' in review


def assert_dataset_main_guard():
    dataset = source("dataset.py")
    assert 'if __name__ == "__main__":' in dataset


def assert_requirements_and_readme():
    req = source("requirements.txt")
    readme = source("README.md")
    for package in ("fastapi", "uvicorn", "pydantic", "crewai", "chromadb", "sentence-transformers", "langchain-core", "autogen-agentchat", "autogen-core"):
        assert package in req
    for heading in ("# SwiftKart", "## 2. Architecture", "## 4. Setup", "## 6. Run the API", "## 12. Evaluation", "## 14. AutoGen review stage"):
        assert heading in readme


def main():
    checks = [
        assert_syntax,
        assert_grounding_api,
        assert_memory_demo,
        assert_tool_routing,
        assert_observation_safe_parser,
        assert_websocket_chunks,
        assert_governance_claims,
        assert_review_tests,
        assert_dataset_main_guard,
        assert_requirements_and_readme,
    ]
    for check in checks:
        check()
        print(f"PASS: {check.__name__}")
    print(f"\n{len(checks)}/{len(checks)} static acceptance checks passed.")


if __name__ == "__main__":
    main()
