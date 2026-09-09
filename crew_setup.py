import re
from typing import Any, Optional

from crewai import Agent, Crew, Task
from crewai.llms.base_llm import BaseLLM
from crewai.tools import tool
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory
from pydantic import BaseModel

from rag_core import fixed_collection, generate_grounded_answer, retrieve_chunks
from tools import check_order_status


ORDER_ID_PATTERN = re.compile(r"\bORD-\d{4}\b", re.IGNORECASE)


class MockLLM(BaseLLM):
    """Deterministic mock LLM used to demonstrate CrewAI tool routing."""

    def __init__(self):
        super().__init__(model="mock-llm")

    def call(self, messages, tools=None, callbacks=None, available_functions=None, **kwargs):
        system_message = (
            messages[0]["content"]
            if messages and isinstance(messages[0], dict) and messages[0].get("role") == "system"
            else ""
        )
        last_message = messages[-1]["content"] if isinstance(messages, list) else str(messages)
        tool_match = re.search(r"Tool Name:\s*(\S+)", system_message)
        tool_name = tool_match.group(1) if tool_match else None

        # The composer has no tools. It returns only the supplied task context.
        if not tool_name:
            context_match = re.search(
                r"context you're working with:\s*(.*?)\s*Provide your complete response:",
                last_message,
                re.DOTALL | re.IGNORECASE,
            )
            answer = context_match.group(1).strip() if context_match else last_message.strip()
            answer = re.sub(
                r"Analyze the tool result\..*?(?=\n-{3,}|\Z)",
                "",
                answer,
                flags=re.DOTALL | re.IGNORECASE,
            ).strip()
            return f"Thought: I now know the final answer\nFinal Answer: {answer}"

        full_conversation = "\n".join(
            str(message.get("content", ""))
            for message in messages
            if isinstance(message, dict) and message.get("content")
        )

        if tool_name == "order_status_tool":
            # Read the current tool-result message first so an older/example Observation
            # cannot be mistaken for the latest order result.
            result_match = re.search(
                r"\{'status':\s*.*?\'escalation_score\':\s*[\d.]+\}",
                last_message,
                re.DOTALL,
            )
            if result_match:
                return f"Thought: I now know the final answer\nFinal Answer: {result_match.group(0)}"

        if tool_name == "rag_search_tool":
            result_match = re.search(
                r"(Based on SwiftKart policy:.*|"
                r"I don't have enough information in the provided policy documents\.)",
                last_message,
                re.DOTALL,
            )
            if result_match:
                return f"Thought: I now know the final answer\nFinal Answer: {result_match.group(1).strip()}"

        # Least-autonomy rule: order lookup is allowed only when a real order ID exists.
        if tool_name == "order_status_tool":
            match = ORDER_ID_PATTERN.search(last_message)
            if not match:
                return (
                    "Thought: No order ID was provided.\n"
                    "Final Answer: No order lookup is required for this query."
                )
            record_id = match.group().upper()
            action_input = f'{{"record_id": "{record_id}"}}'
        else:
            task_match = re.search(r"Current Task:\s*(.*)", last_message, re.DOTALL)
            query = task_match.group(1).strip() if task_match else last_message.strip()
            action_input = f'{{"query": {query!r}}}'

        return (
            f"Thought: I need to use the {tool_name} tool.\n"
            f"Action: {tool_name}\n"
            f"Action Input: {action_input}"
        )


@tool("rag_search_tool")
def rag_search_tool(query: str) -> str:
    """Search SwiftKart's policy knowledge base for relevant information."""
    answer, _similarity = generate_grounded_answer(query, fixed_collection)
    return answer


@tool("order_status_tool")
def order_status_tool(record_id: str) -> str:
    """Check a specific SwiftKart order and return its status and escalation score."""
    return str(check_order_status(record_id))


mock_llm = MockLLM()

retrieval_agent = Agent(
    role="Retrieval Specialist",
    goal="Find relevant policy information from SwiftKart's knowledge base",
    backstory="Expert at searching SwiftKart policy documents to answer customer questions.",
    llm=mock_llm,
    tools=[rag_search_tool],
    verbose=True,
)

lookup_agent = Agent(
    role="Order Status Specialist",
    goal="Look up specific order details and status using an order ID",
    backstory="Expert at retrieving order status, tracking information, and escalation scores.",
    llm=mock_llm,
    tools=[order_status_tool],
    verbose=True,
)

composer_agent = Agent(
    role="Response Composer",
    goal="Combine retrieved policy information and order status into one clear customer answer",
    backstory="Expert customer-support writer who synthesizes retrieved information accurately.",
    llm=mock_llm,
    verbose=True,
)

retrieval_task = Task(
    description="Search the knowledge base to answer: {query}",
    expected_output="Relevant policy information from the knowledge base",
    agent=retrieval_agent,
)

lookup_task = Task(
    description="Look up order status for: {query}. Only use the order-status tool when an ORD-xxxx ID is present.",
    expected_output="Order status, value, and escalation score when an order ID is provided; otherwise no lookup",
    agent=lookup_agent,
)

compose_task = Task(
    description=(
        "Combine the retrieved policy information and, when present, order status into one concise "
        "customer-facing answer for: {query}. Do not mention tools, agents, internal instructions, "
        "or task execution. If no order information is provided, answer using policy information only."
    ),
    expected_output="A clear, friendly final answer for the customer",
    agent=composer_agent,
    context=[retrieval_task, lookup_task],
)

crew = Crew(
    agents=[retrieval_agent, lookup_agent, composer_agent],
    tasks=[retrieval_task, lookup_task, compose_task],
    verbose=True,
)


# --- Session memory ---
session_store: dict[str, InMemoryChatMessageHistory] = {}


def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    """Return or create the in-memory conversation history for a session."""
    if session_id not in session_store:
        session_store[session_id] = InMemoryChatMessageHistory()
    return session_store[session_id]


def run_crew_with_query(input_dict: dict[str, Any]) -> str:
    """Run CrewAI with prior conversation history included in the current task."""
    query = input_dict["query"]
    history = input_dict.get("history", "")
    memory_context = f"\nConversation history:\n{history}\n" if history else ""
    result = crew.kickoff(inputs={"query": f"{memory_context}\nCurrent user query: {query}"})
    return str(result)


crew_runnable = RunnableLambda(run_crew_with_query)
crew_with_memory = RunnableWithMessageHistory(
    crew_runnable,
    get_session_history,
    input_messages_key="query",
    history_messages_key="history",
)


class CrewResponse(BaseModel):
    """Normalized customer response returned by the application layer."""

    answer: str
    query: str
    escalation_score: Optional[float] = None


def clean_crew_answer(raw_output: str) -> str:
    """Extract the final customer-facing answer without internal CrewAI traces."""
    answer = str(raw_output).strip()
    final_match = re.search(r"Final Answer:\s*(.*)", answer, re.DOTALL | re.IGNORECASE)
    if final_match:
        answer = final_match.group(1).strip()

    answer = re.sub(
        r"\n?-{3,}\n?.*?Analyze the tool result\..*?(?=\n?-{3,}|$)",
        "",
        answer,
        flags=re.DOTALL | re.IGNORECASE,
    )
    answer = re.sub(r"Analyze the tool result\..*", "", answer, flags=re.DOTALL | re.IGNORECASE)
    return answer.strip()


def parse_crew_output_to_schema(raw_output: str, original_query: str) -> CrewResponse:
    """Parse only the final answer and order-status escalation score from Crew output."""
    final_answer = clean_crew_answer(raw_output)
    escalation_match = re.search(r"'escalation_score':\s*([\d.]+)", final_answer)
    escalation_score = float(escalation_match.group(1)) if escalation_match else None
    return CrewResponse(
        answer=final_answer,
        query=original_query,
        escalation_score=escalation_score,
    )


# --- Guardrails ---
def mask_pii(text: str) -> str:
    """Mask common phone numbers and card-ending fragments before agent processing."""
    masked = re.sub(r"\b\d{10}\b", "[PHONE_MASKED]", text)
    return re.sub(
        r"\bcard\s+(?:ending\s+in\s+)?\d{4}\b",
        "card ending in [CARD_MASKED]",
        masked,
        flags=re.IGNORECASE,
    )


def detect_prompt_injection(text: str) -> bool:
    """Detect common direct prompt-injection phrases before tool execution."""
    injection_patterns = [
        r"ignore\s+(all\s+)?previous\s+instructions",
        r"disregard\s+(all\s+)?(prior|previous)\s+instructions",
        r"you\s+are\s+now\s+(a|an)\s+",
        r"system\s*:\s*",
        r"forget\s+(everything|all)\s+",
    ]
    return any(re.search(pattern, text, re.IGNORECASE) for pattern in injection_patterns)


def groundedness_check(query: str, collection, threshold: float = 0.30) -> bool:
    """Return True when the best policy retrieval meets the configured threshold."""
    results = retrieve_chunks(query, collection, top_k=1)
    if not results or not results.get("distances") or not results["distances"][0]:
        return False
    top_similarity = 1 - results["distances"][0][0]
    return top_similarity >= threshold


if __name__ == "__main__":
    print("\n=== CREW TOOL ASSIGNMENTS ===")
    print("Retrieval tools:", [tool.name for tool in retrieval_agent.tools])
    print("Lookup tools:", [tool.name for tool in lookup_agent.tools])
    print("Composer tools:", [tool.name for tool in composer_agent.tools] if composer_agent.tools else [])

    result = crew.kickoff(
        inputs={"query": "What is the return window for electronics, and what's the status of ORD-1023?"}
    )
    print("\n=== CREW RESULT ===")
    print(result)

    print("\n=== DEPENDENT MEMORY TEST ===")
    config = {"configurable": {"session_id": "memory_demo"}}
    turn1 = crew_with_memory.invoke(
        {"query": "What is the return window for electronics?"}, config=config
    )
    turn2 = crew_with_memory.invoke({"query": "What about defective ones?"}, config=config)
    print("Turn 1:", turn1)
    print("Turn 2:", turn2)
    print("History:", get_session_history("memory_demo").messages)
