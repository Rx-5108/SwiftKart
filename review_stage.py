"""Task 14: policy-compliance review stage using AutoGen."""

import asyncio
import json

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_agentchat.messages import StructuredMessage
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_core import CancellationToken
from autogen_core.models import ChatCompletionClient, CreateResult, RequestUsage
from pydantic import BaseModel


class MockChatCompletionClient(ChatCompletionClient):
    """Deterministic mock client used to demonstrate the review workflow."""

    def __init__(self):
        self._total_usage = RequestUsage(prompt_tokens=0, completion_tokens=0)

    @property
    def model_info(self):
        return {
            "vision": False,
            "function_calling": False,
            "json_output": True,
            "family": "mock",
            "structured_output": True,
        }

    @property
    def capabilities(self):
        return self.model_info

    async def create(
        self,
        messages,
        *,
        tools=(),
        json_output=None,
        extra_create_args=None,
        cancellation_token: CancellationToken | None = None,
    ):
        system_content = str(messages[0].content) if messages else ""
        full_text = "\n".join(str(message.content) for message in messages)
        response_text = self._generate_mock_response(system_content, full_text)

        usage = RequestUsage(prompt_tokens=10, completion_tokens=10)
        self._total_usage = RequestUsage(
            prompt_tokens=self._total_usage.prompt_tokens + usage.prompt_tokens,
            completion_tokens=self._total_usage.completion_tokens + usage.completion_tokens,
        )

        return CreateResult(
            finish_reason="stop",
            content=response_text,
            usage=usage,
            cached=False,
        )

    async def create_stream(self, *args, **kwargs):
        raise NotImplementedError("Streaming is not needed for this demonstration.")

    def actual_usage(self):
        return self._total_usage

    def total_usage(self):
        return self._total_usage

    def count_tokens(self, messages, *, tools=()):
        return sum(max(1, len(str(message.content)) // 4) for message in messages)

    def remaining_tokens(self, messages, *, tools=()):
        return max(0, 4096 - self.count_tokens(messages, tools=tools))

    async def close(self):
        return None

    def _generate_mock_response(self, system_content: str, full_text: str) -> str:
        if "Policy-Compliance-Reviewer" in system_content:
            suspicious_terms = ("guaranteed", "100% approved", "free upgrade")
            if any(term in full_text.lower() for term in suspicious_terms):
                return (
                    "ISSUE FOUND: This draft contains an unsupported claim not present "
                    "in the retrieved policy context. It should be corrected before "
                    "sending to the customer."
                )
            return "REVIEWED: The draft appears consistent with the retrieved policy context. No issues found."

        if "Final-Editor" in system_content:
            if "ISSUE FOUND" in full_text:
                revised = (
                    "Based on SwiftKart policy, please refer to the official terms; "
                    "unverified claims have been removed for accuracy."
                )
                return json.dumps({
                    "approved": False,
                    "final_answer": revised,
                    "reason": "Removed the ungrounded claim flagged by the compliance reviewer.",
                })

            original_draft = (
                full_text.split("Draft:", 1)[-1].split("Retrieved Context:", 1)[0].strip()
                if "Draft:" in full_text
                else full_text.strip()
            )
            return json.dumps({
                "approved": True,
                "final_answer": original_draft,
                "reason": "Draft is accurate and grounded; approved without changes.",
            })

        return json.dumps({
            "approved": True,
            "final_answer": "No response",
            "reason": "Fallback response.",
        })


class ReviewVerdict(BaseModel):
    approved: bool
    final_answer: str
    reason: str


mock_client = MockChatCompletionClient()

compliance_reviewer = AssistantAgent(
    name="Policy_Compliance_Reviewer",
    model_client=mock_client,
    system_message=(
        "You are the Policy-Compliance-Reviewer. Check the draft answer for claims "
        "not supported by the retrieved policy context. Flag issues clearly."
    ),
)

final_editor = AssistantAgent(
    name="Final_Editor",
    model_client=mock_client,
    system_message=(
        "You are the Final-Editor. Based on the compliance review, either approve "
        "the draft unchanged or revise it. Always output your verdict in the structured format."
    ),
    output_content_type=ReviewVerdict,
)

review_team = RoundRobinGroupChat(
    participants=[compliance_reviewer, final_editor],
    termination_condition=MaxMessageTermination(3),
    custom_message_types=[StructuredMessage[ReviewVerdict]],
)


async def run_review(draft_answer: str, retrieved_context: str):
    """Run compliance review followed by the structured final editor."""
    task = f"Draft: {draft_answer}\n\nRetrieved Context: {retrieved_context}"
    return await review_team.run(task=task)


def extract_review_verdict(result) -> ReviewVerdict:
    """Return the final structured ReviewVerdict from an AutoGen team result."""
    for message in reversed(result.messages):
        content = getattr(message, "content", None)
        if isinstance(content, ReviewVerdict):
            return content
        if isinstance(content, str):
            try:
                payload = json.loads(content)
                return ReviewVerdict.model_validate(payload)
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
    raise ValueError("No valid ReviewVerdict was produced by the review team.")


async def main():
    print("\n=== TASK 14a: APPROVED CASE ===\n")
    good_draft = "SwiftKart allows returns within 7 days for apparel, provided items are unused."
    context = good_draft
    result1 = await run_review(good_draft, context)
    verdict1 = extract_review_verdict(result1)
    print(verdict1.model_dump())

    print("\n=== TASK 14b: REVISED CASE (injected false claim) ===\n")
    bad_draft = "SwiftKart offers a 100% approved guaranteed refund with free upgrade for all electronics."
    result2 = await run_review(bad_draft, context)
    verdict2 = extract_review_verdict(result2)
    print(verdict2.model_dump())


if __name__ == "__main__":
    asyncio.run(main())
