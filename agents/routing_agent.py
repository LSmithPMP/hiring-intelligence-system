from agents.base_agent import BaseAgent
from agents.contracts import InsightOutput
import json

SYSTEM_PROMPT = """You are a Model Routing Agent for an AI hiring intelligence system.
Your job is to decide which model to use for a given task based on complexity.

Rules:
- Use gpt-4o-mini for: simple aggregations, counting, basic pattern matching
- Use gpt-4o for: complex reasoning, ambiguous data, multi-factor analysis

Return a JSON object with exactly these fields:
{
    "selected_model": "gpt-4o-mini" or "gpt-4o",
    "reasoning": "one sentence explaining why",
    "complexity_score": 0.0-1.0
}"""


class RoutingAgent(BaseAgent):
    def __init__(self):
        super().__init__("RoutingAgent", model="gpt-4o-mini")

    def route(self, agent_name: str, data_summary: str) -> dict:
        user_prompt = f"""Decide which model to use for this agent and task.

Agent: {agent_name}
Data summary: {data_summary}

Return JSON with selected_model, reasoning, and complexity_score."""

        response, run = self.call_llm(SYSTEM_PROMPT, user_prompt)
        try:
            decision = json.loads(response)
            model = decision.get("selected_model", "gpt-4o-mini")
            complexity = decision.get("complexity_score", 0.5)
            print(f"[Router] {agent_name} -> {model} (complexity: {complexity})")
            return decision
        except Exception:
            return {"selected_model": "gpt-4o-mini", "reasoning": "Default fallback", "complexity_score": 0.5}

    def run(self, data_context: str) -> InsightOutput:
        pass
