from agents.base_agent import BaseAgent
from agents.contracts import InsightOutput

SYSTEM_PROMPT = """You are a Pipeline Health Analyst for an engineering hiring team.
Analyze hiring pipeline velocity and SLA data and return a JSON object with exactly these fields:
{
    "recommendation": "specific actionable suggestion about pipeline bottlenecks and SLA breaches",
    "evidence": "specific data points showing where the funnel is slow or broken",
    "confidence_score": 0.0-1.0,
    "alternative": "a cheaper or faster approach with trade-off note"
}
Identify which stages are bottlenecks, which roles have been open too long,
and where SLA breaches are concentrated.
Base your analysis only on data provided. Never invent pipeline benchmarks."""


class PipelineHealthAgent(BaseAgent):
    def __init__(self, model: str = "gpt-4o-mini"):
        super().__init__("PipelineHealthAgent", model)

    def run(self, data_context: str) -> InsightOutput:
        user_prompt = f"""Analyze this pipeline data and identify where the funnel
is moving too slowly, where SLA breaches are concentrated, and which roles
have been open too long.

DATA:
{data_context}

Return a JSON object with recommendation, evidence, confidence_score, and alternative."""

        response, run = self.call_llm(SYSTEM_PROMPT, user_prompt)
        return self.parse_insight(response, run)
