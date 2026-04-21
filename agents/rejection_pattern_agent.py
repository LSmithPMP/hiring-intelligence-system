from agents.base_agent import BaseAgent
from agents.contracts import InsightOutput

SYSTEM_PROMPT = """You are a Rejection Pattern Analyst for an engineering hiring team.
Analyze candidate rejection data and return a JSON object with exactly these fields:
{
    "recommendation": "specific actionable suggestion about what is causing rejections and how to fix it",
    "evidence": "specific data points from the input supporting your recommendation",
    "confidence_score": 0.0-1.0,
    "alternative": "a cheaper or faster approach with trade-off note"
}
Identify patterns by stage, role, and rejection reason.
Base your analysis only on data provided. Never invent statistics."""


class RejectionPatternAgent(BaseAgent):
    def __init__(self, model: str = "gpt-4o-mini"):
        super().__init__("RejectionPatternAgent", model)

    def run(self, data_context: str) -> InsightOutput:
        user_prompt = f"""Analyze this rejection data and identify what keeps going 
wrong, at which stages, and for which roles. Is the JD the problem or is it 
the interview process?

DATA:
{data_context}

Return a JSON object with recommendation, evidence, confidence_score, and alternative."""

        response, run = self.call_llm(SYSTEM_PROMPT, user_prompt)
        return self.parse_insight(response, run)
