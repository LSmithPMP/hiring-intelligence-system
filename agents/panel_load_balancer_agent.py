from agents.base_agent import BaseAgent
from agents.contracts import InsightOutput

SYSTEM_PROMPT = """You are a Panel Load Balancer for an engineering hiring team.
Analyze interviewer assignment data and return a JSON object with exactly these fields:
{
    "recommendation": "specific actionable suggestion about how to rebalance interviewer load",
    "evidence": "specific data points showing who is overloaded and who is underutilized",
    "confidence_score": 0.0-1.0,
    "alternative": "a cheaper or faster approach with trade-off note"
}
Identify interviewers who are drowning in panels and those who are barely assigned.
Base your analysis only on data provided. Never invent load numbers."""


class PanelLoadBalancerAgent(BaseAgent):
    def __init__(self, model: str = "gpt-4o-mini"):
        super().__init__("PanelLoadBalancerAgent", model)

    def run(self, data_context: str) -> InsightOutput:
        user_prompt = f"""Analyze this interviewer assignment data and identify who 
is overloaded, who is underutilized, and how the panel load should be rebalanced.

DATA:
{data_context}

Return a JSON object with recommendation, evidence, confidence_score, and alternative."""

        response, run = self.call_llm(SYSTEM_PROMPT, user_prompt)
        return self.parse_insight(response, run)
