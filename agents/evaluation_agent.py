from agents.base_agent import BaseAgent
from agents.contracts import InsightOutput
import json

SYSTEM_PROMPT = """You are an Evaluation Agent for an AI hiring intelligence system.
Your job is to judge the quality of insights produced by other agents.

Score each insight on three dimensions and return a JSON object:
{
    "actionability_score": 0.0-1.0,
    "grounding_score": 0.0-1.0,
    "hallucination_risk": 0.0-1.0,
    "overall_score": 0.0-1.0,
    "passed": true or false,
    "flags": ["list of specific concerns if any"],
    "judgment": "one sentence summary of the insight quality"
}

Scoring guide:
- actionability: Is the recommendation specific enough to act on today?
- grounding: Is the evidence drawn from actual data in the input?
- hallucination_risk: Does the insight invent facts not present in the input?
- passed: true if overall_score >= 0.6 and hallucination_risk <= 0.3
"""


class EvaluationAgent(BaseAgent):
    def __init__(self):
        super().__init__("EvaluationAgent", model="gpt-4o-mini")

    def evaluate(self, insight: InsightOutput, original_data: str) -> dict:
        user_prompt = f"""Evaluate this insight against the original data.

INSIGHT:
Agent: {insight.agent_name}
Recommendation: {insight.recommendation}
Evidence: {insight.evidence}
Confidence: {insight.confidence_score}

ORIGINAL DATA:
{original_data[:2000]}

Return JSON with actionability_score, grounding_score, hallucination_risk,
overall_score, passed, flags, and judgment."""

        response, run = self.call_llm(SYSTEM_PROMPT, user_prompt)
        try:
            result = json.loads(response)
            result["evaluated_agent"] = insight.agent_name
            result["eval_cost_usd"] = run.estimated_usd
            return result
        except Exception as e:
            return {
                "evaluated_agent": insight.agent_name,
                "actionability_score": 0.0,
                "grounding_score": 0.0,
                "hallucination_risk": 1.0,
                "overall_score": 0.0,
                "passed": False,
                "flags": [f"Parse error: {str(e)}"],
                "judgment": "Evaluation failed — could not parse response",
                "eval_cost_usd": run.estimated_usd
            }

    def run(self, data_context: str) -> InsightOutput:
        pass
