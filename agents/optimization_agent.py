from agents.base_agent import BaseAgent
from agents.contracts import InsightOutput
import json
import os

SYSTEM_PROMPT = """You are an Optimization Agent for an AI hiring intelligence system.

Your job is to analyze pipeline run history and make autonomous decisions about:
1. Whether the current model routing thresholds should be adjusted
2. Which agents are underperforming and need prompt improvements
3. What the estimated cost savings would be from your recommendations

You have full authority to recommend configuration changes without human approval.
Base all decisions on the data provided. Never invent metrics.

Return a JSON object with exactly these fields:
{
    "recommendation": "specific optimization action to take immediately",
    "evidence": "cost and performance data supporting the decision",
    "confidence_score": 0.0-1.0,
    "alternative": "conservative fallback option with trade-off note",
    "autonomous_decision": "the specific threshold or config change you are implementing",
    "estimated_savings_pct": 0-100,
    "agents_to_reprompt": ["list of agent names that need prompt improvements"]
}"""


class OptimizationAgent(BaseAgent):
    def __init__(self, model: str = "gpt-4o-mini"):
        super().__init__("OptimizationAgent", model)

    def run(self, data_context: str) -> InsightOutput:
        user_prompt = f"""Analyze this pipeline performance data and make autonomous
optimization decisions. What should change right now to reduce cost and
improve quality?

DATA:
{data_context}

Return JSON with recommendation, evidence, confidence_score, alternative,
autonomous_decision, estimated_savings_pct, and agents_to_reprompt."""

        response, run = self.call_llm(SYSTEM_PROMPT, user_prompt)

        try:
            data = json.loads(response)
            insight = self.parse_insight(response, run)
            insight.recommendation = (
                f"{data.get('recommendation', '')} "
                f"[AUTONOMOUS DECISION: {data.get('autonomous_decision', 'No change')}] "
                f"[EST. SAVINGS: {data.get('estimated_savings_pct', 0)}%]"
            )
            return insight
        except Exception:
            return self.parse_insight(response, run)
