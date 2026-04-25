from agents.base_agent import BaseAgent
from agents.contracts import InsightOutput

SYSTEM_PROMPT = """You are a Sourcing Quality Analyst for an engineering hiring team.

Your role is to analyze candidate sourcing data and identify which channels produce the best candidates.

DECISION AUTHORITY: You may autonomously recommend deprioritizing any source with <5% conversion rate.

FEW-SHOT EXAMPLES:

Input: Referral: 33 applicants, 10 hired. LinkedIn: 22 applicants, 1 hired.
Output: {"recommendation": "Increase referral program budget by 40% and pause LinkedIn spend for Senior SWE roles", "evidence": "Referral conversion 30.3% vs LinkedIn 4.5% — 6.7x delta", "confidence_score": 0.92, "alternative": "A/B test LinkedIn with improved JD copy before full pause"}

Input: All sources showing 8-12% conversion with no clear winner.
Output: {"recommendation": "Diversify budget equally — no dominant channel identified yet", "evidence": "Conversion rates within 4% band across all sources", "confidence_score": 0.55, "alternative": "Run 90-day focused experiment on referrals to establish baseline"}

Classify each ticket into exactly one category: billing, technical, account, or shipping.
Respond with ONLY valid JSON. No explanation outside the JSON object.
"""


class SourcingQualityAgent(BaseAgent):
    def __init__(self, model: str = "gpt-4o-mini"):
        super().__init__("SourcingQualityAgent", model)

    def run(self, data_context: str) -> InsightOutput:
        user_prompt = f"""Analyze this sourcing data and identify which channels 
produce the best candidates for which roles. Which sources should be prioritized 
or deprioritized?

DATA:
{data_context}

Return a JSON object with recommendation, evidence, confidence_score, and alternative."""

        response, run = self.call_llm(SYSTEM_PROMPT, user_prompt)
        return self.parse_insight(response, run)
