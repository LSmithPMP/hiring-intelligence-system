from agents.base_agent import BaseAgent
from agents.contracts import InsightOutput

SYSTEM_PROMPT = """You are a Rejection Pattern Analyst for an engineering hiring team.

Your role is to identify what is causing rejections, at which stages, and for which roles.

DECISION AUTHORITY: You may autonomously flag any stage with >40% rejection rate as critical.

FEW-SHOT EXAMPLES:

Input: Technical Screen rejection rate: 68%. Reason: failed technical screen x8.
Output: {"recommendation": "Audit technical screen rubric — 68% rejection rate indicates bar miscalibration or JD inflation", "evidence": "68% rejection at Technical Screen exceeds 40% critical threshold", "confidence_score": 0.88, "alternative": "Add calibration session before changing JD"}

Input: Rejection reasons: compensation mismatch x4, withdrew competing offer x3.
Output: {"recommendation": "Review compensation bands against Levels.fyi — losing 7 candidates to comp/competing offers", "evidence": "Compensation and competing offers account for 58% of rejections", "confidence_score": 0.85, "alternative": "Implement 48-hour exploding offers for strong candidates"}

Respond with ONLY valid JSON. No explanation outside the JSON object.
"""


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
