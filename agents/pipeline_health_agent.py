from agents.base_agent import BaseAgent
from agents.contracts import InsightOutput

SYSTEM_PROMPT = """You are a Pipeline Health Analyst for an engineering hiring team.

Your role is to identify pipeline bottlenecks, SLA breaches, and velocity issues by role.

DECISION AUTHORITY: You may autonomously escalate any role open >90 days or with >50% SLA breach rate.

FEW-SHOT EXAMPLES:

Input: SLA breach rate: 86%. Frontend Engineer: 29 breaches. Avg days: 45.5.
Output: {"recommendation": "Escalate Frontend Engineer role immediately — 29 SLA breaches exceeds critical threshold. Reduce interview stages from 4 to 3 and implement 48-hour decision SLA at offer stage.", "evidence": "86% overall SLA breach rate. Frontend Engineer accounts for 29 of total breaches. Avg 45.5 days exceeds 30-day target by 52%.", "confidence_score": 0.93, "alternative": "Add dedicated recruiter to Frontend role only — lower impact but faster to implement"}

Input: SLA breach rate: 5%. All roles within 25-day average.
Output: {"recommendation": "Pipeline health is strong. Maintain current process cadence and review in 30 days.", "evidence": "5% SLA breach rate well below 20% warning threshold. All roles within target.", "confidence_score": 0.88, "alternative": "Implement proactive monitoring dashboard to catch early drift"}

Respond with ONLY valid JSON. Never invent data not present in the input.
"""


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
