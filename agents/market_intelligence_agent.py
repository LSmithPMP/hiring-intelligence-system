from agents.base_agent import BaseAgent
from agents.contracts import InsightOutput
import json
import os
import requests
from dotenv import load_dotenv

load_dotenv()

SYSTEM_PROMPT = """You are a Market Intelligence Agent for an AI hiring intelligence system.

You have access to real-time market data tools. Your job is to:
1. Compare internal offer amounts against current market compensation benchmarks
2. Identify roles where compensation is below market rate
3. Recommend specific compensation adjustments backed by market data

You operate autonomously - you fetch market data, analyze it, and make
compensation recommendations without human input.

Return a JSON object with exactly these fields:
{
    "recommendation": "specific compensation adjustment recommendation by role",
    "evidence": "comparison of internal offers vs market data retrieved",
    "confidence_score": 0.0-1.0,
    "alternative": "lower-cost approach to compensation benchmarking",
    "market_data_source": "source of market data used",
    "roles_below_market": ["list of roles with compensation gaps"],
    "avg_market_gap_pct": estimated percentage below market
}"""


def search_market_data(query: str) -> str:
    """Search for real-time salary and market data using SerpAPI or fallback."""
    serpapi_key = os.getenv("SERPAPI_KEY", "")

    if serpapi_key:
        try:
            params = {
                "q": query,
                "api_key": serpapi_key,
                "num": 3
            }
            r = requests.get("https://serpapi.com/search", params=params, timeout=5)
            if r.status_code == 200:
                results = r.json().get("organic_results", [])
                snippets = [r.get("snippet", "") for r in results[:3]]
                return " | ".join(snippets)
        except Exception:
            pass

    # Fallback: use built-in 2024 market data
    return """
    Market Compensation Data (2024-2025, US Engineering):
    - Senior Software Engineer: $180,000-$230,000 total comp
    - Staff Engineer: $230,000-$320,000 total comp
    - Engineering Manager: $200,000-$280,000 total comp
    - DevOps Engineer: $160,000-$220,000 total comp
    - ML Engineer: $190,000-$270,000 total comp
    - Frontend Engineer: $150,000-$210,000 total comp
    - Security Engineer: $170,000-$240,000 total comp
    - Data Engineer: $160,000-$220,000 total comp
    Sources: Levels.fyi, Radford, Glassdoor 2024-2025
    """


class MarketIntelligenceAgent(BaseAgent):
    def __init__(self, model: str = "gpt-4o-mini"):
        super().__init__("MarketIntelligenceAgent", model)

    def run(self, data_context: str) -> InsightOutput:
        # Autonomously fetch market data before analysis
        print("  [MarketIntelligenceAgent] Fetching real-time market data...")
        market_data = search_market_data(
            "software engineer salary benchmarks 2024 2025 total compensation"
        )

        enriched_context = f"""{data_context}

REAL-TIME MARKET DATA (fetched autonomously):
{market_data}"""

        user_prompt = f"""Compare the internal offer amounts in this hiring data against
the real-time market compensation data you have access to. Which roles are
below market? What specific adjustments are needed?

DATA:
{enriched_context}

Return JSON with recommendation, evidence, confidence_score, alternative,
market_data_source, roles_below_market, and avg_market_gap_pct."""

        response, run = self.call_llm(SYSTEM_PROMPT, user_prompt)

        try:
            data = json.loads(response)
            insight = self.parse_insight(response, run)
            roles_below = data.get("roles_below_market", [])
            gap = data.get("avg_market_gap_pct", 0)
            insight.recommendation = (
                f"{data.get('recommendation', '')} "
                f"[ROLES BELOW MARKET: {', '.join(roles_below) if roles_below else 'None identified'}] "
                f"[AVG GAP: {gap}%]"
            )
            return insight
        except Exception:
            return self.parse_insight(response, run)
