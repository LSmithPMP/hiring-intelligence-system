"""
================================================================================
AI-POWERED HIRING INTELLIGENCE SYSTEM
Capstone 2 — Interview Kickstart Applied Agentic AI
Author: Lamonte Smith | Lamonte Smith Automotive, LLC | April 2026
GitHub: github.com/LSmithPMP/hiring-intelligence-system
================================================================================

SYSTEM OVERVIEW
---------------
This file documents the complete architecture of the AI-Powered Hiring
Intelligence System — a 9-agent multi-agent pipeline that analyzes engineering
talent acquisition data and delivers actionable, benchmark-grounded insights
in under 50 seconds for less than one cent per run.

TECH STACK
----------
- LangChain        — agent orchestration and LLM calls
- OpenAI           — GPT-4o-mini and GPT-4o as LLM backends
- ChromaDB         — vector store for RAG knowledge base
- n8n Cloud        — orchestration workflows (webhook + scheduled)
- FastAPI          — REST API layer with authentication
- Streamlit        — executive dashboard
- Pydantic v2      — shared output contract enforcement

TABLE OF CONTENTS
-----------------
1.  Environment Setup & Configuration
2.  Shared Output Contract (Pydantic)
3.  Base Agent Class (cost tracking, input sanitization, LLM calls)
4.  RAG Pipeline (ChromaDB knowledge base)
5.  Routing Agent (autonomous model selection)
6.  Sourcing Quality Agent
7.  Rejection Pattern Agent
8.  Panel Load Balancer Agent
9.  Offer Insights Agent
10. Pipeline Health Agent
11. Optimization Agent (autonomous)
12. Market Intelligence Agent (external tools + autonomous)
13. Evaluation Agent (LLM-as-judge)
14. Orchestrator (pipeline coordinator)
15. FastAPI Endpoints
16. Streamlit Dashboard
17. Golden Dataset Evaluation
18. n8n Workflow Integration
================================================================================
"""

# ==============================================================================
# SECTION 1: ENVIRONMENT SETUP & CONFIGURATION
# ==============================================================================
# All sensitive credentials are loaded from a .env file at runtime.
# The .env file is excluded from git via .gitignore — no credentials
# are ever committed to the repository.
#
# Required environment variables:
#   OPENAI_API_KEY      — OpenAI API key for LLM calls
#   LANGSMITH_API_KEY   — LangSmith tracing key
#   LANGSMITH_TRACING   — set to "true" to enable tracing
#   API_KEY             — FastAPI X-API-Key authentication key
#   N8N_WEBHOOK_SECRET  — n8n webhook secret header value
#   SERPAPI_KEY         — (optional) SerpAPI key for web search
# ==============================================================================

import os
import re
import time
import json
import uuid
import secrets
import requests
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from typing import Optional, Literal
from openai import OpenAI
from pydantic import BaseModel, Field
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter

load_dotenv()

# Initialize OpenAI client — key loaded from .env, never hardcoded
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Model pricing per 1M tokens (April 2026)
MODEL_PRICING = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o":      {"input": 2.50, "output": 10.00},
}

# Project paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "ats_mock_data.csv")
KB_PATH   = os.path.join(BASE_DIR, "knowledge_base", "hiring_benchmarks.md")
CHROMA_PATH = os.path.join(BASE_DIR, "knowledge_base", "chroma_db")


# ==============================================================================
# SECTION 2: SHARED OUTPUT CONTRACT (PYDANTIC)
# ==============================================================================
# Every agent — regardless of what it analyzes — returns the same 5-field
# structure. This is enforced by Pydantic v2 at runtime.
#
# WHY THIS MATTERS:
# - Enables reliable downstream processing without agent-specific parsing
# - Ensures every insight has a cost estimate attached
# - Forces agents to always provide an alternative (cheaper/faster option)
# - Makes evaluation consistent — EvaluationAgent scores the same fields
# ==============================================================================

class InsightOutput(BaseModel):
    """Shared output contract — enforced across all 9 agents."""
    agent_name:       str   = Field(description="Name of the agent producing this insight")
    recommendation:   str   = Field(description="Specific actionable suggestion — not a generic observation")
    evidence:         str   = Field(description="Data points from input that support the recommendation")
    confidence_score: float = Field(ge=0.0, le=1.0, description="Calibrated confidence 0.0-1.0")
    cost_of_insight:  dict  = Field(description="Model, tokens used, and estimated USD")
    alternative:      str   = Field(description="Cheaper/faster option with trade-off note")
    timestamp:        str   = Field(default_factory=lambda: datetime.utcnow().isoformat())


class AgentRun(BaseModel):
    """Tracks cost and latency for a single LLM call."""
    agent_name:        str
    model_used:        str
    input_tokens:      int
    output_tokens:     int
    latency_seconds:   float
    estimated_usd:     float
    success:           bool
    error_message:     Optional[str] = None
    timestamp:         str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class PipelineRun(BaseModel):
    """Tracks a complete pipeline execution across all agents."""
    run_id:                  str
    total_agents:            int
    successful_agents:       int
    failed_agents:           int
    total_input_tokens:      int
    total_output_tokens:     int
    total_latency_seconds:   float
    total_estimated_usd:     float
    agent_runs:              list[AgentRun]
    insights:                list[InsightOutput]
    timestamp:               str = Field(default_factory=lambda: datetime.utcnow().isoformat())


# ==============================================================================
# SECTION 3: BASE AGENT CLASS
# ==============================================================================
# All 9 agents inherit from BaseAgent. This class provides:
#
# 1. COST TRACKING      — every LLM call logs tokens and USD
# 2. INPUT SANITIZATION — prompt injection filtering before every call
# 3. LLM CALL WRAPPER   — structured JSON output format enforced
# 4. PARSE INSIGHT      — converts raw LLM JSON to InsightOutput contract
#
# SECURITY CONTROL: _sanitize_input() filters 12 injection patterns
# including "ignore previous instructions", "jailbreak", script tags,
# and SQL injection patterns. Applied before every LLM call.
# ==============================================================================

def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Calculate estimated USD cost for a model call."""
    pricing = MODEL_PRICING.get(model, MODEL_PRICING["gpt-4o-mini"])
    return round(
        (input_tokens  / 1_000_000) * pricing["input"] +
        (output_tokens / 1_000_000) * pricing["output"],
        6
    )


class BaseAgent:
    """Base class providing cost tracking, security controls, and LLM calls."""

    def __init__(self, agent_name: str, model: str = "gpt-4o-mini"):
        self.agent_name = agent_name
        self.model = model
        self.run_log: list[AgentRun] = []

    def _sanitize_input(self, text: str) -> str:
        """
        SECURITY CONTROL: Filter prompt injection patterns before LLM call.
        Blocks 12 attack patterns and truncates inputs > 8000 chars.
        """
        patterns = [
            r"ignore previous instructions",
            r"ignore all previous",
            r"disregard.*instructions",
            r"you are now",
            r"act as",
            r"jailbreak",
            r"<[|].*?[|]>",
            r"\[INST\]",
            r"###.*?###",
            r"eval[(]",
            r"exec[(]",
            r"DROP TABLE",
        ]
        cleaned = text
        for pattern in patterns:
            cleaned = re.sub(pattern, "[FILTERED]", cleaned, flags=re.IGNORECASE | re.DOTALL)
        if len(cleaned) > 8000:
            cleaned = cleaned[:8000] + "...[TRUNCATED]"
        return cleaned

    def call_llm(self, system_prompt: str, user_prompt: str) -> tuple[str, AgentRun]:
        """
        Make a tracked LLM call with:
        - Input sanitization (security)
        - JSON output format enforcement
        - Token and cost logging
        - Error handling with graceful fallback
        """
        start = time.time()
        user_prompt = self._sanitize_input(user_prompt)  # Security control applied here
        success, error_message, response_text = True, None, "{}"
        input_tokens, output_tokens = 0, 0

        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt}
                ],
                temperature=0.1,
                response_format={"type": "json_object"}  # Enforces structured output
            )
            response_text  = response.choices[0].message.content
            input_tokens   = response.usage.prompt_tokens
            output_tokens  = response.usage.completion_tokens
        except Exception as e:
            success, error_message = False, str(e)

        run = AgentRun(
            agent_name=self.agent_name,
            model_used=self.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_seconds=round(time.time() - start, 2),
            estimated_usd=calculate_cost(self.model, input_tokens, output_tokens),
            success=success,
            error_message=error_message
        )
        self.run_log.append(run)
        return response_text, run

    def _to_string(self, value) -> str:
        """Convert any LLM output type to string for Pydantic contract."""
        if isinstance(value, str):   return value
        if isinstance(value, dict):  return json.dumps(value)
        if isinstance(value, list):  return "; ".join(str(v) for v in value)
        return str(value)

    def parse_insight(self, raw_response: str, run: AgentRun) -> InsightOutput:
        """
        Parse raw LLM JSON into InsightOutput contract.
        On parse failure, returns a graceful error insight rather than crashing.
        This prevents a single agent failure from taking down the pipeline.
        """
        try:
            data = json.loads(raw_response)
            return InsightOutput(
                agent_name=self.agent_name,
                recommendation=self._to_string(data.get("recommendation", "No recommendation")),
                evidence=self._to_string(data.get("evidence", "No evidence")),
                confidence_score=float(data.get("confidence_score", 0.5)),
                cost_of_insight={
                    "model": run.model_used,
                    "input_tokens": run.input_tokens,
                    "output_tokens": run.output_tokens,
                    "estimated_usd": run.estimated_usd
                },
                alternative=self._to_string(data.get("alternative", "No alternative"))
            )
        except Exception as e:
            return InsightOutput(
                agent_name=self.agent_name,
                recommendation=f"Parse error: {str(e)}",
                evidence="Raw response could not be parsed",
                confidence_score=0.0,
                cost_of_insight={"model": run.model_used, "input_tokens": run.input_tokens,
                                  "output_tokens": run.output_tokens, "estimated_usd": run.estimated_usd},
                alternative="Fix parsing error before retry"
            )

    def run(self, data_context: str) -> InsightOutput:
        raise NotImplementedError("Each agent must implement run()")


# ==============================================================================
# SECTION 4: RAG PIPELINE (CHROMADB KNOWLEDGE BASE)
# ==============================================================================
# RAG (Retrieval-Augmented Generation) grounds every agent in real industry
# benchmarks before it generates insights.
#
# HOW IT WORKS:
# 1. hiring_benchmarks.md is split into 16 chunks by markdown headers
# 2. Each chunk is embedded using text-embedding-3-small
# 3. Before each agent runs, it queries ChromaDB for top-3 relevant chunks
# 4. Those chunks are appended to the agent's data context
#
# WHY THIS MATTERS:
# Without RAG, agents invent benchmarks from model priors — which vary
# unpredictably. With RAG, "referral conversion 30% vs LinkedIn 4.5%"
# comes from the knowledge base, not a hallucination.
#
# COST: text-embedding-3-small costs $0.00002/1K tokens — negligible.
# ==============================================================================

def build_vector_store() -> Chroma:
    """Index hiring_benchmarks.md into ChromaDB. Run once at setup."""
    loader = TextLoader(KB_PATH)
    documents = loader.load()
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "section"), ("##", "subsection"), ("###", "topic")]
    )
    splits = splitter.split_text(documents[0].page_content)
    for split in splits:
        split.metadata["source"] = "hiring_benchmarks"
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory=CHROMA_PATH)


def get_vector_store() -> Chroma:
    """Load existing ChromaDB vector store."""
    return Chroma(persist_directory=CHROMA_PATH,
                  embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"))


def retrieve_context(query: str, k: int = 3) -> str:
    """Retrieve top-k most relevant benchmark chunks for an agent query."""
    docs = get_vector_store().similarity_search(query, k=k)
    return "\n\n".join([doc.page_content for doc in docs])


# Per-agent RAG queries — each agent retrieves benchmarks relevant to its domain
RAG_QUERIES = {
    "SourcingQualityAgent":    "sourcing channel conversion rates benchmarks cost per hire",
    "RejectionPatternAgent":   "rejection patterns stage conversion rates benchmarks",
    "PanelLoadBalancerAgent":  "interviewer load balance panels per month benchmarks",
    "OfferInsightsAgent":      "offer acceptance rates decline reasons compensation benchmarks",
    "PipelineHealthAgent":     "SLA targets time to hire pipeline velocity benchmarks",
    "OptimizationAgent":       "cost optimization model selection prompt engineering benchmarks",
    "MarketIntelligenceAgent": "compensation benchmarks salary market rates engineering roles",
}


# ==============================================================================
# SECTION 5: ROUTING AGENT (AUTONOMOUS MODEL SELECTION)
# ==============================================================================
# The RoutingAgent is the first autonomous agent. It evaluates task complexity
# before each insight agent runs and selects the appropriate model WITHOUT
# human input.
#
# DECISION RULES:
# - Complexity score < 0.5  → gpt-4o-mini ($0.15/1M input tokens)
# - Complexity score >= 0.5 → gpt-4o     ($2.50/1M input tokens)
#
# COST IMPACT:
# Before routing: all gpt-4o → ~$0.040/run
# After routing:  mixed      → ~$0.007/run  (83% cost reduction)
#
# AUTONOMY: The RoutingAgent makes this decision independently every run.
# No human approves or overrides the model selection in real time.
# ==============================================================================

class RoutingAgent(BaseAgent):
    """Autonomously selects the optimal model per task complexity."""

    def __init__(self):
        super().__init__("RoutingAgent", model="gpt-4o-mini")

    SYSTEM_PROMPT = """You are a Model Routing Agent for an AI hiring intelligence system.
Evaluate task complexity and select the most cost-effective model.

Rules:
- Use gpt-4o-mini for: simple aggregations, counting, basic pattern matching
- Use gpt-4o for: complex reasoning, ambiguous data, multi-factor analysis

Return JSON: {"selected_model": "gpt-4o-mini" or "gpt-4o", "reasoning": "one sentence", "complexity_score": 0.0-1.0}"""

    def route(self, agent_name: str, data_summary: str) -> dict:
        """Make autonomous model selection decision."""
        response, run = self.call_llm(
            self.SYSTEM_PROMPT,
            f"Agent: {agent_name}\nData summary: {data_summary[:500]}\nReturn routing JSON."
        )
        try:
            decision = json.loads(response)
            model = decision.get("selected_model", "gpt-4o-mini")
            complexity = decision.get("complexity_score", 0.5)
            print(f"  [Router] {agent_name} -> {model} (complexity: {complexity})")
            return decision
        except Exception:
            return {"selected_model": "gpt-4o-mini", "reasoning": "Default fallback", "complexity_score": 0.5}

    def run(self, data_context: str) -> InsightOutput:
        pass  # RoutingAgent is called via route(), not run()


# ==============================================================================
# SECTION 6: SOURCING QUALITY AGENT
# ==============================================================================
# Analyzes which sourcing channels produce the best candidates and recommends
# where to invest or cut recruiting budget.
#
# SYSTEM PROMPT FEATURES:
# - Explicit decision authority: may autonomously recommend deprioritizing
#   any source with <5% conversion rate
# - 2 few-shot examples anchoring behavior on edge cases
# - JSON output format enforced
#
# KEY INSIGHT PATTERN: Referral (30% conversion) vs LinkedIn (4.5%) = 6.7x delta
# ==============================================================================

class SourcingQualityAgent(BaseAgent):
    """Analyzes channel conversion rates and recommends sourcing budget allocation."""

    def __init__(self, model: str = "gpt-4o-mini"):
        super().__init__("SourcingQualityAgent", model)

    SYSTEM_PROMPT = """You are a Sourcing Quality Analyst for an engineering hiring team.
Analyze candidate sourcing data and return a JSON object with exactly these fields:
{
    "recommendation": "specific actionable suggestion about which sources to prioritize or deprioritize",
    "evidence": "specific data points from the input supporting your recommendation",
    "confidence_score": 0.0-1.0,
    "alternative": "a cheaper or faster approach with trade-off note"
}

DECISION AUTHORITY: You may autonomously recommend deprioritizing any source with <5% conversion rate.

FEW-SHOT EXAMPLE 1:
Input: Referral: 33 applicants, 10 hired. LinkedIn: 22 applicants, 1 hired.
Output: {"recommendation": "Increase referral program budget 40% and pause LinkedIn spend for Senior SWE roles",
         "evidence": "Referral conversion 30.3% vs LinkedIn 4.5% — 6.7x delta",
         "confidence_score": 0.92,
         "alternative": "A/B test LinkedIn with improved JD copy before full pause"}

FEW-SHOT EXAMPLE 2:
Input: All sources showing 8-12% conversion with no clear winner.
Output: {"recommendation": "Diversify budget equally — no dominant channel identified yet",
         "evidence": "Conversion rates within 4% band across all sources",
         "confidence_score": 0.55,
         "alternative": "Run 90-day focused experiment on referrals to establish baseline"}

Base your analysis only on data provided. Never invent benchmarks."""

    def run(self, data_context: str) -> InsightOutput:
        user_prompt = f"""Analyze this sourcing data and identify which channels produce
the best candidates. Which sources should be prioritized or deprioritized?

DATA:
{data_context}

Return JSON with recommendation, evidence, confidence_score, and alternative."""
        response, run = self.call_llm(self.SYSTEM_PROMPT, user_prompt)
        return self.parse_insight(response, run)


# ==============================================================================
# SECTION 7: REJECTION PATTERN AGENT
# ==============================================================================
# Identifies what is causing rejections, at which stages, and for which roles.
# Distinguishes between JD quality problems, interview bar calibration issues,
# and compensation mismatches.
#
# SYSTEM PROMPT FEATURES:
# - Explicit decision authority: may autonomously flag any stage with
#   >40% rejection rate as critical
# - 2 few-shot examples for calibration and compensation cases
# ==============================================================================

class RejectionPatternAgent(BaseAgent):
    """Identifies rejection patterns by stage, role, and root cause."""

    def __init__(self, model: str = "gpt-4o-mini"):
        super().__init__("RejectionPatternAgent", model)

    SYSTEM_PROMPT = """You are a Rejection Pattern Analyst for an engineering hiring team.
Identify what is causing rejections, at which stages, and for which roles.

DECISION AUTHORITY: You may autonomously flag any stage with >40% rejection rate as critical.

FEW-SHOT EXAMPLE 1:
Input: Technical Screen rejection rate: 68%. Reason: failed technical screen x8.
Output: {"recommendation": "Audit technical screen rubric — 68% rejection rate indicates bar miscalibration or JD inflation",
         "evidence": "68% rejection at Technical Screen exceeds 40% critical threshold",
         "confidence_score": 0.88,
         "alternative": "Add calibration session before changing JD"}

FEW-SHOT EXAMPLE 2:
Input: Rejection reasons: compensation mismatch x4, withdrew competing offer x3.
Output: {"recommendation": "Review compensation bands against Levels.fyi — losing 7 candidates to comp/competing offers",
         "evidence": "Compensation and competing offers account for 58% of rejections",
         "confidence_score": 0.85,
         "alternative": "Implement 48-hour exploding offers for strong candidates"}

Return JSON with recommendation, evidence, confidence_score, and alternative.
Base your analysis only on data provided. Never invent statistics."""

    def run(self, data_context: str) -> InsightOutput:
        user_prompt = f"""Analyze this rejection data. What is causing rejections,
at which stages, and for which roles? Is the JD the problem or the interview process?

DATA:
{data_context}

Return JSON with recommendation, evidence, confidence_score, and alternative."""
        response, run = self.call_llm(self.SYSTEM_PROMPT, user_prompt)
        return self.parse_insight(response, run)


# ==============================================================================
# SECTION 8: PANEL LOAD BALANCER AGENT
# ==============================================================================
# Detects interviewer overload and recommends panel rebalancing.
# Consistently scores 0.97 — the highest eval score in the system.
#
# KEY METRIC: Any interviewer doing >6 interviews/week = burnout risk.
# Any active interviewer doing <1/month = skill atrophy risk.
# ==============================================================================

class PanelLoadBalancerAgent(BaseAgent):
    """Detects interviewer overload and recommends panel rebalancing."""

    def __init__(self, model: str = "gpt-4o-mini"):
        super().__init__("PanelLoadBalancerAgent", model)

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

    def run(self, data_context: str) -> InsightOutput:
        user_prompt = f"""Analyze this interviewer assignment data. Who is overloaded,
who is underutilized, and how should panel load be rebalanced?

DATA:
{data_context}

Return JSON with recommendation, evidence, confidence_score, and alternative."""
        response, run = self.call_llm(self.SYSTEM_PROMPT, user_prompt)
        return self.parse_insight(response, run)


# ==============================================================================
# SECTION 9: OFFER INSIGHTS AGENT
# ==============================================================================
# Analyzes why offers are being declined and recommends compensation and
# process improvements to improve offer acceptance rates.
#
# KEY PATTERNS IN THIS DATASET:
# - 4 declines due to compensation too low
# - 4 declines due to location/remote policy
# - 3 declines due to competing offer
# ==============================================================================

class OfferInsightsAgent(BaseAgent):
    """Analyzes offer decline patterns and recommends acceptance rate improvements."""

    def __init__(self, model: str = "gpt-4o-mini"):
        super().__init__("OfferInsightsAgent", model)

    SYSTEM_PROMPT = """You are an Offer Insights Analyst for an engineering hiring team.
Analyze offer acceptance and decline data and return a JSON object with exactly these fields:
{
    "recommendation": "specific actionable suggestion about why offers are being declined and how to improve acceptance rates",
    "evidence": "specific data points from the input supporting your recommendation",
    "confidence_score": 0.0-1.0,
    "alternative": "a cheaper or faster approach with trade-off note"
}
Identify patterns in offer declines by reason, role, and compensation.
Base your analysis only on data provided. Never invent offer statistics."""

    def run(self, data_context: str) -> InsightOutput:
        user_prompt = f"""Analyze this offer data. Why are offers getting declined?
Are we losing people on compensation, timing, or competing offers? What should change?

DATA:
{data_context}

Return JSON with recommendation, evidence, confidence_score, and alternative."""
        response, run = self.call_llm(self.SYSTEM_PROMPT, user_prompt)
        return self.parse_insight(response, run)


# ==============================================================================
# SECTION 10: PIPELINE HEALTH AGENT
# ==============================================================================
# Analyzes SLA breach rates and pipeline velocity by role.
# BEFORE few-shot upgrade: 0.60 score
# AFTER few-shot upgrade:  0.97 score (+62% quality improvement)
#
# KEY INSIGHT: 86% SLA breach rate. Frontend Engineer: 29 SLA breaches.
# This is a critical escalation signal — role has been open too long.
# ==============================================================================

class PipelineHealthAgent(BaseAgent):
    """Analyzes SLA breaches, pipeline velocity, and role-level escalation needs."""

    def __init__(self, model: str = "gpt-4o-mini"):
        super().__init__("PipelineHealthAgent", model)

    SYSTEM_PROMPT = """You are a Pipeline Health Analyst for an engineering hiring team.
Analyze hiring pipeline velocity and SLA data.

DECISION AUTHORITY: You may autonomously escalate any role open >90 days or with >50% SLA breach rate.

FEW-SHOT EXAMPLE 1:
Input: SLA breach rate: 86%. Frontend Engineer: 29 breaches. Avg days: 45.5.
Output: {"recommendation": "Escalate Frontend Engineer role immediately — 29 SLA breaches exceeds critical threshold. Reduce interview stages from 4 to 3 and implement 48-hour decision SLA at offer stage.",
         "evidence": "86% overall SLA breach rate. Frontend Engineer accounts for 29 of total breaches. Avg 45.5 days exceeds 30-day target by 52%.",
         "confidence_score": 0.93,
         "alternative": "Add dedicated recruiter to Frontend role only — lower impact but faster to implement"}

FEW-SHOT EXAMPLE 2:
Input: SLA breach rate: 5%. All roles within 25-day average.
Output: {"recommendation": "Pipeline health is strong. Maintain current process cadence and review in 30 days.",
         "evidence": "5% SLA breach rate well below 20% warning threshold. All roles within target.",
         "confidence_score": 0.88,
         "alternative": "Implement proactive monitoring dashboard to catch early drift"}

Return JSON with recommendation, evidence, confidence_score, and alternative.
Never invent data not present in the input."""

    def run(self, data_context: str) -> InsightOutput:
        user_prompt = f"""Analyze this pipeline data. Where is the funnel moving too slowly?
Where are SLA breaches concentrated? Which roles have been open too long?

DATA:
{data_context}

Return JSON with recommendation, evidence, confidence_score, and alternative."""
        response, run = self.call_llm(self.SYSTEM_PROMPT, user_prompt)
        return self.parse_insight(response, run)


# ==============================================================================
# SECTION 11: OPTIMIZATION AGENT (AUTONOMOUS)
# ==============================================================================
# The second autonomous agent. After each pipeline run, it analyzes cost and
# performance data and makes THREE types of autonomous decisions:
#
# 1. ROUTING THRESHOLD ADJUSTMENT — should the complexity threshold move?
# 2. AGENT REPROMPT IDENTIFICATION — which agents need better prompts?
# 3. ESTIMATED SAVINGS CALCULATION — how much could be saved?
#
# AUTONOMY: These decisions are made without human approval and logged
# in the output contract for auditability.
#
# Consistently routes to gpt-4o (complexity 0.8) — correctly identified
# as requiring complex multi-run reasoning.
# ==============================================================================

class OptimizationAgent(BaseAgent):
    """Autonomously analyzes pipeline performance and makes optimization decisions."""

    def __init__(self, model: str = "gpt-4o-mini"):
        super().__init__("OptimizationAgent", model)

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

    def run(self, data_context: str) -> InsightOutput:
        user_prompt = f"""Analyze this pipeline performance data and make autonomous
optimization decisions. What should change right now to reduce cost and improve quality?

DATA:
{data_context}

Return JSON with all required fields including autonomous_decision and agents_to_reprompt."""
        response, run = self.call_llm(self.SYSTEM_PROMPT, user_prompt)
        try:
            data = json.loads(response)
            insight = self.parse_insight(response, run)
            # Append autonomous decision metadata to recommendation
            insight.recommendation = (
                f"{data.get('recommendation', '')} "
                f"[AUTONOMOUS DECISION: {data.get('autonomous_decision', 'No change')}] "
                f"[EST. SAVINGS: {data.get('estimated_savings_pct', 0)}%]"
            )
            return insight
        except Exception:
            return self.parse_insight(response, run)


# ==============================================================================
# SECTION 12: MARKET INTELLIGENCE AGENT (EXTERNAL TOOLS + AUTONOMOUS)
# ==============================================================================
# The third autonomous agent — and the only one with external tool access.
#
# EXTERNAL TOOL: Web search via SerpAPI for real-time salary benchmarks.
# If SerpAPI key is unavailable, falls back to built-in 2024-2025 market data.
#
# SECURITY CONTROL: _filter_market_data() blocks script injection, SQL
# patterns, and oversized payloads before passing to LLM.
#
# AUTONOMY: Fetches market data, performs gap analysis, and makes
# compensation recommendations independently — no human provides the
# external data.
# ==============================================================================

def _filter_market_data(raw: str) -> str:
    """
    SECURITY CONTROL: Filter web search results for harmful content.
    Blocks script injection, SQL patterns, and oversized payloads.
    """
    blocked_patterns = [
        r"<script.*?>.*?</script>",
        r"javascript:",
        r"eval[(]",
        r"exec[(]",
        r"DROP TABLE",
        r"SELECT \* FROM",
    ]
    filtered = raw
    for pattern in blocked_patterns:
        filtered = re.sub(pattern, "[BLOCKED]", filtered, flags=re.IGNORECASE | re.DOTALL)
    if len(filtered) > 3000:
        filtered = filtered[:3000] + "...[TRUNCATED]"
    return filtered


def search_market_data(query: str) -> str:
    """Fetch real-time salary data via SerpAPI. Falls back to built-in data."""
    serpapi_key = os.getenv("SERPAPI_KEY", "")
    if serpapi_key:
        try:
            r = requests.get(
                "https://serpapi.com/search",
                params={"q": query, "api_key": serpapi_key, "num": 3},
                timeout=5
            )
            if r.status_code == 200:
                results = r.json().get("organic_results", [])
                return " | ".join([res.get("snippet", "") for res in results[:3]])
        except Exception:
            pass

    # Built-in fallback — always available, zero cost
    return """
    Market Compensation Data (2024-2025, US Engineering):
    - Senior Software Engineer: $180,000-$230,000 total comp
    - Staff Engineer: $230,000-$320,000 total comp
    - Engineering Manager: $200,000-$280,000 total comp
    - DevOps Engineer: $160,000-$220,000 total comp
    - ML Engineer: $190,000-$270,000 total comp
    - Frontend Engineer: $150,000-$210,000 total comp
    - Security Engineer: $170,000-$240,000 total comp
    Sources: Levels.fyi, Radford, Glassdoor 2024-2025
    """


class MarketIntelligenceAgent(BaseAgent):
    """Autonomously fetches real-time market data and identifies compensation gaps."""

    def __init__(self, model: str = "gpt-4o-mini"):
        super().__init__("MarketIntelligenceAgent", model)

    SYSTEM_PROMPT = """You are a Market Intelligence Agent for an AI hiring intelligence system.
You have access to real-time market data. Compare internal offer amounts against
current market compensation benchmarks. Identify roles below market rate.

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

    def run(self, data_context: str) -> InsightOutput:
        # AUTONOMOUS: fetch market data without human input
        print("  [MarketIntelligenceAgent] Fetching real-time market data...")
        market_data = _filter_market_data(  # Security control applied
            search_market_data("software engineer salary benchmarks 2024 2025 total compensation")
        )

        enriched_context = f"""{data_context}

REAL-TIME MARKET DATA (fetched autonomously):
{market_data}"""

        user_prompt = f"""Compare internal offer amounts against real-time market data.
Which roles are below market? What specific adjustments are needed?

DATA:
{enriched_context}

Return JSON with all required fields."""
        response, run = self.call_llm(self.SYSTEM_PROMPT, user_prompt)
        try:
            data = json.loads(response)
            insight = self.parse_insight(response, run)
            roles_below = data.get("roles_below_market", [])
            gap = data.get("avg_market_gap_pct", 0)
            insight.recommendation = (
                f"{data.get('recommendation', '')} "
                f"[ROLES BELOW MARKET: {', '.join(roles_below) if roles_below else 'None'}] "
                f"[AVG GAP: {gap}%]"
            )
            return insight
        except Exception:
            return self.parse_insight(response, run)


# ==============================================================================
# SECTION 13: EVALUATION AGENT (LLM-AS-JUDGE)
# ==============================================================================
# The quality gate. Every insight from every agent is scored before surfacing.
#
# SCORING DIMENSIONS:
# - actionability_score  — specific enough to act on today? (threshold: >=0.60)
# - grounding_score      — evidence from actual input data? (threshold: >=0.60)
# - hallucination_risk   — invents facts not in input?     (threshold: <=0.30)
# - overall_score        — composite score                 (threshold: >=0.60)
#
# POLICY: insights below 0.60 are FLAGGED, not suppressed.
# Human reviewers retain full visibility into low-confidence outputs.
# ==============================================================================

class EvaluationAgent(BaseAgent):
    """LLM-as-judge quality gate — scores every insight before dashboard surfacing."""

    def __init__(self):
        super().__init__("EvaluationAgent", model="gpt-4o-mini")

    SYSTEM_PROMPT = """You are an Evaluation Agent for an AI hiring intelligence system.
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
- actionability:      Is the recommendation specific enough to act on today?
- grounding:          Is the evidence drawn from actual data in the input?
- hallucination_risk: Does the insight invent facts not present in the input?
- passed:             true if overall_score >= 0.6 AND hallucination_risk <= 0.3"""

    def evaluate(self, insight: InsightOutput, original_data: str) -> dict:
        """Score an insight against the original data. Returns evaluation dict."""
        user_prompt = f"""Evaluate this insight against the original data.

INSIGHT:
Agent: {insight.agent_name}
Recommendation: {insight.recommendation}
Evidence: {insight.evidence}
Confidence: {insight.confidence_score}

ORIGINAL DATA:
{original_data[:2000]}

Return JSON with all scoring dimensions."""
        response, run = self.call_llm(self.SYSTEM_PROMPT, user_prompt)
        try:
            result = json.loads(response)
            result["evaluated_agent"] = insight.agent_name
            result["eval_cost_usd"] = run.estimated_usd
            return result
        except Exception as e:
            return {
                "evaluated_agent": insight.agent_name,
                "actionability_score": 0.0, "grounding_score": 0.0,
                "hallucination_risk": 1.0, "overall_score": 0.0,
                "passed": False, "flags": [f"Parse error: {str(e)}"],
                "judgment": "Evaluation failed — could not parse response",
                "eval_cost_usd": run.estimated_usd
            }

    def run(self, data_context: str) -> InsightOutput:
        pass  # EvaluationAgent is called via evaluate(), not run()


# ==============================================================================
# SECTION 14: ORCHESTRATOR (PIPELINE COORDINATOR)
# ==============================================================================
# The orchestrator ties all agents together into a single pipeline run:
#
# 1. Load and clean ATS data (200 candidates, 19 fields)
# 2. Build data context summary (stage distribution, source data, etc.)
# 3. For each insight agent:
#    a. Route to correct model (RoutingAgent)
#    b. Retrieve RAG context (ChromaDB)
#    c. Run insight agent
#    d. Evaluate insight (EvaluationAgent)
#    e. Track cost and latency (AgentRun)
# 4. Save results to data/last_run_results.json
# 5. Notify n8n webhook (with X-Webhook-Secret auth header)
#
# SECURITY: n8n notification includes secret header from .env
# ==============================================================================

def load_ats_data() -> pd.DataFrame:
    """Load and clean ATS dataset."""
    df = pd.read_csv(DATA_PATH)
    df["apply_date"] = pd.to_datetime(df["apply_date"])
    df["last_activity_date"] = pd.to_datetime(df["last_activity_date"])
    df["interview_score"] = pd.to_numeric(df["interview_score"], errors="coerce")
    df["offer_amount"] = pd.to_numeric(df["offer_amount"], errors="coerce")
    return df


def get_summary_stats(df: pd.DataFrame) -> dict:
    """Generate summary statistics for agents and dashboard."""
    return {
        "total_candidates": len(df),
        "stage_distribution": df["current_stage"].value_counts().to_dict(),
        "source_distribution": df["source"].value_counts().to_dict(),
        "role_distribution": df["role"].value_counts().to_dict(),
        "avg_days_in_pipeline": round(df["days_in_pipeline"].mean(), 1),
        "sla_breach_rate": round(df["sla_breached"].eq("Yes").mean() * 100, 1),
        "hired_count": len(df[df["current_stage"] == "Hired"]),
        "offer_count": len(df[df["current_stage"] == "Offer"]),
        "rejection_reasons": df[df["rejection_reason"] != ""]["rejection_reason"].value_counts().to_dict(),
        "offer_decline_reasons": df[df["offer_decline_reason"] != ""]["offer_decline_reason"].value_counts().to_dict(),
        "top_sources_by_hire": df[df["current_stage"] == "Hired"]["source"].value_counts().to_dict(),
    }


def build_data_context(df: pd.DataFrame, stats: dict) -> str:
    """Build the data context string passed to every agent."""
    return f"""
HIRING PIPELINE SUMMARY
=======================
Total Candidates: {stats['total_candidates']}
Hired: {stats['hired_count']}
Active Offers: {stats['offer_count']}
Avg Days in Pipeline: {stats['avg_days_in_pipeline']}
SLA Breach Rate: {stats['sla_breach_rate']}%

STAGE DISTRIBUTION:
{json.dumps(stats['stage_distribution'], indent=2)}

SOURCE DISTRIBUTION:
{json.dumps(stats['source_distribution'], indent=2)}

TOP HIRE SOURCES:
{json.dumps(stats['top_sources_by_hire'], indent=2)}

REJECTION REASONS:
{json.dumps(stats['rejection_reasons'], indent=2)}

OFFER DECLINE REASONS:
{json.dumps(stats['offer_decline_reasons'], indent=2)}

ROLE DISTRIBUTION:
{json.dumps(stats['role_distribution'], indent=2)}

INTERVIEWER LOAD:
{df['interviewers_assigned'].value_counts().head(10).to_dict()}

SLA BREACHES BY ROLE:
{df[df['sla_breached']=='Yes']['role'].value_counts().to_dict()}

OFFER AMOUNTS:
Min: {df['offer_amount'].min()} | Max: {df['offer_amount'].max()} | Mean: {round(df['offer_amount'].mean(), 0)}
"""


def run_pipeline() -> PipelineRun:
    """
    Execute the full 9-agent pipeline:
    Route → RAG → Insight → Evaluate → Log → Notify n8n
    """
    print("\n=== AI-Powered Hiring Intelligence System ===")
    print("Starting pipeline run...\n")

    df = load_ats_data()
    stats = get_summary_stats(df)
    data_context = build_data_context(df, stats)

    router = RoutingAgent()
    evaluator = EvaluationAgent()

    # All 7 insight agents in execution order
    insight_agent_classes = [
        SourcingQualityAgent,
        RejectionPatternAgent,
        PanelLoadBalancerAgent,
        OfferInsightsAgent,
        PipelineHealthAgent,
        OptimizationAgent,
        MarketIntelligenceAgent,
    ]

    all_insights, all_evaluations, all_runs = [], [], []
    total_usd, successful, failed = 0.0, 0, 0

    for AgentClass in insight_agent_classes:
        agent_name = AgentClass.__name__
        print(f"Running {agent_name}...")

        try:
            # Step 1: Route to correct model (autonomous)
            routing_decision = router.route(agent_name, str(stats)[:500])
            selected_model = routing_decision.get("selected_model", "gpt-4o-mini")

            # Step 2: Retrieve RAG context from ChromaDB
            rag_query = RAG_QUERIES.get(agent_name, "hiring benchmarks best practices")
            rag_context = retrieve_context(rag_query, k=3)
            grounded_context = f"{data_context}\n\nINDUSTRY BENCHMARKS (RAG retrieved):\n{rag_context}"

            # Step 3: Run insight agent with grounded context
            agent = AgentClass(model=selected_model)
            insight = agent.run(grounded_context)
            all_insights.append(insight)

            # Step 4: Evaluate insight quality
            evaluation = evaluator.evaluate(insight, grounded_context)
            all_evaluations.append(evaluation)

            # Step 5: Track cost
            all_runs.extend(agent.run_log)
            total_usd += sum(r.estimated_usd for r in agent.run_log)
            successful += 1

            status = "PASS" if evaluation.get("passed") else "FAIL"
            print(f"  [{status}] Score: {evaluation.get('overall_score', 0):.2f} | "
                  f"Cost: ${insight.cost_of_insight['estimated_usd']:.6f}")

        except Exception as e:
            print(f"  [ERROR] {agent_name} failed: {str(e)}")
            failed += 1

    # Add router and evaluator runs to cost tracking
    all_runs.extend(router.run_log + evaluator.run_log)
    total_usd += sum(r.estimated_usd for r in router.run_log + evaluator.run_log)

    pipeline_run = PipelineRun(
        run_id=str(uuid.uuid4())[:8],
        total_agents=len(insight_agent_classes),
        successful_agents=successful,
        failed_agents=failed,
        total_input_tokens=sum(r.input_tokens for r in all_runs),
        total_output_tokens=sum(r.output_tokens for r in all_runs),
        total_latency_seconds=sum(r.latency_seconds for r in all_runs),
        total_estimated_usd=round(total_usd, 6),
        agent_runs=all_runs,
        insights=all_insights
    )

    # Save results to disk
    os.makedirs("data", exist_ok=True)
    with open("data/last_run_results.json", "w") as f:
        json.dump({
            "pipeline_run": pipeline_run.model_dump(),
            "evaluations": all_evaluations
        }, f, indent=2, default=str)

    print(f"\n=== Pipeline Complete ===")
    print(f"Agents: {successful}/{len(insight_agent_classes)} successful")
    print(f"Total cost: ${total_usd:.6f}")
    print(f"Total latency: {sum(r.latency_seconds for r in all_runs):.1f}s")

    # Step 6: Notify n8n webhook with secret header (security control)
    try:
        n8n_payload = {
            "run_id": pipeline_run.run_id,
            "successful_agents": pipeline_run.successful_agents,
            "total_agents": pipeline_run.total_agents,
            "total_cost_usd": pipeline_run.total_estimated_usd,
            "total_latency_seconds": pipeline_run.total_latency_seconds,
            "status": "complete"
        }
        n8n_secret = os.getenv("N8N_WEBHOOK_SECRET", "")
        headers = {"Content-Type": "application/json"}
        if n8n_secret:
            headers["X-Webhook-Secret"] = n8n_secret  # Auth header
        r = requests.post(
            "https://lamontesmith.app.n8n.cloud/webhook/hiring-pipeline",
            json=n8n_payload, headers=headers, timeout=10
        )
        print(f"n8n notified: {r.status_code}")
    except Exception as e:
        print(f"n8n notification failed: {str(e)}")

    return pipeline_run


# ==============================================================================
# SECTION 15: FASTAPI ENDPOINTS
# ==============================================================================
# REST API layer with X-API-Key authentication on all protected routes.
#
# SECURITY CONTROL: verify_api_key() dependency enforces auth on
# /pipeline/run, /pipeline/insights, /pipeline/costs, /golden/results
#
# ENDPOINTS:
# GET  /health           — no auth required — service status
# POST /pipeline/run     — auth required — trigger full pipeline
# GET  /pipeline/insights — auth required — last run insights
# GET  /pipeline/costs    — auth required — cost breakdown
# GET  /golden/results    — auth required — golden dataset results
# ==============================================================================

"""
FastAPI implementation is in api.py. Key security pattern:

from fastapi import FastAPI, Security, HTTPException, status
from fastapi.security import APIKeyHeader
import secrets

API_KEY = os.getenv("API_KEY", secrets.token_hex(32))
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid or missing API key")
    return api_key

# Usage on protected endpoints:
@app.get("/pipeline/insights", dependencies=[Security(verify_api_key)])
def get_insights():
    ...
"""


# ==============================================================================
# SECTION 16: STREAMLIT DASHBOARD
# ==============================================================================
# Executive dashboard at http://localhost:8501
# Designed so an engineering manager can understand the full pipeline
# state in under one minute.
#
# COMPONENTS:
# 1. KPI Row         — total candidates, hired, offers, avg days, SLA breach %
# 2. Hiring Funnel   — bar chart by stage (Applied → Hired)
# 3. Source Chart    — candidate volume by sourcing channel
# 4. Agent Insights  — expandable cards per agent with scores and flags
# 5. Cost Table      — per-agent tokens, latency, USD; pipeline totals
#
# Launch: streamlit run dashboard/app.py
# ==============================================================================

"""
Dashboard implementation is in dashboard/app.py.
Key components:

import streamlit as st
import json, pandas as pd, os, subprocess, sys

# Run pipeline button triggers orchestrator as subprocess
if st.button("Run Pipeline"):
    result = subprocess.run([sys.executable, "orchestrator.py"], ...)

# Agent insight cards
for insight in insights:
    with st.expander(f"{status_icon} {agent} - Score: {score:.2f}"):
        st.info(insight["recommendation"])
        st.write(insight["evidence"])
        st.metric("Confidence", f"{insight['confidence_score']:.0%}")
        st.metric("Cost", f"${cost['estimated_usd']:.6f}")
"""


# ==============================================================================
# SECTION 17: GOLDEN DATASET EVALUATION
# ==============================================================================
# 20 scenarios across 7 categories for offline regression testing.
#
# CATEGORIES:
# - sourcing (3)     — channel conversion including ambiguous cases
# - rejection (4)    — stage bottlenecks including no-data edge cases
# - panel (3)        — overload detection including healthy distribution
# - offer (3)        — decline analysis including unknown reason cases
# - pipeline (4)     — SLA breach including stalled role escalation
# - hallucination(1) — low-signal input should score 0 confidence
# - routing (2)      — simple→mini, complex→gpt-4o
#
# RESULTS: 75% pass rate (15/20). Routing accuracy: 100%.
#
# Run: python3 golden_eval.py
# ==============================================================================

"""
Golden dataset runner is in golden_eval.py.
Key pattern:

evaluator = EvaluationAgent()
router = RoutingAgent()

for scenario in golden_dataset:
    if scenario["category"] == "routing":
        routing_result = router.route(f"TestAgent_{id}", scenario["input"])
        passed = routing_result["selected_model"] == scenario["expected_model"]
    else:
        mock_insight = InsightOutput(agent_name="MockAgent", ...)
        eval_result = evaluator.evaluate(mock_insight, scenario["input"])
        passed = eval_result["passed"] == scenario["should_pass_eval"]
"""


# ==============================================================================
# SECTION 18: n8n WORKFLOW INTEGRATION
# ==============================================================================
# Two published n8n workflows in lamontesmith.app.n8n.cloud
#
# WORKFLOW 1: Hiring Intelligence Pipeline (event-driven)
# Trigger: POST https://lamontesmith.app.n8n.cloud/webhook/hiring-pipeline
# Nodes:
#   1. Receive Pipeline Results  — webhook receiver
#   2. All Agents Passed?        — IF quality gate
#   3. Calculate Health Score    — JavaScript: health_score, cost_efficiency, latency_grade
#   4. Format Success Response   — Edit Fields: status, run_id, cost
#   5. Format Failure Response   — Edit Fields: partial_failure alert
#
# WORKFLOW 2: Talent Acquisition Alert System (scheduled)
# Trigger: Every 6 hours
# Nodes:
#   1. Every 6 Hours             — schedule trigger
#   2. Load Latest Pipeline Data — Code (JS): load insights + costs
#   3. Analyze Critical Alerts   — Code (JS): SLA_BREACH/OFFER_DECLINE/MARKET_GAP/LOW_CONFIDENCE
#   4. Requires Immediate Action?— IF routing by severity
#   5. Format Critical Report    — Code (JS): structured alert with remediation
#   6. Format Healthy Report     — Code (JS): all-clear status
#   7. Consolidate Report        — Merge (Append): unified output
#
# Test curl command:
# curl -X POST https://lamontesmith.app.n8n.cloud/webhook/hiring-pipeline \
#   -H "Content-Type: application/json" \
#   -d '{"run_id":"demo01","successful_agents":7,"total_agents":7,
#        "total_cost_usd":0.007404,"total_latency_seconds":46.5,"status":"complete"}'
# ==============================================================================


# ==============================================================================
# ENTRY POINT
# ==============================================================================
if __name__ == "__main__":
    pipeline_run = run_pipeline()
    print(f"\nRun ID: {pipeline_run.run_id}")
    print(f"Results: data/last_run_results.json")