from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class InsightOutput(BaseModel):
    """Shared output contract for all insight agents."""
    
    agent_name: str = Field(description="Name of the agent that produced this insight")
    recommendation: str = Field(description="Specific, actionable suggestion — not a generic observation")
    evidence: str = Field(description="Data points from the input that support the recommendation")
    confidence_score: float = Field(ge=0.0, le=1.0, description="Confidence in the recommendation, 0 to 1")
    cost_of_insight: dict = Field(description="Tokens used, model name, and rough USD estimate")
    alternative: str = Field(description="A cheaper or faster option with trade-off note")
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    
    class Config:
        json_schema_extra = {
            "example": {
                "agent_name": "SourcingQualityAgent",
                "recommendation": "Deprioritize Indeed for Senior SWE roles — only 8% conversion vs 31% for Referrals",
                "evidence": "Indeed: 25 applicants, 2 hired. Referral: 33 applicants, 10 hired.",
                "confidence_score": 0.87,
                "cost_of_insight": {
                    "model": "gpt-4o-mini",
                    "input_tokens": 420,
                    "output_tokens": 85,
                    "estimated_usd": 0.0001
                },
                "alternative": "Use gpt-3.5-turbo for sourcing analysis — 90% quality at 10x lower cost",
                "timestamp": "2026-04-08T03:00:00"
            }
        }


class AgentRun(BaseModel):
    """Tracks a full agent execution including cost and latency."""
    
    agent_name: str
    model_used: str
    input_tokens: int
    output_tokens: int
    latency_seconds: float
    estimated_usd: float
    success: bool
    error_message: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class PipelineRun(BaseModel):
    """Tracks a full pipeline execution across all agents."""
    
    run_id: str
    total_agents: int
    successful_agents: int
    failed_agents: int
    total_input_tokens: int
    total_output_tokens: int
    total_latency_seconds: float
    total_estimated_usd: float
    agent_runs: list[AgentRun]
    insights: list[InsightOutput]
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
