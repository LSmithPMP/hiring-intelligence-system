# AI-Powered Hiring Intelligence System

**Capstone 2 | Interview Kickstart Applied Agentic AI | Lamonte Smith | April 2026**

Multi-agent pipeline for engineering talent acquisition insights.

## Tech Stack
LangChain · n8n Cloud · FastAPI · ChromaDB · Streamlit · OpenAI

## Agents (9 Total)
### Supporting Agents
- RoutingAgent — autonomous model selection per task complexity
- EvaluationAgent — LLM-as-judge scoring (actionability, grounding, hallucination)
- OptimizationAgent — autonomous routing threshold decisions and cost analysis

### Insight Agents
- SourcingQualityAgent — channel conversion rates and cost per hire
- RejectionPatternAgent — stage bottlenecks and JD mismatch patterns
- PanelLoadBalancerAgent — interviewer overload detection
- OfferInsightsAgent — offer decline analysis and compensation gaps
- PipelineHealthAgent — SLA breach analysis and velocity metrics
- MarketIntelligenceAgent — external web search tool, real-time comp benchmarks

## n8n Workflows
- **Workflow 1:** Hiring Intelligence Pipeline — webhook-triggered, 5 nodes
- **Workflow 2:** Talent Acquisition Alert System — scheduled every 6 hours, 7 nodes

## Results
- 7/7 agents passing per run
- Eval scores: 0.60-0.97
- Cost per run: $0.007404
- Golden dataset: 75% pass rate (15/20)

## GitHub
github.com/LSmithPMP
