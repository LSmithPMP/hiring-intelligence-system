# AI-Powered Hiring Intelligence System
## Capstone 2 - Interview Kickstart Applied Agentic AI
**Author:** Lamonte Smith | April 2026

## Overview
Multi-agent AI system analyzing engineering hiring pipeline data.
Built with LangChain, n8n, FastAPI, ChromaDB, and Streamlit.

## Agents
- RoutingAgent: model selection per task complexity
- EvaluationAgent: scores insights on actionability and grounding
- SourcingQualityAgent: channel conversion rates
- RejectionPatternAgent: stage bottlenecks and JD mismatch
- PanelLoadBalancerAgent: interviewer overload detection
- OfferInsightsAgent: offer decline and compensation gaps
- PipelineHealthAgent: SLA breach and velocity metrics

## Results
- Agents: 5/5 passing per run
- Eval scores: 0.60-0.97
- Cost per run: 0.002-0.007 USD
- Golden dataset pass rate: 75% (15/20)
- n8n notifications: 200 OK
