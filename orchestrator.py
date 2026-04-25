import os
import uuid
import json
from dotenv import load_dotenv
from agents.data_loader import load_ats_data, get_summary_stats
from agents.contracts import PipelineRun
from agents.routing_agent import RoutingAgent
from agents.evaluation_agent import EvaluationAgent
from agents.sourcing_quality_agent import SourcingQualityAgent
from agents.rejection_pattern_agent import RejectionPatternAgent
from agents.panel_load_balancer_agent import PanelLoadBalancerAgent
from agents.offer_insights_agent import OfferInsightsAgent
from agents.pipeline_health_agent import PipelineHealthAgent
from agents.optimization_agent import OptimizationAgent
from agents.market_intelligence_agent import MarketIntelligenceAgent

load_dotenv()


def build_data_context(df, stats) -> str:
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

INTERVIEWER LOAD (top assignments):
{df['interviewers_assigned'].value_counts().head(10).to_dict()}

SLA BREACHES BY ROLE:
{df[df['sla_breached']=='Yes']['role'].value_counts().to_dict()}

OFFER AMOUNTS (where available):
Min: {df['offer_amount'].min()}
Max: {df['offer_amount'].max()}
Mean: {round(df['offer_amount'].mean(), 0)}
"""


def run_pipeline() -> PipelineRun:
    print("\n=== AI-Powered Hiring Intelligence System ===")
    print("Starting pipeline run...\n")

    df = load_ats_data()
    stats = get_summary_stats(df)
    data_context = build_data_context(df, stats)

    router = RoutingAgent()
    evaluator = EvaluationAgent()

    insight_agents = [
        SourcingQualityAgent,
        RejectionPatternAgent,
        PanelLoadBalancerAgent,
        OfferInsightsAgent,
        PipelineHealthAgent,
        OptimizationAgent,
        MarketIntelligenceAgent,
    ]

    all_insights = []
    all_evaluations = []
    all_runs = []
    total_usd = 0.0
    successful = 0
    failed = 0

    for AgentClass in insight_agents:
        agent_name = AgentClass.__name__
        print(f"Running {agent_name}...")

        try:
            # Route to correct model
            routing_decision = router.route(agent_name, str(stats)[:500])
            selected_model = routing_decision.get("selected_model", "gpt-4o-mini")

            # Run agent with routed model
            agent = AgentClass(model=selected_model)
            insight = agent.run(data_context)
            all_insights.append(insight)

            # Evaluate the insight
            evaluation = evaluator.evaluate(insight, data_context)
            all_evaluations.append(evaluation)

            # Track runs
            all_runs.extend(agent.run_log)
            total_usd += sum(r.estimated_usd for r in agent.run_log)
            successful += 1

            status = "PASS" if evaluation.get("passed") else "FAIL"
            print(f"  [{status}] Score: {evaluation.get('overall_score', 0):.2f} | Cost: ${insight.cost_of_insight['estimated_usd']:.6f}")

        except Exception as e:
            print(f"  [ERROR] {agent_name} failed: {str(e)}")
            failed += 1

    # Add router and evaluator runs
    all_runs.extend(router.run_log)
    all_runs.extend(evaluator.run_log)
    total_usd += sum(r.estimated_usd for r in router.run_log)
    total_usd += sum(r.estimated_usd for r in evaluator.run_log)

    pipeline_run = PipelineRun(
        run_id=str(uuid.uuid4())[:8],
        total_agents=len(insight_agents),
        successful_agents=successful,
        failed_agents=failed,
        total_input_tokens=sum(r.input_tokens for r in all_runs),
        total_output_tokens=sum(r.output_tokens for r in all_runs),
        total_latency_seconds=sum(r.latency_seconds for r in all_runs),
        total_estimated_usd=round(total_usd, 6),
        agent_runs=all_runs,
        insights=all_insights
    )

    # Save results
    results = {
        "pipeline_run": pipeline_run.model_dump(),
        "evaluations": all_evaluations
    }
    with open("data/last_run_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n=== Pipeline Complete ===")
    print(f"Agents: {successful}/{len(insight_agents)} successful")
    print(f"Total cost: ${total_usd:.6f}")
    print(f"Total latency: {sum(r.latency_seconds for r in all_runs):.1f}s")
    print(f"Results saved to data/last_run_results.json")

    
    # Notify n8n webhook
    import requests
    try:
        n8n_payload = {
            'run_id': pipeline_run.run_id,
            'successful_agents': pipeline_run.successful_agents,
            'total_agents': pipeline_run.total_agents,
            'total_cost_usd': pipeline_run.total_estimated_usd,
            'total_latency_seconds': pipeline_run.total_latency_seconds,
            'status': 'complete'
        }
        r = requests.post(
            'https://lamontesmith.app.n8n.cloud/webhook/hiring-pipeline',
            json=n8n_payload,
            timeout=10
        )
        print(f'n8n notified: {r.status_code}')
    except Exception as e:
        print(f'n8n notification failed: {str(e)}')

    
    # Notify n8n webhook
    import requests
    try:
        n8n_payload = {
            'run_id': pipeline_run.run_id,
            'successful_agents': pipeline_run.successful_agents,
            'total_agents': pipeline_run.total_agents,
            'total_cost_usd': pipeline_run.total_estimated_usd,
            'total_latency_seconds': pipeline_run.total_latency_seconds,
            'status': 'complete'
        }
        r = requests.post(
            'https://lamontesmith.app.n8n.cloud/webhook/hiring-pipeline',
            json=n8n_payload,
            timeout=10
        )
        print(f'n8n notified: {r.status_code}')
    except Exception as e:
        print(f'n8n notification failed: {str(e)}')

    return pipeline_run


if __name__ == "__main__":
    run_pipeline()
