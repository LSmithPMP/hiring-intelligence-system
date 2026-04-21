from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
import os
import sys
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

app = FastAPI(title="Hiring Intelligence API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class PipelineRequest(BaseModel):
    trigger: str = "manual"
    filters: dict = {}


@app.get("/health")
def health():
    return {"status": "ok", "service": "Hiring Intelligence API"}


@app.post("/pipeline/run")
def run_pipeline_endpoint(request: PipelineRequest):
    from orchestrator import run_pipeline
    pipeline_run = run_pipeline()
    return {
        "status": "complete",
        "run_id": pipeline_run.run_id,
        "successful_agents": pipeline_run.successful_agents,
        "total_agents": pipeline_run.total_agents,
        "total_cost_usd": pipeline_run.total_estimated_usd,
        "total_latency_seconds": pipeline_run.total_latency_seconds,
    }


@app.get("/pipeline/results")
def get_results():
    results_path = "data/last_run_results.json"
    if not os.path.exists(results_path):
        return {"error": "No results available. Run pipeline first."}
    with open(results_path) as f:
        return json.load(f)


@app.get("/pipeline/insights")
def get_insights():
    results_path = "data/last_run_results.json"
    if not os.path.exists(results_path):
        return {"error": "No results available. Run pipeline first."}
    with open(results_path) as f:
        results = json.load(f)
    return {
        "insights": results["pipeline_run"]["insights"],
        "evaluations": results["evaluations"]
    }


@app.get("/pipeline/costs")
def get_costs():
    results_path = "data/last_run_results.json"
    if not os.path.exists(results_path):
        return {"error": "No results available. Run pipeline first."}
    with open(results_path) as f:
        results = json.load(f)
    pr = results["pipeline_run"]
    return {
        "total_cost_usd": pr["total_estimated_usd"],
        "total_input_tokens": pr["total_input_tokens"],
        "total_output_tokens": pr["total_output_tokens"],
        "total_latency_seconds": pr["total_latency_seconds"],
        "agent_runs": pr["agent_runs"]
    }


@app.get("/golden/results")
def get_golden_results():
    golden_path = "data/golden_dataset_results.json"
    if not os.path.exists(golden_path):
        return {"error": "No golden dataset results. Run golden_eval.py first."}
    with open(golden_path) as f:
        results = json.load(f)
    passed = sum(1 for r in results if r.get("scenario_passed") or r.get("passed"))
    return {
        "total_scenarios": len(results),
        "passed": passed,
        "failed": len(results) - passed,
        "pass_rate": round(passed / len(results) * 100, 1),
        "results": results
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
