import json
import os
from dotenv import load_dotenv
from agents.evaluation_agent import EvaluationAgent
from agents.routing_agent import RoutingAgent
from agents.contracts import InsightOutput

load_dotenv()


def run_golden_dataset_eval():
    print("\n=== Golden Dataset Evaluation ===")
    
    with open('data/golden_dataset.json') as f:
        scenarios = json.load(f)
    
    evaluator = EvaluationAgent()
    router = RoutingAgent()
    
    results = []
    passed = 0
    failed = 0
    
    for scenario in scenarios:
        scenario_id = scenario['scenario_id']
        category = scenario['category']
        
        # Skip routing scenarios - tested separately
        if category == 'routing':
            routing_result = router.route(
                f"TestAgent_{scenario_id}",
                scenario['input']
            )
            expected_model = scenario.get('expected_model', 'gpt-4o-mini')
            actual_model = routing_result.get('selected_model', 'gpt-4o-mini')
            complexity = routing_result.get('complexity_score', 0.5)
            
            route_passed = actual_model == expected_model
            results.append({
                "scenario_id": scenario_id,
                "category": category,
                "passed": route_passed,
                "expected_model": expected_model,
                "actual_model": actual_model,
                "complexity_score": complexity,
                "notes": scenario.get('notes', '')
            })
            status = "PASS" if route_passed else "FAIL"
            print(f"[{status}] {scenario_id} ({category}): routed to {actual_model} (expected {expected_model})")
            if route_passed:
                passed += 1
            else:
                failed += 1
            continue
        
        # For insight scenarios, create a mock insight and evaluate it
        mock_insight = InsightOutput(
            agent_name=f"MockAgent_{category}",
            recommendation=f"Mock recommendation based on: {scenario['input'][:100]}",
            evidence=scenario['input'],
            confidence_score=scenario.get('expected_confidence_min', 0.5),
            cost_of_insight={
                "model": "gpt-4o-mini",
                "input_tokens": 100,
                "output_tokens": 50,
                "estimated_usd": 0.0001
            },
            alternative="Alternative approach for cost savings"
        )
        
        eval_result = evaluator.evaluate(mock_insight, scenario['input'])
        
        expected_pass = scenario.get('should_pass_eval', True)
        actual_pass = eval_result.get('passed', False)
        overall_score = eval_result.get('overall_score', 0)
        
        # Check keyword presence in judgment
        judgment = eval_result.get('judgment', '').lower()
        flags = eval_result.get('flags', [])
        
        scenario_passed = (actual_pass == expected_pass)
        
        results.append({
            "scenario_id": scenario_id,
            "category": category,
            "expected_pass": expected_pass,
            "actual_pass": actual_pass,
            "overall_score": overall_score,
            "scenario_passed": scenario_passed,
            "judgment": eval_result.get('judgment', ''),
            "flags": flags,
            "notes": scenario.get('notes', '')
        })
        
        status = "PASS" if scenario_passed else "FAIL"
        print(f"[{status}] {scenario_id} ({category}): score={overall_score:.2f}, passed={actual_pass}")
        
        if scenario_passed:
            passed += 1
        else:
            failed += 1
    
    # Save results
    with open('data/golden_dataset_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n=== Golden Dataset Results ===")
    print(f"Total scenarios: {len(scenarios)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Pass rate: {passed/len(scenarios)*100:.1f}%")
    print(f"Results saved to data/golden_dataset_results.json")
    
    return results


if __name__ == "__main__":
    run_golden_dataset_eval()
