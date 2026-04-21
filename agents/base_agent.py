import os
import time
import json
from dotenv import load_dotenv
from openai import OpenAI
from agents.contracts import InsightOutput, AgentRun

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

MODEL_PRICING = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
}


def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    pricing = MODEL_PRICING.get(model, MODEL_PRICING["gpt-4o-mini"])
    input_cost = (input_tokens / 1_000_000) * pricing["input"]
    output_cost = (output_tokens / 1_000_000) * pricing["output"]
    return round(input_cost + output_cost, 6)


class BaseAgent:
    def __init__(self, agent_name: str, model: str = "gpt-4o-mini"):
        self.agent_name = agent_name
        self.model = model
        self.run_log: list[AgentRun] = []

    def call_llm(self, system_prompt: str, user_prompt: str) -> tuple[str, AgentRun]:
        start = time.time()
        success = True
        error_message = None
        response_text = ""
        input_tokens = 0
        output_tokens = 0

        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            response_text = response.choices[0].message.content
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens

        except Exception as e:
            success = False
            error_message = str(e)
            response_text = "{}"

        latency = round(time.time() - start, 2)
        estimated_usd = calculate_cost(self.model, input_tokens, output_tokens)

        run = AgentRun(
            agent_name=self.agent_name,
            model_used=self.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_seconds=latency,
            estimated_usd=estimated_usd,
            success=success,
            error_message=error_message
        )
        self.run_log.append(run)
        return response_text, run

    def _to_string(self, value) -> str:
        if isinstance(value, str):
            return value
        elif isinstance(value, dict):
            return json.dumps(value)
        elif isinstance(value, list):
            return "; ".join(str(v) for v in value)
        return str(value)

    def parse_insight(self, raw_response: str, run: AgentRun) -> InsightOutput:
        try:
            data = json.loads(raw_response)
            return InsightOutput(
                agent_name=self.agent_name,
                recommendation=self._to_string(data.get("recommendation", "No recommendation generated")),
                evidence=self._to_string(data.get("evidence", "No evidence provided")),
                confidence_score=float(data.get("confidence_score", 0.5)),
                cost_of_insight={
                    "model": run.model_used,
                    "input_tokens": run.input_tokens,
                    "output_tokens": run.output_tokens,
                    "estimated_usd": run.estimated_usd
                },
                alternative=self._to_string(data.get("alternative", "No alternative suggested"))
            )
        except Exception as e:
            return InsightOutput(
                agent_name=self.agent_name,
                recommendation=f"Parse error: {str(e)}",
                evidence="Raw response could not be parsed",
                confidence_score=0.0,
                cost_of_insight={
                    "model": run.model_used,
                    "input_tokens": run.input_tokens,
                    "output_tokens": run.output_tokens,
                    "estimated_usd": run.estimated_usd
                },
                alternative="Fix parsing error before retry"
            )

    def run(self, data_context: str) -> InsightOutput:
        raise NotImplementedError("Each agent must implement run()")
