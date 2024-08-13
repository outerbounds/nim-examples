from metaflow import (
    FlowSpec,
    step,
    nim,
    current,
    card,
    retry,
    catch,
    IncludeFile,
    JSONType,
    Parameter,
)
from metaflow.cards import Table, VegaChart
import time
import json

MODELS = ["meta/llama3-8b-instruct", "meta/llama3-70b-instruct"]


class ParallelLLMEval(FlowSpec):

    n = Parameter("n", default=20)
    n_per_batch = Parameter("n-per-batch", default=20)
    json_file = IncludeFile("v", default="vega_spec.json")

    @step
    def start(self):
        self.worker = list(range(self.n))
        self.next(self.query, foreach="worker")

    @nim(models=MODELS)
    @card
    @step
    def query(self):

        q = "Write a fanciful tale of princesses, a dragon, and garbage collection."

        self.openai_client_args = dict(
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": q},
            ],
            temperature=0.25,
            max_tokens=177,
        )

        # level 0 prompt tracking/versioning with Outerbounds
        self.prompt_trace = []
        for _ in range(self.n_per_batch):
            for model_name in MODELS:
                llm = current.nim.models[model_name]
                t0 = time.time()
                resp = llm(**self.openai_client_args)
                tf = time.time()
                print(
                    f"{model_name} returned {resp['usage']['completion_tokens']} tokens to client in {round(tf - t0, 3)} seconds."
                )
                assert (
                    resp["model"] == model_name
                ), f"Response model mismatch with {model_name}"
                assert len(resp["choices"]) == 1, "Too many completions in response"
                assert (
                    resp["usage"]["completion_tokens"]
                    <= self.openai_client_args["max_tokens"]
                ), "Too many tokens in completion"
                self.prompt_trace.append(
                    {
                        "prompt": self.openai_client_args,
                        "response": resp,
                        "model": model_name,
                        "time": tf - t0,
                    }
                )

        current.card.append(
            Table(
                headers=["Date", "Prompt", "Llama3 8b Response", "Llama3 70b Response"],
                data=[
                    [
                        time.strftime("%Y-%m-%d %H:%M:%S"),
                        q,
                        self.prompt_trace[0]["response"]["choices"][0]["message"][
                            "content"
                        ],
                        self.prompt_trace[1]["response"]["choices"][0]["message"][
                            "content"
                        ],
                    ]
                ],
            )
        )

        self.next(self.join)

    @card
    @step
    def join(self, inputs):
        self.prompt_trace = [trial for i in inputs for trial in i.prompt_trace]
        data = []
        for p in self.prompt_trace:
            data.append({"model": p["model"], "time": p["time"]})

        vega_spec = json.loads(self.json_file)
        for i, data_source in enumerate(vega_spec["data"]):
            if data_source["name"] == "times":
                break
        vega_spec["data"][i]["values"] = data
        chart = VegaChart(vega_spec)
        current.card.append(chart)

        self.next(self.end)

    @step
    def end(self):
        print(len(self.prompt_trace))


if __name__ == "__main__":
    ParallelLLMEval()
