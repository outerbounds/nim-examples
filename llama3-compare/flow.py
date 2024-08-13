from metaflow import FlowSpec, step, nim, current, card
from metaflow.cards import Table
import time

MODELS = [
    'meta/llama3-8b-instruct', 
    'meta/llama3-70b-instruct'
]

class Llama3Comparison(FlowSpec):

    @card
    @nim(models=MODELS)
    @step
    def start(self):

        q =  "What's the weather like today?"

        self.openai_client_args = dict(
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": q}
            ],
            max_tokens=62
        )

        # level 0 prompt tracking/versioning with Outerbounds
        self.prompt_trace = []
        for i in range(10):
            for model_name in MODELS: 
                llm = current.nim.models[model_name]
                t0 = time.time()
                resp = llm(**self.openai_client_args)
            tf = time.time()
            print(f"{model_name} returned {resp['usage']['completion_tokens']} tokens to client in {round(tf - t0, 3)} seconds.")
            assert resp['model'] == model_name, f'Response model mismatch with {model_name}'
            assert len(resp['choices']) == 1, 'Too many completions in response'
            assert resp['usage']['completion_tokens'] <= self.openai_client_args['max_tokens'], 'Too many tokens in completion'
            self.prompt_trace.append({
                'prompt': self.openai_client_args,
                'response': resp,
                'model': model_name,
                'time': tf - t0
            })

        current.card.append(
            Table(
                headers = ['Date', 'Prompt', 'Llama3 8b Response', 'Llama3 70b Response'],
                data = [
                    [
                        time.strftime('%Y-%m-%d %H:%M:%S'),
                        q, 
                        self.prompt_trace[0]['response']['choices'][0]['message']['content'],
                        self.prompt_trace[1]['response']['choices'][0]['message']['content']
                    ]
                ]
            )
        )

        self.next(self.end)

    @step
    def end(self):
        # print(self.prompt_trace)
        pass


if __name__ == '__main__':
    Llama3Comparison()