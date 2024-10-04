from metaflow import FlowSpec, step, current, card, Parameter, pypi, nim
from metaflow.cards import Table, VegaChart, Markdown, ProgressBar
import time

MODELS = [
    "nvidia/nv-rerankqa-mistral-4b-v3"
]

@nim(models=MODELS)
class Rerank(FlowSpec):

    dataset = Parameter(
        name="data", 
        default="jinaai/hotpotqa-reranking-en", 
        help="Name of HuggingFace Hub dataset"
    )
    max_parallel = Parameter(
        name="max_parallel",
        default=5,
        type=int,
        help="Maximum number of concurrent requests"
    )
    max_per_batch = Parameter(
        name="max_per_batch",
        default=50,
        type=int,
        help="Maximum number of queries per batch"
    )
    model = MODELS[0]

    @pypi(packages={'pandas': '2.2.2', 'pyarrow': '17.0.0', 'huggingface_hub': '0.24.2'})
    @step
    def start(self):
        import pandas as pd

        splits = {
            'test': 'data/test-00000-of-00001.parquet', 
            'eval': 'data/eval-00000-of-00001.parquet', 
            'dev': 'data/test-00000-of-00001.parquet'
        }
        # self.df = pd.read_parquet("hf://datasets/jinaai/hotpotqa-reranking-en/" + splits["dev"])
        # df_eval = pd.read_parquet("hf://datasets/jinaai/hotpotqa-reranking-en/" + splits["eval"])
        self.df = pd.read_parquet("hf://datasets/jinaai/hotpotqa-reranking-en/" + splits["test"])

        def batch_dataframe(dataframe, batch_size, max_batches=None):
            num_batches = len(dataframe) // batch_size + (1 if len(dataframe) % batch_size != 0 else 0)
            batches = [dataframe[i*batch_size:(i+1)*batch_size] for i in range(num_batches)]
            if max_batches:
                return batches[:max_batches]
            return batches

        if self.max_per_batch is not None:
            n_per_batch = self.max_per_batch
        else:
            n_per_batch = (self.df.shape[0] // self.max_parallel) + 1
        self.batch = batch_dataframe(self.df, n_per_batch, self.max_parallel)
    
        self.next(self.rerank, foreach='batch')

    @card(type='blank', id='progress', refresh_interval=1)
    @pypi(packages={'numpy': '2.0.1', 'pandas': '2.2.2'})
    @step
    def rerank(self):
        import requests
        import pandas as pd
        
        headers = {
            "accept": "application/json",
            "Content-Type": "application/json"
        }
        pbar = ProgressBar(max=len(self.input), label="Queries completed")
        current.card['progress'].append(pbar)
        current.card['progress'].refresh()
        self.exp_tracking_data = []
        for i, (index, row) in enumerate(self.input.iterrows()):      
            request_data = {
                "query": {"text": row.query},
                "passages": [{"text": p} for p in row.positive],
                "truncate": "END"
            }
            t0 = time.time()
            res_json = current.nim.models[self.model](**request_data)
            # API Reference: https://docs.nvidia.com/nim/nemo-retriever/text-reranking/latest/reference.html
            tf = time.time()
            pbar.update(i+1)
            current.card['progress'].refresh()
            _df = request_data
            _df['client-e2e-time'] = tf - t0
            _df['rankings'] = res_json['rankings']
            self.exp_tracking_data.append(_df)
        self.next(self.join)

    @card(type='blank', id='exp_track_task')
    @pypi(packages={'numpy': '2.0.1', 'pandas': '2.2.2', 'altair': '5.3.0'})
    @step
    def join(self, inputs):
        import pandas as pd
        import altair as alt

        self.exp_tracking_data = pd.concat([
            pd.DataFrame(i.exp_tracking_data)
            for i in inputs
        ])
        bar = alt.Chart(self.exp_tracking_data).mark_boxplot().encode(
            alt.X('client-e2e-time:Q').title('Client end-to-end time (s)')
        )
        text_min = alt.Chart(self.exp_tracking_data).mark_text(align='right', dx=-5).encode(
            x='min(client-e2e-time):Q',
            text=alt.Text('min(client-e2e-time):Q', format='.2f')
        )
        text_max = alt.Chart(self.exp_tracking_data).mark_text(align='left', dx=5).encode(
            x='max(client-e2e-time):Q',
            text=alt.Text('max(client-e2e-time):Q', format='.2f')
        )
        usage_plot = (bar + text_min + text_max).properties(
            title=alt.Title(text=f"Reranking usage stats of {self.model}", subtitle=f'Based on {len(self.exp_tracking_data)} observations'),
            width=250,
            height=25
        )
        current.card['exp_track_task'].append(
            Markdown("### Usage metrics")
        )
        current.card['exp_track_task'].append(VegaChart.from_altair_chart(usage_plot))
        current.card['exp_track_task'].append(Markdown("### Reranking Trace"))
        current.card['exp_track_task'].append(
            Table.from_dataframe(
                pd.DataFrame(self.exp_tracking_data)
            )
        )
        self.next(self.end)

    @step
    def end(self):
        pass


if __name__ == '__main__':
    Rerank()