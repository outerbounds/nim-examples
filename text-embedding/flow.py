from metaflow import FlowSpec, step, current, card, Parameter, IncludeFile, pypi, nim
from metaflow.cards import VegaChart, Markdown

MODELS = [
    # "nvidia/nv-embedqa-e5-v5",
    'nvidia/nv-embedqa-mistral-7b-v2',
    # 'snowflake/arctic-embed-l'
]


@nim(models=MODELS)
class TextEmbedding(FlowSpec):

    text = IncludeFile(name="data", default="data.txt", help="Texts to embed")
    batch_size = Parameter("-bsz", default=10)
    model = MODELS[0]

    @step
    def start(self):
        self.text_chunks = self.text.split("\n")

        self.batch = [
            self.text_chunks[i : i + self.batch_size]
            for i in range(0, len(self.text_chunks), self.batch_size)
        ]
        self.next(self.embed, foreach="batch")

    @step
    def embed(self):
        import time
        import requests

        self.text_batch = self.input

        t0 = time.time()
        res_json = current.nim.models[self.model](
            input=self.text_batch, input_type="query"
        )
        # API reference: https://docs.nvidia.com/nim/nemo-retriever/text-embedding/latest/reference.html
        tf = time.time()
        res_data = res_json["data"]
        self.usage_stats = res_json["usage"]
        self.usage_stats["client-e2e-time"] = tf - t0
        self.embeddings = res_data
        self.next(self.join)

    @pypi(packages={"numpy": "2.0.1", "pandas": "2.2.2"})
    @step
    def join(self, inputs):
        import numpy as np
        import pandas as pd

        self.embeddings = None
        self.embeddings_meta = pd.DataFrame()
        self.stats = []
        for batch_input in inputs:
            for i in range(len(batch_input.embeddings)):
                if self.embeddings is not None:
                    self.embeddings = np.vstack(
                        [self.embeddings, batch_input.embeddings[i]["embedding"]]
                    )
                else:
                    self.embeddings = [batch_input.embeddings[i]["embedding"]]
            _start_idx = len(self.embeddings_meta)
            _end_idx = _start_idx + len(batch_input.text_batch)
            _meta_df = {
                "text": batch_input.text_batch,
                "embedding_row_idx": range(_start_idx, _end_idx),
            }
            self.embeddings_meta = pd.concat(
                [self.embeddings_meta, pd.DataFrame(_meta_df)]
            )
            self.stats.append(batch_input.usage_stats)
        self.next(self.end)

    @card(type="blank", id="usage_stats")
    @card(type="blank", id="plot")
    @pypi(
        packages={
            "numpy": "2.0.1",
            "pandas": "2.2.2",
            "scikit-learn": "1.5.1",
            "altair": "5.3.0",
        }
    )
    @step
    def end(self):
        import altair as alt
        from sklearn.manifold import TSNE
        import pandas as pd

        # plot usage
        usage_df = pd.DataFrame(self.stats)
        bar = (
            alt.Chart(usage_df)
            .mark_boxplot()
            .encode(alt.X("client-e2e-time:Q").title("Client end-to-end time (s)"))
        )
        text_min = (
            alt.Chart(usage_df)
            .mark_text(align="right", dx=-5)
            .encode(
                x="min(client-e2e-time):Q",
                text=alt.Text("min(client-e2e-time):Q", format=".2f"),
            )
        )
        text_max = (
            alt.Chart(usage_df)
            .mark_text(align="left", dx=5)
            .encode(
                x="max(client-e2e-time):Q",
                text=alt.Text("max(client-e2e-time):Q", format=".2f"),
            )
        )
        usage_plot = (bar + text_min + text_max).properties(
            title=alt.Title(
                text=f"Text embedding usage stats of {self.model}",
                subtitle=f"For each batch based on {len(usage_df)} batches of {self.batch_size} observations",
            ),
            width=250,
            height=25,
        )
        current.card["usage_stats"].append(Markdown("### Usage metrics"))
        current.card["usage_stats"].append(VegaChart.from_altair_chart(usage_plot))

        # plot embeddings
        tsne = TSNE(n_components=2, random_state=77)
        embeddings_2d = tsne.fit_transform(self.embeddings)
        plot_df = pd.DataFrame(embeddings_2d, columns=["x", "y"])
        plot_df["text"] = self.embeddings_meta.text.values
        chart = (
            alt.Chart(plot_df)
            .mark_circle(size=60)
            .encode(
                x="x",
                y="y",
                tooltip=["text"],
            )
            .properties(
                title=f"Text Embeddings of {self.model} projected to 2D with TSNE",
                width=500,
                height=350,
            )
        )
        current.card["plot"].append(VegaChart.from_altair_chart(chart))


if __name__ == "__main__":
    TextEmbedding()
