
# Logging metrics to Tensorboard

Outerbounds comes with a `@tensorboard` decorator that allows you to emit
metrics and other artifacts from Metaflow tasks to a Tensorboard dashboard.

Metrics are organized by Metaflow namespace and run ID, making it easy to
compare results across tasks (e.g. for hyperparameter optimization) or
across runs. Optionally, you can decorate the flow with
[the `@project` decorator](https://docs.metaflow.org/production/coordinating-larger-metaflow-projects)
which further isolates metrics between production and experimentation,
as well as separate branches of production.

## Usage: Emitting metrics

```
from metaflow import tensorboard
```

1. Decorate tasks emit metrics with `@tensorboard`.

2. The `@tensorboard` decorator exposes a handle `self.obtb` which is an
   instance of [PyTorch's `SummaryWriter`](https://pytorch.org/docs/stable/tensorboard.html).
   Follow PyTorch examples for emitting metrics using the handle.

You need to have `torch` and `tensorboard` packages available in the task
environment to use the decorator, e.g. using `@pypi`:
```
@pypi(packages={'torch': '2.4.1', 'tensorboard': '2.18.0'})
```

## Usage: Inspecting results on Tensorboard

The beginning of the task output (visible in the task UI) shows lines like:

```
INSPECTING RESULTS
Execute one of these commands on your workstation:
Compare tasks of this run: obtb tb_mnist.user.alice@example.com.TBMnistFlow/TBMnistFlow/4725
Compare across runs: obtb tb_mnist.user.alice@example.com.TBMnistFlow/TBMnistFlow
```

You can copy-paste one the lines to the terminal of your workstation, e.g.
```
obtb tb_mnist.user.alice@example.com.TBMnistFlow/TBMnistFlow
```
to compare results across your personal runs of `TBMnistFlow`.

The `obtb` command starts a tensorboard instance on the workstation, configuring
it to show results only for the desired prefix.

You can then open the dashboard by clicking *tensorboard* link in the workspace
UI.
