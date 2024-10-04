
from metaflow import tensorboard, project, FlowSpec, step, pypi, resources

@project(name='tb_mnist')
class TBMnistFlow(FlowSpec):

    @step
    def start(self):
        self.batch_sizes = [32, 64, 128]
        self.next(self.train, foreach='batch_sizes')

    @pypi(packages={'torch': '2.4.1',
                    'tensorboard': '2.18.0',
                    'torchvision': '0.19.1'})
    @resources(cpu=2)
    @tensorboard
    @step
    def train(self):
        import mnist_torch
        self.batch_size = self.input
        self.accuracy = mnist_torch.train_model(self.obtb, batch_size=self.batch_size)
        print(f"Results: {self.batch_size} accuracy: {self.accuracy}")
        self.next(self.join)

    @step
    def join(self, inputs):
        self.best = max(inputs, key=lambda x: x.accuracy).batch_size
        print(f"Best results: batch size = {self.best}")
        self.next(self.end)

    @step
    def end(self):
        pass

if __name__ == "__main__":
    TBMnistFlow()
