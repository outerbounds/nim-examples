import random
from metaflow import FlowSpec, step, resources, tensorboard

class SimpleTb(FlowSpec):

    @step
    def start(self):
        self.countries = ['US', 'CA', 'BR', 'CN']
        self.next(self.train, foreach='countries')

    @tensorboard
    @step
    def train(self):
        print('training model...')
        import time
        idx = self.countries.index(self.input)
        for i in range(100):
            s = idx + random.random()
            self.obtb.add_scalar('score', s, i)
        self.score = random.randint(0, 10)
        self.country = self.input
        self.next(self.join)

    @step
    def join(self, inputs):
        self.best = max(inputs, key=lambda x: x.score).country
        self.next(self.end)

    @step
    def end(self):
        print(self.best, 'produced best results')

if __name__ == '__main__':
    SimpleTb()
