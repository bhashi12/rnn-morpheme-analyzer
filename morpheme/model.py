import itertools

import numpy as np
import chainer
from chainer import Variable
from chainer import Chain, ChainList
import chainer.functions as cf
import chainer.links as cl

class Recognizer(Chain):
    def __init__(self, n_words, n_cwords, n_memory, n_output):
        self.n_words = n_words
        self.n_cwords = n_cwords
        self.n_memory = n_memory
        self.n_output = n_output
        super().__init__(
            input=cl.EmbedID(self.n_words, self.n_cwords),
            memory=cl.LSTM(self.n_cwords, self.n_memory),
            output=cl.Linear(self.n_memory, self.n_output)
        )

    def reset_state(self):
        self.memory.reset_state()

    def __call__(self, x, y=None):
        input = self.input(x)
        if y is None:
            memory = self.memory(input)
        else:
            memory = self.memory(input + y)
        output = self.output(memory)
        return output


class BiClassifier(Chain):
    def __init__(self, n_input):
        self.n_input = n_input
        super().__init__(
            output=cl.Linear(self.n_input, 1)
        )

    def __call__(self, x):
        return self.output(x)


class Classifier(Chain):
    def __init__(self, n_input, n_output):
        self.n_input = n_input
        self.n_output = n_output
        super().__init__(
            output=cl.Linear(self.n_input, self.n_output)
        )

    def __call__(self, x):
        return self.output(x)

class BidirectionalRecognizer(Chain):
    def __init__(self, backward, forward):
        assert backward.n_output == forward.n_cwords
        super().__init__(
            backward=backward,
            forward=forward
        )

    def reset_state(self):
        self.backward.reset_state()
        self.forward.reset_state()

    def __call__(self, xs):
        ys = []
        for x in reversed(xs):
            ys.append(self.backward(x))
        ys.reverse()
        for x, y in zip(xs, ys):
            yield self.forward(x, y)


class Tagger(Chain):
    def __init__(self, segmenter, classifiers):
        super().__init__(
            segmenter=segmenter,
            classifiers=classifiers
        )

    def reset_state(self):
        self.recognizer.reset_state()

    def __call__(self, x):
        segs = list(itertools.accumulate(
            clf.n_input for clf in self.classifiers
        ))
        if segs:
            xs = cf.split_axis(x, segs, 1)
        else:
            xs = [x]
        
        y = self.segmenter(xs[-1])
        zs = tuple(clf(x) for x, clf in zip(xs[:-1], self.classifiers))
        return y, zs


class Analyzer(Chain):
    def __init__(self, recognizer, tagger):
        super().__init__(
            recognizer=recognizer,
            tagger=tagger
        )

    def reset_state(self):
        self.recognizer.reset_state()

    def __call__(self, xs):
        for y in self.recognizer(xs):
            yield self.tagger(y)

