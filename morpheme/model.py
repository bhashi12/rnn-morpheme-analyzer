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
    def __init__(self, n_input):
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
        y = self.segmenter(x)
        zs = tuple(clf(x) for clf in self.classifiers)
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

    def loss(self, xs, ts, uss=None):
        tags = self(Variable(
            np.array([x], dtype=np.int32)
        ) for x in xs)
        zss = []
        loss = Variable(np.array(0, dtype=np.float32))
        for t, (y, zs) in zip(ts, tags):
            loss += cf.sigmoid_cross_entropy(
                y, Variable(np.array([[t]], dtype=np.int32))
            )
            if t:
                vss.append(zs)
        if uss:
            assert len(uss) == len(zss)
            for us, zs in zip(uss, zss):
                for u, z, clf in zip(us, zs, self.tagger.classifier):
                    loss += cf.softmax_cross_entropy(
                        z, Variable(np.array([u], dtype=np.int32)
                    )
        return loss

    def test(self, xs, ts, uss=None):
        tags = self(Variable(
            np.array([x], dtype=np.int32)
        ) for x in xs)
        zss = []
        
        y_mat = np.zeros((2, 2))
        zs_mat = tuple(
            np.zeros((clf.n_output, clf.n_output))
            for clf in self.tagger.classifiers
        )
        for t, (y, zs) in zip(ts, tags):
            y_mat[t, int(cf.sigmoid(y).data[0, 0] > 0.5)] += 1.0
            if t:
                attrs.append(zs)
        if uss:
            assert len(uss) == len(zss)
            for us, zs in zip(uss, zss):
                for m, u, z, clf in zip(zs_mat, us, zs):
                    m[u, cf.softmax(z).data.argmax(1)[0]] += 1
        return y_mat, z_mats

    def generate(self, xs):
        tags = self(Variable(
            np.array([x], dtype=np.int32)
        ) for x in xs)
        buf = bytearray()
        for x, (y, zs) in zip(xs, tags):
            buf.append(x)
            if cf.sigmoid(y).data[0, 0] > 0.5:
                yield (
                    buf.decode('utf-8', 'replace'),
                    tuple(
                        cf.softmax(z).data.argmax(1)[0]
                        for z in zs
                    )
                )
                buf = bytearray()
