from pathlib import Path
from collections import Counter
from random import random, shuffle

import argparse
import itertools
import pickle
import csv

import numpy as np
import chainer
from chainer import Function, Variable, optimizers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as cf
import chainer.links as cl

import dataset
import util


src = Path(__file__).parent
def_dir = src/'def'
pos_id_def = def_dir/'pos-id.def'
conj_type_def = def_dir/'conj_type-id.csv'
conj_form_def = def_dir/'conj_form-id.csv'
meta_name = 'meta.pickle'
model_name = lambda epoch: 'model-{}.model'.format(epoch)
optimizer_name = lambda epoch: 'optimizer-{}.model'.format(epoch)

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
        self.value = self.output(x)
        return cf.sigmoid(self.value)


class Classifier(Chain):
    def __init__(self, n_input, mapping):
        self.n_input = n_input
        self.n_pos = len(mapping)
        self.mapping = mapping
        super().__init__(
            output=cl.Linear(self.n_input, self.n_pos)
        )

    def __call__(self, x):
        self.value = self.output(x)
        return cf.softmax(self.value)


class ForwardAnalyzer(Chain):
    def __init__(self, recognizer, segmenter,
                 pos_classifier,
                 conj_type_classifier,                 
                 conj_form_classifier):
        super().__init__(
            recognizer=recognizer,
            segmenter=segmenter,
            pos_classifier=pos_classifier,
            conj_type_classifier=conj_type_classifier,
            conj_form_classifier=conj_form_classifier
        )

    def reset_state(self):
        self.recognizer.reset_state()

    def __call__(self, x, y=None):
        info = self.recognizer(x, y)
        spliter = list(itertools.accumulate([
            self.segmenter.n_input,
            self.pos_classifier.n_input,
            self.conj_type_classifier.n_input
        ]))
        in_segmenter, in_pos, in_conjt, in_conjf = cf.split_axis(
            info, spliter, 1
        )
        self.is_eow = self.segmenter(in_segmenter)
        self.pos = self.pos_classifier(in_pos)
        self.conj_type = self.conj_type_classifier(in_conjt)
        self.conj_form = self.conj_form_classifier(in_conjf)
        return self.is_eow, self.pos, self.conj_type, self.conj_form
    
class Analyzer(Chain):
    def __init__(self, forward, backward):
        super().__init__(
            forward=forward,
            backward=backward
        )

    def reset_state(self):
        self.forward.reset_state()
        self.backward.reset_state()

    def _solve(self, sentence):
        self.reset_state()
        self.zerograds()
        rxs = generate_data(sentence[::-1])
        xs = generate_data(sentence)
        ts = generate_label(sentence)
        ys = []
        labels = []
        data = []
        for x in rxs:
            ys.append(
                self.backward(Variable(np.array([x], dtype=np.int32)))
            )
        ys.reverse()
        for x, label, y in zip(xs, ts, ys):
            self.forward(Variable(np.array([x], dtype=np.int32)), y)
            labels.append(self.forward.segmenter.value)
            if label:
                data.append([
                    self.forward.pos_classifier.value,
                    self.forward.conj_type_classifier.value,
                    self.forward.conj_form_classifier.value
                ])
        return labels, data

    def train(self, sentence):
        labels, data = self._solve(sentence)
        
        ts = generate_label(sentence)
        self.loss = Variable(np.array(0, dtype=np.float32))
        for exp, act in zip(ts, labels):
            self.loss += cf.sigmoid_cross_entropy(
                act,
                Variable(np.array([[exp]], dtype=np.int32))
            )
        for exp, act in zip(sentence, data):
            es = [
                self.forward.pos_classifier.mapping[exp.pos],
                self.forward.conj_type_classifier.mapping[exp.conj_type],
                self.forward.conj_form_classifier.mapping[exp.conj_form]
            ]
            self.loss += sum([
                cf.softmax_cross_entropy(
                    a, Variable(np.array([ev], dtype=np.int32))
                )
                for a, ev in zip(act, es)
            ])
        return self.loss            

    def test(self, sentence):
        labels, data = self._solve(sentence)
        
        mat = np.zeros((2, 2))
        cnt = np.zeros((3,))
        acc = np.zeros((3,))
        
        ts = generate_label(sentence)
        for exp, act in zip(ts, labels):
            mat[exp, int(cf.sigmoid(act).data[0, 0] > 0.5)] += 1.0
        for exp, act in zip(sentence, data):
            es = [
                self.forward.pos_classifier.mapping[exp.pos],
                self.forward.conj_type_classifier.mapping[exp.conj_type],
                self.forward.conj_form_classifier.mapping[exp.conj_form]
            ]
            for i, (a, e) in enumerate(zip(act, es)):
                cnt[i] += 1
                acc[i] += int(cf.softmax(a).data.argmax(1)[0] == e)
        return mat, cnt, acc

    def generate(self, sentence):
        labels, _ = self._solve(sentence)
        
        xs = generate_data(sentence)
        buf = bytearray()
        words = []
        info = []
        for x, label in zip(xs, labels):
            buf.append(x)
            if cf.sigmoid(label).data[0, 0] > 0.5:
                words.append(buf.decode('utf-8', 'replace'))
                buf = bytearray()
        return words

    
root = Path(__file__).parent
data_dir = root/'data'/'Translate'
out_dir = root/'out'
            

def train(model, optimizer, sentence):
    model.zerograds()
    model.reset_state()    
    loss = model.train(sentence)
    loss.backward()
    optimizer.clip_grads(10)
    optimizer.update()
            

def test(model, sentence):
    model.reset_state()
    return model.test(sentence)

def init():
    with pos_id_def.open('r') as f:
        def parse(line):
            attr, pos_id = line.split()
            attr = tuple(attr.split(','))
            return (attr, int(pos_id))
        pos_mapping = util.OneToOneMapping(
            parse(line) for line in f
        )
    with conj_type_def.open('r') as f:
        conj_type_mapping = util.OneToOneMapping(
            (row[1], int(row[0])) for row in csv.reader(f)
        )
    with conj_form_def.open('r') as f:
        conj_form_mapping = util.OneToOneMapping(
            (row[1], int(row[0])) for row in csv.reader(f)   
        )
    model = Analyzer(
        ForwardAnalyzer(
            Recognizer(256, 256, 256, 64 + 256 + 128 + 128),
            BiClassifier(64),
            Classifier(256, pos_mapping),
            Classifier(128, conj_type_mapping),
            Classifier(128, conj_form_mapping)
        ),
        Recognizer(256, 256, 256, 256),
    )        
    optimizer = optimizers.AdaGrad(lr=0.01)
    optimizer.setup(model)
    return model, optimizer

    
def load(load_dir, epoch):
    with (load_dir/meta_name).open('rb') as f:
        model, optimizer = np.load(f)
    serializers.load_npz(
        str(load_dir/model_name(epoch)),
        model
    )
    serializers.load_npz(
        str(load_dir/optimizer_name(epoch)),
        optimizer
    )
    return model, optimizer

def generate_data(sentence):
    return itertools.chain.from_iterable(
        info.surface_form.encode('utf-8') for info in sentence
    )

def generate_label(sentence):
    return itertools.chain.from_iterable(
        (int(i == len(info.surface_form.encode('utf-8')))
         for i, c in enumerate(info.surface_form.encode('utf-8'), 1))
        for info in sentence
    )

    
def run_training(args):
    out_dir = Path(args.directory)
    sentences = dataset.load(args.source)
    if args.epoch is not None:
        start = args.epoch + 1
        model, optimizer = load(out_dir, args.epoch)
        sentences = itertools.islice(sentences, start, None)
    else:
        start = 0
        model, optimizer = init()        
        if (out_dir/meta_name).exists():
            if input('Overwrite? [y/N]: ').strip().lower() != 'y':
                exit(1)
        with (out_dir/meta_name).open('wb') as f:
            np.save(f, [model, optimizer])
    batchsize = 1000
    for i, sentence in enumerate(sentences, start):
        if i % batchsize == 0:
            print()
            serializers.save_npz(str(out_dir/model_name(i)), model)
            serializers.save_npz(str(out_dir/optimizer_name(i)), optimizer)
        else:
            print(util.progress('sentence',
                           (i % batchsize) / batchsize, 100),
                  end='')
        train(model, optimizer, sentence)
        
        

def run_test(args):
    out_dir = Path(args.directory)
    sentences = dataset.load(args.source)
    model, _ = load(out_dir, args.epoch)
    mat_sum = np.zeros((2, 2))
    count_sum = np.zeros((3, ))
    acc_sum = np.zeros((3, ))
    for i, sentence in enumerate(itertools.islice(sentences, 100)):
        mat, count, acc = test(model, sentence)
        mat_sum += mat
        count_sum += count
        acc_sum += acc
        
        prec = mat_sum[1, 1] / (mat_sum[1, 1] + mat_sum[0, 1])
        rec = mat_sum[1, 1] / (mat_sum[1, 1] + mat_sum[1, 0])
        print('precision:', prec)
        print('recall:', rec)
        print('F-measure:', 2 * rec * prec / (rec + prec))

        print('accuracy:', acc_sum / count_sum)

        model.reset_state()
        print('expect:', '/'.join(info.surface_form for info in sentence))
        print('actual:', '/'.join(model.generate(sentence)))
            
def main():
    parser = argparse.ArgumentParser(
        description='Encoder-Decoder Model'
    )
    parser.set_defaults(func=lambda x: parser.print_usage())
    subparsers = parser.add_subparsers()
    train_parser = subparsers.add_parser(
        'train',
        help='Training Encoder-Decoder Model'
    )
    train_parser.add_argument(
        '-e', '--epoch',
        type=int,
        help='restore and continue training from EPOCH'
    )    
    train_parser.add_argument(
        'source',
        help='path to translation source'
    )
    train_parser.add_argument(
        'directory',
        help='directory in which model is saved'
    )
    train_parser.set_defaults(func=run_training)


    test_parser = subparsers.add_parser(
        'test',
        help='Testing Encoder-Decoder Model'
    )
    test_parser.add_argument(
        'directory',
        help='directory in which model was saved'
    )
    test_parser.add_argument(
        'epoch',
        help='generation of model'
    )
    test_parser.add_argument(
        'source',
        help='path to translation source'
    )
    test_parser.set_defaults(func=run_test)

    args = parser.parse_args()
    if args.func is None:
        parser.usage()
        exit(1)
    else:
        args.func(args)

    
if __name__ == '__main__':
    main()
            
