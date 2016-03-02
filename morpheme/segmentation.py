import pathlib
import collections
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
import statistic
import model as md


meta_name = 'meta.pickle'
model_name = lambda epoch: 'model-{}.model'.format(epoch)
optimizer_name = lambda epoch: 'optimizer-{}.model'.format(epoch)
        
def loss(model, xs, ts, uss=None):
    model.reset_state()
    tags = model([Variable(
        np.array([x], dtype=np.int32)
    ) for x in xs])
    zss = []
    d = Variable(np.array(0, dtype=np.float32))
    for t, (y, zs) in zip(ts, tags):
        d += cf.sigmoid_cross_entropy(
            y, Variable(np.array([[t]], dtype=np.int32))
        )
        if t:
            zss.append(zs)
    if uss:
        assert len(uss) == len(zss)
        for us, zs in zip(uss, zss):
            for u, z in zip(us, zs):
                d += cf.softmax_cross_entropy(
                    z, Variable(np.array([u], dtype=np.int32))
                )
    return d

def test(model, xs, ts, uss=None):
    model.reset_state()
    tags = model([Variable(
        np.array([x], dtype=np.int32)
    ) for x in xs])
    zss = []

    y_mat = np.zeros((2, 2))
    zs_mat = tuple(
        np.zeros((clf.n_output, clf.n_output))
        for clf in model.tagger.classifiers
    )
    for t, (y, zs) in zip(ts, tags):
        y_mat[t, int(cf.sigmoid(y).data[0, 0] > 0.5)] += 1.0
        if t:
            zss.append(zs)
    if uss:
        assert len(uss) == len(zss)
        for us, zs in zip(uss, zss):
            for m, u, z in zip(zs_mat, us, zs):
                m[u, cf.softmax(z).data.argmax(1)[0]] += 1
    return y_mat, zs_mat

def generate(model, xs):
    model.reset_state()
    tags = model([Variable(
        np.array([x], dtype=np.int32)
    ) for x in xs])
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


def train(model, optimizer, xs, ts, uss=None):
    model.zerograds()
    d = loss(model, xs, ts, uss)
    d.backward()
    optimizer.clip_grads(10)
    optimizer.update()

Storage = collections.namedtuple(
    'Storage',
    [
        'model',
        'optimizer'
    ]    
)

def generate_data(sentence):
    return bytearray(itertools.chain.from_iterable(
        word.encode('utf-8') for word in sentence
    ))

def generate_label(sentence):
    return tuple(itertools.chain.from_iterable(
        (int(i == len(word.encode('utf-8')))
         for i, c in enumerate(word.encode('utf-8'), 1))
        for word in sentence
    ))



def load(load_dir, epoch):
    with (load_dir/meta_name).open('rb') as f:
        storage = Storage(*np.load(f)[0])
        print(storage)
    serializers.load_npz(
        str(load_dir/model_name(epoch)),
        storage.model
    )
    serializers.load_npz(
        str(load_dir/optimizer_name(epoch)),
        storage.optimizer
    )
    return storage


def init(args):
    def parse(line):
        attr, pos_id = line.split()
        attr = tuple(attr.split(','))
        return (attr, int(pos_id))
    model = md.Analyzer(
        md.BidirectionalRecognizer(
            md.Recognizer(256, 100, 100, 100),
            md.Recognizer(256, 100, 100, 100)
        ),
        md.Tagger(
            md.BiClassifier(100),
            chainer.ChainList()
        )
    )        
    optimizer = optimizers.AdaGrad(lr=0.01)
    optimizer.setup(model)
    return Storage(model, optimizer)

    
def run_training(args):
    out_dir = pathlib.Path(args.directory)
    sentences = [line.split() for line in args.source]
    
    if args.epoch is not None:
        start = args.epoch + 1
        storage = load(out_dir, args.epoch)
        sentences = itertools.islice(sentences, start, None)
    else:
        start = 1
        storage = init(args)        
        if (out_dir/meta_name).exists():
            if input('Overwrite? [y/N]: ').strip().lower() != 'y':
                exit(1)
        with (out_dir/meta_name).open('wb') as f:
            np.save(f, [storage])
        
    batchsize = 10000
    for i, sentence in enumerate(sentences, start):
        if i % batchsize == 0:
            print()
            serializers.save_npz(
                str(out_dir/model_name(i)),
                storage.model
            )
            serializers.save_npz(
                str(out_dir/optimizer_name(i)),
                storage.optimizer
            )
        else:
            print(
                util.progress(
                    'batch {}'.format(i // batchsize),
                    (i % batchsize) / batchsize, 100),
                end=''
            )
        train(storage.model,
              storage.optimizer,
              generate_data(sentence),
              generate_label(sentence)
        )


def run_test(args):
    out_dir = pathlib.Path(args.directory)
    sentences = [line.split() for line in args.source]
    storage = load(out_dir, args.epoch)
    y_sum = None
    zs_sum = None
    
    for i, sentence in enumerate(sentences):
        y_mat, zs_mat = test(
            storage.model,
            generate_data(sentence),
            generate_label(sentence)
        )
        if i == 0:
            y_sum = y_mat
            zs_sum = zs_mat
        else:
            y_sum += y_mat
            for z_sum, z_mat in zip(zs_sum, zs_mat):
                z_sum += z_mat
        
        prec, rec, f = statistic.f_measure(y_sum)
        print('== segmentation ==')
        print('precision:', prec)
        print('recall:', rec)
        print('F-measure:', f)        

        print('expect:', '/'.join(
            word for word in sentence)
        )
        print('actual:', '/'.join(
            y for (y, zs) in generate(
                storage.model,
                generate_data(sentence)
            )
        ))
            
def main():
    src = pathlib.Path(__file__).parent
    def_dir = src/'def'
    pos_id_def = def_dir/'pos-id.def'
    conj_type_def = def_dir/'conj_type-id.csv'
    conj_form_def = def_dir/'conj_form-id.csv'
    
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
        type=argparse.FileType('r'),
        help='path to corpus'
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
        type=argparse.FileType('r'),
        help='path to corpus'
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
            
