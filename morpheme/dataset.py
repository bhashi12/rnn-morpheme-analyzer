import argparse
import pathlib
import pickle
import random
import lzma

import mecab
import util

def load(filename):
    with lzma.open(filename, 'rb') as dataset:
        while True:
            try:
                yield pickle.load(dataset)
            except EOFError:
                break

def separate(lines, p, train_out, test_out):
    dataset = list(mecab.load(lines))
    random.shuffle(dataset)
    for i, sentence in enumerate(dataset, 1):
        if random.random() < p:
            pickle.dump(sentence, train_out)
        else:
            pickle.dump(sentence, test_out)
        if i % 100 == 0:
            print(util.progress('write', i / len(dataset), 100), end='')
    print()

def main():
    parser = argparse.ArgumentParser(
        description='dataset generator'
    )
    parser.add_argument(
        '-p', '--possibility',
        type=float,
        default=0.9,
        help='possibility to add train dataset'
    )
    parser.add_argument(
        'source',
        help='path to mecab-processed corpus (xz compressed)'
    )
    parser.add_argument(
        'train',
        help='path for writing training dataset (xz compressed)'
    )
    parser.add_argument(
        'test',
        help='path for writing testing dataset (xz compressed)'
    )
    args = parser.parse_args()
    with lzma.open(args.source, 'rt') as source,\
         lzma.open(args.train, 'wb') as train,\
         lzma.open(args.test, 'wb') as test:
            separate(source, args.possibility, train, test)
    
if __name__ == '__main__':
    main()
