import collections.abc
import itertools

#ref: http://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console
def progress(prompt, percent, bar_length=20):
    hashes = '#' * int(round(percent * bar_length))
    spaces = ' ' * (bar_length - len(hashes))    
    return "\r{}: [{}] {}%".format(
        prompt,
        hashes + spaces,
        int(round(percent * 100)))

#ref: https://docs.python.org/3.5/library/itertools.html#itertools.zip_longest
def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return itertools.zip_longest(*args, fillvalue=fillvalue)


class OneToOneMapping(collections.abc.MutableMapping):

    def __init__(self, *args, **kwargs):
        self.forward = {}
        self.backward = {}
        self.update(dict(*args, **kwargs))

    def __getitem__(self, key):
        return self.forward[key]

    def __call__(self, value):
        return self.backward[value]

    def __setitem__(self, key, value):
        if key in self.forward:
            raise KeyError(
                'key {} is already mapped to value {}'.format(
                    key, self.forward[key]
                )
            )
        self.forward[key] = value
        
        if value in self.backward:
            raise KeyError(
                'value {} is already mapped to key {}'.format(
                    value, self.backward[key]
                )
            )
        self.backward[value] = key

    def __delitem__(self, key):
        del self.backward[self.forward[key]]
        del self.forward[key]

    def __iter__(self):
        return iter(self.forward)

    def __len__(self):
        assert len(self.forward) == len(self.backward)
        return len(self.forward)


class AutoIncrementMapping(OneToOneMapping):
    def add(self, key):
        value = len(self)
        self[key] = value

