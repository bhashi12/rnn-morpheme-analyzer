import collections

Entry = collections.namedtuple(
    'Entry',
    [
        'surface_form',
        'pos',
        'conj_type',
        'conj_form',
        'base_form',
        'ruby',
        'pronunciation'

    ]                     
)

NIL = '*'

def parse(line):
    try:
        w, attr = line.split('\t')
    except ValueError:
        w = ' '
        attr = line
    attr = attr.split(',')[:9]
    attr += [NIL] * (9 - len(attr))
    attr = [tuple(attr[:4])] + attr[4:]
    return Entry(*([w] + attr))

def load(lines):
    buf = []
    for line in lines:
        if line.strip() != 'EOS':
            buf.append(parse(line.strip()))
        elif len(buf) > 0:
            yield tuple(buf)
            buf = []
    if len(buf) > 0:
        yield buf
