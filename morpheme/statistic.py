import numpy as np

def f_measure(a):
    assert a.shape == (2, 2)
    prec = a[1, 1] / (a[1, 1] + a[0, 1] + 1e-9)
    rec = a[1, 1] / (a[1, 1] + a[1, 0] + 1e-9)
    return np.array((prec, rec, 2 * rec * prec / (rec + prec + 1e-9)))

def f_measure_micro_average(a):
    assert len(a.shape) == 2
    assert a.shape[0] == a.shape[1]
    n = a.shape[0]
    b = np.zeros((2, 2))
    s = np.sum(a)
    for i in range(n):
        b[1, 1] += a[i, i]
        b[1, 0] += np.sum(a[i, :]) - a[i, i]
        b[0, 1] += np.sum(a[:, i]) - a[i, i]
        b[0, 0] += s - b[1, 1] - b[0, 1] - b[1, 0]
    return f_measure(b / n)

def f_measure_macro_average(a):
    assert len(a.shape) == 2
    assert a.shape[0] == a.shape[1]
    n = a.shape[0]
    s = np.sum(a)
    ret = np.zeros((3,))
    t = 0
    for i in range(n):
        if np.sum(a[i,:]) == 0:
            continue
        b = np.zeros((2, 2))
        b[1, 1] = a[i, i]
        b[1, 0] = np.sum(a[i, :]) - a[i, i]
        b[0, 1] = np.sum(a[:, i]) - a[i, i]
        b[0, 0] = s - a[i, i]
        ret += f_measure(b)
        t += 1
    return ret / t
