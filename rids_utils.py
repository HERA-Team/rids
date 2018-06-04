from __future__ import print_function


def spectrum_reader(filename, spec, polarization=None):
    """
    This reads in an ascii spectrum file.
    """
    if filename is None:
        return
    if polarization is not None:
        spec.polarization = polarization
    with open(filename, 'r') as f:
        for line in f:
            data = [float(x) for x in line.split()]
            spec.freq.append(data[0])
            spec.val.append(data[1])


def spectrum_plotter(e, x, y, fmt, clr):
    import matplotlib.pyplot as plt
    try:
        _X = x[:len(y)]
        if 'baseline' in e.lower():
            plt.plot(_X, y, clr)
        else:
            plt.plot(_X, y, fmt, color=clr)
    except ValueError:
        _Y = y[:len(x)]
        if 'baseline' in e.lower():
            plt.plot(x, _Y, clr)
        else:
            plt.plot(x, _Y, fmt, color=clr)


def peel_type_polarization(v):
    ftypes = {'ave': 'ave', 'maxh': 'maxhold', 'minh': 'minhold'}
    ftype = None
    pol = None
    for t in ftypes:
        if t in v.lower():
            ftype = ftypes[t]
            break
    if '_E' in v:
        pol = 'E'
    elif '_N' in v:
        pol = 'N'
    return ftype, pol


def peel_time_stamp(v):
    if v is None:
        return None
    z = v.split('/')[-1]
    ts = "{}-{}".format(z.split('_')[0], z.split('_')[1])
    return ts


def fix_json_list(jsd):
    spaces = ['\n', ' ']
    fixed = ''
    in_list = False
    for c in jsd:
        if c == '[':
            in_list = True
        elif c == ']':
            in_list = False

        if not in_list:
            fixed += c
        elif in_list and c not in spaces:
            fixed += c
    return fixed
