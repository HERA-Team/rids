from __future__ import print_function


def spectrum_reader(filename, spec, polarization=None):
    """
    This reads in an ascii spectrum file.
    If two columns stores as freq, val (cal and baseline)
    If three columns stores as freq, ave, maxhold (event)
    """
    if filename is None:
        return
    if polarization is not None:
        spec.polarization = polarization
    with open(filename, 'r') as f:
        for line in f:
            data = [float(x) for x in line.split()]
            spec.freq.append(data[0])
            if len(data) == 2:
                spec.val.append(data[1])
            else:
                spec.ave.append(data[1])
                spec.maxhold.append(data[2])


def peel_polarization(v):
    if 'E' in v:
        return 'E'
    if 'N' in v:
        return 'N'


def peel_time_stamp(v):
    ts = "{}-{}".format(v.split('_')[0], v.split('_')[1])
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
