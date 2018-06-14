from __future__ import print_function, absolute_import, division
import math
import numpy as np


def bw_finder(f, v, pts, bw_range):
    bw = []
    df = f[1] - f[0]  # Assumes all the same
    i_off = [int(math.ceil(bw_range[0] / df)), int(math.ceil(bw_range[1] / df))]
    v_zero = np.array(v) - min(v)
    for ipts in pts:
        bw_entry = [0.0, 0.0]
        for k in range(len(bw_entry)):
            on_peak = True
            bw_dir = int(abs(i_off[k]) / i_off[k])
            for j in range(1, abs(i_off[k])):
                off1 = ipts + bw_dir * j
                off2 = ipts + bw_dir * (j - 1)
                comp = (v_zero[off1] - v_zero[off2]) / v_zero[off2]
                if comp < -0.5:
                    break
                if comp > -0.02 and on_peak:
                    continue
                on_peak = False
                if comp > -0.02:
                    break
            bw_entry[k] = j * df * bw_dir
        bw.append(bw_entry)
    return bw
