from __future__ import print_function, absolute_import, division
import math


def bw_finder(f, v, pts, bw_range):
    bw = [-1, 1]
    df = f[1] - f[0]  # Assumes all the same
    i_off = [int(math.ceil(bw_range[0] / df)), int(math.ceil(bw_range[1] / df))]
    for ipts in pts:
        for k in range(len(bw)):
            on_peak = True
            for j in range(1: i_off[k]):
                j_off = bw[k] * j
                if v[ipts + j_off] > 0.95 * v[ipts] and on_peak:
                    continue
                on_peak = False
                ave_v = (v[ipts + j_off + 1] + v[ipts + j_off] + v[ipts + j_off - 1]) / 3.0
                if ave_v > 0.95 * v[ipts]:
                    break
            bw[k] = j * df
    return bw
