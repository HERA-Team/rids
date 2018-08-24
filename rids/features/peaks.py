# -*- coding: utf-8 -*-
from __future__ import print_function, absolute_import, division
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import os.path


def procfiles(files_to_proc, threshold=-15.0, cwt_range=[1, 7], rc_range=[4, 4], f_range=[20.0, 300.0],
              title_prefix='', dir=None, saveit=False, yrange='minmax'):
    fmin = f_range[0]
    fmax = f_range[1]
    if isinstance(files_to_proc, (str, unicode)):
        files_to_proc = [files_to_proc]
    for fn in files_to_proc:
        fno = fn
        if dir is not None:
            fno = os.path.join(dir, fn)
        a = np.loadtxt(fno)
        xs = a[:, 0]
        ys = a[:, 1]
        xlimg = np.where(xs > fmin)[0]
        xlim = xlimg[np.where(xs[xlimg] < fmax)]
        x = xs[xlim]
        y = ys[xlim]
        hipk = fp(y, threshold, cwt_range, rc_range)

        plt.figure(fn)
        plt.xlabel('MHz')
        plt.ylabel('dBm')
        plot_title = title_prefix + os.path.basename(fn).split('.')[0]
        plt.title(plot_title)
        # plt.plot(xs[pkin], ys[pkin], 'o')
        plt.plot([fmin, fmin], [min(y), max(y)], 'b--')
        plt.plot([fmax, fmax], [min(y), max(y)], 'b--')
        plt.plot([fmin, fmax], [threshold, threshold], 'b:')
        plt.plot(x, y)
        if isinstance(yrange, (str, unicode)):
            ymin = min(y)
            ymax = max(y)
        else:
            ymin = yrange[0]
            ymax = yrange[1]
        plt.axis([fmin, fmax + 75.0, ymin, ymax + 1.0])
        plt.plot(x[hipk], y[hipk], 'kv')
        print(" MHz      dBm")
        print("------   ------")
        plt.text(fmax, ymax - 0.5, ' MHz      dBm')
        plt.text(fmax, ymax - 0.75, '_______   ______')
        yoff = -2.5
        for xp, yp in zip(x[hipk], y[hipk]):
            s = "{:>6.2f}  {:>6.2f}".format(xp, yp)
            yval = ymax + yoff
            yoff -= 1.5
            plt.text(fmax, yval, s)
            print("{:>6.2f}  {:>6.2f}".format(xp, yp))
        if saveit:
            ofn = fn.split('.')[0] + '.png'
            plt.savefig(ofn, dpi=200)


def fp(y, threshold=-15.0, cwt_range=[1, 7], rc_range=[4, 4]):
    pkin = signal.find_peaks_cwt(y, np.arange(cwt_range[0], cwt_range[1]))
    hipk = recenter_and_threshold(y, pkin, [4, 4], threshold)
    return hipk


def recenter_and_threshold(y, pkin, rng, threshold):
    pkout = []
    dirout = []
    for p in pkin:
        dir = None
        newp = p
        for r in range(1, rng[0] + 1):
            if p - r < 0:
                break
            if y[p - r] > y[newp]:
                newp = p - r
                dir = 'left'
            else:
                break
        if dir is None:
            for r in range(1, rng[0] + 1):
                if p + r > len(y) - 1:
                    break
                if y[p + r] > y[newp]:
                    newp = p + r
                    dir = 'right'
                else:
                    break

        if (y[newp] > threshold) and (newp not in pkout):
            pkout.append(newp)
            dirout.append(dir)
    return pkout


def ps(fns, fmin=20.0, fmax=300.0):
    for fn in fns:
        a = np.loadtxt(fn)
        lbl = fn.split('_')[0] + '-' + fn.split('_')[1]
        xs = a[:, 0]
        ys = a[:, 1]

        xlimx = np.where(xs > fmin)[0]
        xlim = xlimx[np.where(xs[xlimx] < fmax)]
        plt.plot(xs[xlim], ys[xlim], label=lbl)

    plt.legend()
