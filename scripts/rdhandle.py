#! /usr/bin/env python
# _*_ coding: utf-8 _*_
# Copy 2018 the HERA Project
# Licensed under the 2-clause BSD license
from __future__ import print_function, division, absolute_import

import argparse
import os.path
import matplotlib.pyplot as plt

from rids import spectrum_peak as sp

ap = argparse.ArgumentParser()
ap.add_argument('file', help="file(s) to use", default=None)
ap.add_argument('--wf', help="plot raw_data as waterfall in that file ('val', 'maxhold', or 'minhold')", default=None)
ap.add_argument('--stack', help="plot raw_data as stack in that file ('val', 'maxhold', or 'minhold')", default=None)
ap.add_argument('--stream', help="plot raw_data as time streams in that file ('val', 'maxhold, or 'minhold')", default=None)
ap.add_argument('-f', '--f_range', help="range in freq for plots (min,max)", default=None)
ap.add_argument('-t', '--t_range', help="range in time for plots (min,max)", default=None)
ap.add_argument('--legend', help="include a legend on stack/stream plots", action='store_true')
ap.add_argument('--keys', help="plot specific keys - generally use with stack (key1,key2,...)", default=None)

args = ap.parse_args()

if args.f_range is not None:
    args.f_range = [float(x) for x in args.f_range.split(',')]

if args.t_range is not None:
    args.t_range = [float(x) for x in args.t_range.split(',')]

if args.keys is not None:
    args.keys = args.keys.split(',')

if __name__ == '__main__':
    r = sp.spectrum_peak.SpectrumPeak()
    r.reader(args.file)
    s = sp.sp_handling.SPHandling()
    if args.wf is not None:
        s.raw_data_plot(r, args.wf, plot_type='waterfall', f_range=args.f_range, t_range=args.t_range, keys=args.keys)
    if args.stack is not None:
        s.raw_data_plot(r, args.stack, plot_type='stack', f_range=args.f_range, t_range=args.t_range, legend=args.legend, keys=args.keys)
    if args.stream is not None:
        s.raw_data_plot(r, args.stream, plot_type='stream', f_range=args.f_range, t_range=args.t_range, legend=args.legend, keys=args.keys)
    plt.show()
