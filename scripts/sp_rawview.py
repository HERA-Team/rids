#! /usr/bin/env python
# _*_ coding: utf-8 _*_
# Copy 2018 the HERA Project
# Licensed under the 2-clause BSD license
from __future__ import print_function, division, absolute_import

import argparse
import os.path
import matplotlib.pyplot as plt

from rids import features

ap = argparse.ArgumentParser()
ap.add_argument('file', help="file(s) to use.  If non-RID file, will read in filenames contained in that file.", default=None)
ap.add_argument('--wf', help="plot raw_data feature components as waterfall in that file ('val', 'maxhold', or 'minhold')", default=None)
ap.add_argument('--stack', help="plot raw_data feature components as stack in that file ( '' or csv list)", default=None)
ap.add_argument('--stream', help="plot raw_data as time streams in that file ( '' or csv list)", default=None)
ap.add_argument('-f', '--f_range', help="range in freq for plots (min,max)", default=None)
ap.add_argument('-t', '--t_range', help="range in time for plots (min,max)", default=None)
ap.add_argument('--legend', help="include a legend on stack/stream plots", action='store_true')
ap.add_argument('--keys', help="plot specific keys - generally use with stack (key1,key2,...)", default=None)
ap.add_argument('--all_same_plot', help="put different feature components on same plot (not wf)", action='store_true')
ap.add_argument('--suppress_wf_gaps', help="flag to ignore time gaps in wf plot, so the time-scale doesn't match", action='store_true')
ap.add_argument('--wf_time_fill', help="value or scheme to use for missing values for suppress_wf_gaps", default='default')

args = ap.parse_args()

if ',' in args.file:
    args.file = args.file.split(',')

if args.f_range is not None:
    args.f_range = [float(x) for x in args.f_range.split(',')]

if args.t_range is not None:
    args.t_range = [float(x) for x in args.t_range.split(',')]

if args.keys is not None:
    args.keys = args.keys.split(',')

if args.wf is not None:
    args.wf = args.wf.split(',')

if args.stack is not None:
    args.stack = args.stack.split(',')

if args.stream is not None:
    args.stream = args.stream.split(',')

if args.suppress_wf_gaps:
    args.wf_time_fill = None
else:
    try:
        args.wf_time_fill = float(args.wf_time_fill)
    except ValueError:
        if args.wf_time_fill == 'default':
            print("Need to properly do this, but for now = 0.0")
            args.wf_time_fill = 0.0
        else:
            import sys
            print("Invalid wf_time_fill:  {}".format(args.wf_time_fill))
            sys.exit()

if __name__ == '__main__':
    r = features.spectrum_peak.SpectrumPeak()
    r.reader(args.file)
    s = features.sp_handling_raw.SPHandling()
    if args.wf is not None:
        s.raw_data_plot(r, args.wf, plot_type='waterfall', f_range=args.f_range, t_range=args.t_range,
                        wf_time_fill=args.wf_time_fill, keys=args.keys)
    if args.stack is not None:
        s.raw_data_plot(r, args.stack, plot_type='stack', f_range=args.f_range, t_range=args.t_range,
                        legend=args.legend, keys=args.keys, all_same_plot=args.all_same_plot)
    if args.stream is not None:
        s.raw_data_plot(r, args.stream, plot_type='stream', f_range=args.f_range, t_range=args.t_range,
                        legend=args.legend, keys=args.keys, all_same_plot=args.all_same_plot)
    plt.show()
