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
ap.add_argument('-w', '--wf', help="plot raw_data feature components as waterfall in that file ('val', 'maxhold', or 'minhold')", default=None)
ap.add_argument('--stack', help="plot raw_data feature components as stack in that file ( '' or csv list)", default=None)
ap.add_argument('--stream', help="plot raw_data as time streams in that file ( '' or csv list)", default=None)
ap.add_argument('-f', '--f_range', help="range in freq for plots (min,max)", default=None)
ap.add_argument('-t', '--t_range', help="range in time for plots (min,max)", default=None)
ap.add_argument('-l', '--legend', help="include a legend on stack/stream plots", action='store_true')
ap.add_argument('-k', '--keys', help="plot specific keys - generally use with stack (key1,key2,...)", default=None)
ap.add_argument('--all_same_plot', help="put different feature components on same plot (not wf)", action='store_true')
ap.add_argument('--suppress_wf_gaps', help="flag to ignore time gaps in wf plot [the time-scale won't match", action='store_true')
ap.add_argument('--wf_time_fill', help="value or scheme to use for missing values if not suppress_wf_gaps", default='default')
ap.add_argument('--hide-edits', dest='show_edits', help="Flag to display info on what was needed to make arrays same length.", action='store_false')

args = ap.parse_args()

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
    # Read in data
    r = features.spectrum_peak.SpectrumPeak()
    r.reader(args.file)
    # Set up data parameters for plot
    s = features.sp_handling_raw.SPHandling(r)
    s.set_feature_keys(keys=args.keys)
    s.set_freq(f_range=args.f_range)
    s.set_time_range(t_range=args.t_range)
    s.time_filter()
    # Plot it
    if args.wf is not None:
        s.process(args.wf, wf_time_file=args.wf_time_fill, show_edits=args.show_edits)
        s.raw_waterfall_plot()
    if args.stack is not None:
        s.process(args.stack, show_edits=args.show_edits)
        s.raw_2D_plot(plot_type='stack', legend=args.legend, all_same_plot=args.all_same_plot)
    if args.stream is not None:
        s.process(args.stream, show_edits=args.show_edits)
        s.raw_2D_plot(plot_type='stream', legend=args.legend, all_same_plot=args.all_same_plot)
    plt.show()
