#! /usr/bin/env python
# _*_ coding: utf-8 _*_
# Copy 2018 the HERA Project
# Licensed under the 2-clause BSD license
from __future__ import print_function, division, absolute_import

import argparse
import os.path
import matplotlib.pyplot as plt
import datetime

from rids import features
ap = argparse.ArgumentParser()

ap.add_argument('file', help="file(s) to use.  If non-RID file, will read in filenames contained in that file.", default=None)
ap.add_argument('-p', '--pol', help="polarization to use", default=None)
ap.add_argument('-w', '--wf', help="plot raw_data feature components as waterfall in that file ('val', 'maxhold', or 'minhold')", default=None)
ap.add_argument('--stack', help="plot raw_data feature components as stack in that file ( '' or csv list)", default=None)
ap.add_argument('--stream', help="plot raw_data as time streams in that file ( '' or csv list)", default=None)
ap.add_argument('--totalpower', help="plot total power (specify dB/dBv/linear/linearv -- not assumes dB)", default=None)
ap.add_argument('-f', '--f-range', dest='f_range', help="range in freq for plots (min,max)", default=None)
ap.add_argument('-l', '--legend', help="include a legend on stack/stream plots", action='store_true')
ap.add_argument('-k', '--keys', help="plot specific keys - generally use with stack (key1,key2,...)", default=None)
ap.add_argument('--title', help="plot title", default=None)
ap.add_argument('--all-same-plot', dest='all_same_plot', help="put different feature components on same plot (not wf)", action='store_true')
ap.add_argument('--wf-fill', dest='wf_time_fill', help="value or scheme to use for missing values if showing wf_gaps", default='default')
ap.add_argument('--show-edits', dest='show_edits', help="Flag to display info on what was needed to make arrays same length.", action='store_true')
ap.add_argument('--flip', dest='flip_range', help="Flag to plot converse of t_range", action='store_true')
ap.add_argument('--csv', help="Save data array to csv file (<fc>_out.csv).", action='store_true')
# Only used in script
ap.add_argument('--dB', help="convert to dB (assume values are linear power)", action='store_true')
ap.add_argument('--linear', help="convert to linear power (assume values are dB)", action='store_true')
ap.add_argument('--dBv', help="convert to dB (assume values are linear volts)", action='store_true')
ap.add_argument('--linearv', help="convert to linear voltage (assume values are dB)", action='store_true')
ap.add_argument('-0', '--start-time', dest='start_time', help="start-time to use:  YY-MM-DD.HH:MM[:SS] - default is data start", default=None)
ap.add_argument('-1', '--stop-time', dest='stop_time', help="stop-time to use:  YY-MM-DD.HH:MM[:SS] - default is data end", default=None)
ap.add_argument('--hide-gaps', dest='wf_gaps', help="flag to ignore time gaps in wf plot [the time-scale won't match", action='store_false')

args = ap.parse_args()

args.file = args.file.split(',')

if args.pol is None:
    raise ValueError("polarization must be set.")

if args.f_range is not None:
    args.f_range = [float(x) for x in args.f_range.split(',')]

if args.start_time is not None:
    if args.start_time.count(':') == 1:
        args.start_time += ':00'
    args.start_time = datetime.datetime.strptime(args.start_time, '%y-%m-%d.%H:%M:%S')

if args.stop_time is not None:
    if args.stop_time.count(':') == 1:
        args.stop_time += ':00'
    args.stop_time = datetime.datetime.strptime(args.stop_time, '%y-%m-%d.%H:%M:%S')

t_range = [args.start_time, args.stop_time]

unit_conversion = 'none'
if args.dB:
    unit_conversion = 'dB'
if args.linear:
    unit_conversion = 'linear'
if args.dBv:
    unit_conversion = 'dBv'
if args.linearv:
    unit_conversion = 'linearv'

if args.csv:
    args.csv = 'out.csv'

if args.keys is not None:
    args.keys = args.keys.split(',')

if args.wf is not None:
    args.wf = args.wf.split(',')

if args.stack is not None:
    args.stack = args.stack.split(',')

if args.stream is not None:
    args.stream = args.stream.split(',')

if args.totalpower is not None:
    args.totalpower = args.totalpower.split(',')

if args.wf_gaps:
    try:
        args.wf_time_fill = float(args.wf_time_fill)
    except ValueError:
        if args.wf_time_fill == 'default':
            args.wf_time_fill = 0.0
        else:
            import sys
            print("Invalid wf_time_fill:  {}".format(args.wf_time_fill))
            sys.exit()
else:
    args.wf_time_fill = None

# Read in data
r = features.spectrum_peak.SpectrumPeak()
r.reader(args.file)
# Set up data parameters for plot
s = features.sp_handling_raw.SPHandling(r)
s.set_feature_keys(pol=args.pol, keys=args.keys)
s.set_freq(f_range=args.f_range)
s.set_time_range(t_range=t_range, flip=args.flip_range)
# Plot it
if args.wf is not None:
    s.time_filter(args.wf)
    s.process(wf_time_fill=args.wf_time_fill, show_edits=args.show_edits, unit_conversion=unit_conversion, csv=args.csv)
    s.raw_waterfall_plot(title=args.title)
if args.stack is not None:
    s.time_filter(args.stack)
    s.process(show_edits=args.show_edits, unit_conversion=unit_conversion, csv=args.csv)
    s.raw_2D_plot(plot_type='stack', legend=args.legend, all_same_plot=args.all_same_plot, title=args.title)
if args.stream is not None:
    s.time_filter(args.stream)
    s.process(show_edits=args.show_edits, unit_conversion=unit_conversion, csv=args.csv)
    s.raw_2D_plot(plot_type='stream', legend=args.legend, all_same_plot=args.all_same_plot, title=args.title)
if args.totalpower is not None:
    s.time_filter(args.totalpower)
    s.process(show_edits=args.show_edits, total_power_only=True, unit_conversion=unit_conversion, csv=args.csv)
    s.raw_totalpower_plot(title=args.title)
plt.show()
