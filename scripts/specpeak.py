#! /usr/bin/env python
# _*_ coding: utf-8 _*_
# Copy 2018 the HERA Project
# Licensed under the 2-clause BSD license
from __future__ import print_function, division, absolute_import

import argparse
import os.path

from rids import spectrum_peak as sp

ap = argparse.ArgumentParser()
ap.add_argument('parameters', help="parameter/header json filename", default=None)
ap.add_argument('--id', help="can be a specific id name or 'all'", default='all')
ap.add_argument('-e', '--ecal', help="E-pol cal filename", default=None)
ap.add_argument('-n', '--ncal', help="N-pol cal filename", default=None)
ap.add_argument('-#', '--sets_per_pol', help="number of sets per pol per file", default=10000)
ap.add_argument('-c', '--comment', help="append a comment", default=None)
ap.add_argument('-r', '--rawdata', help="csv indices for raw data to keep, or +step ('n' to stop if view)", default='0,-1')
ap.add_argument('-i', '--info', help="show the info for provided filename", action="store_true")
ap.add_argument('-v', '--view', help="show plot for provided filename", action="store_true")
ap.add_argument('--show_fc', help="csv list of feature components to show (if different)", default='all')
ap.add_argument('--share_freq', help="if you know all spectra have same freq axis, set to True", action="store_true")
ap.add_argument('--peak_on', help="Peak on event component (if other than default)", default=None)
ap.add_argument('--view_peaks_ongoing', help="view all peaks in process (diagnostic only!)", action="store_true")
ap.add_argument('--directory', help="directory for process files and where parameter file lives", default='.')
ap.add_argument('--threshold_view', help="new threshold for viewing (if possible)", default=None)
ap.add_argument('--max_loops', help="maximum number of iteration loops", default=1000)
ap.add_argument('--data_only', help="flag to only store data and not peaks", action='store_true')
ap.add_argument('--data_only_override', help="flag to force data_only without saving all", action='store_true')
ap.add_argument('--archive_data', help="Flag to archive all data (shortcut for data_only=True and rawdata='+1').", action='store_true')
ap.add_argument('--show_keys', help="Show the feature_set keys", action='store_true')

args = ap.parse_args()
if args.archive_data:
    args.data_only = True
    args.rawdata = '+1'
if args.data_only and args.rawdata != '+1' and not args.data_only_override:
    print("Warning:  This will delete the data but not save all of the raw or processed data.")
    raise ValueError("If this is desired then rerun with flag --data_only_override")

args.max_loops = int(args.max_loops)
args.sets_per_pol = int(args.sets_per_pol)
if args.view:
    if args.rawdata[0].lower() == 'n':
        args.rawdata = False
    else:
        args.rawdata = True
elif args.rawdata[0] == '+':
    step = int(args.rawdata[1:])
    args.rawdata = range(0, args.sets_per_pol, step)
    if (args.sets_per_pol - args.rawdata[-1]) >= step:
        args.rawdata.append(-1)
else:
    a = args.rawdata.split(',')
    args.rawdata = [int(x) for x in a]
if args.show_fc.lower() != 'all':
    args.show_fc = args.show_fc.split(',')
if args.threshold_view is not None:
    args.threshold_view = float(args.threshold_view)
full_filename = os.path.join(args.directory, args.parameters)

if __name__ == '__main__':
    r = sp.spectrum_peak.SpectrumPeak(share_freq=args.share_freq, view_ongoing=args.view_peaks_ongoing)
    if args.info:
        r.reader(full_filename, reset=False)
        r.info()
    elif args.view:
        r.reader(full_filename, reset=False)
        r.info()
        r.viewer(threshold=args.threshold_view, show_components=args.show_fc, show_data=args.rawdata)
    elif args.show_keys:
        r.reader(full_filename, reset=False)
        for k in r.feature_sets.keys():
            print(k)
    else:
        if '.' not in args.parameters:
            full_filename += '.ridm'
        r.reader(full_filename, reset=False)
        r.append_comment(args.comment)
        if args.ecal is not None:
            r.read_cal(args.ecal, 'E')
        if args.ncal is not None:
            r.read_cal(args.ncal, 'N')
        r.process_files(directory=args.directory,
                        ident=args.id,
                        data=args.rawdata,
                        peak_on=args.peak_on,
                        data_only=args.data_only,
                        sets_per_pol=args.sets_per_pol,
                        max_loops=args.max_loops)
