#! /usr/bin/env python
# _*_ coding: utf-8 _*_
# Copy 2018 the HERA Project
# Licensed under the 2-clause BSD license
"""
Script used to generate SpectrumPeak ridz files, as well as some low-level info and viewing of existing files.
"""
from __future__ import print_function, division, absolute_import

import argparse
import os.path

from rids import features

ap = argparse.ArgumentParser()
ap.add_argument('rid_filename', help="rids meta-data filename or filename to be viewed"
                "(note:  type 'fnhelp' to see format of spectrum filenames)", default=None)
ap.add_argument('--directory', help="directory for process files and where parameter/rids file lives", default='.')

# parameters used only in script
ap.add_argument('-i', '--info', help="show the info for provided filename", action="store_true")
ap.add_argument('-v', '--view', help="show plot for provided filename", action="store_true")
ap.add_argument('-k', '--keys', help="Show the feature_set keys", action='store_true')
ap.add_argument('--archive-data', dest='archive_data', help="Flag to archive all data"
                "(shortcut for data_only=True and rawdata='+1').", action='store_true')
ap.add_argument('--data_only_override', help="flag to force data_only without saving all", action='store_true')

# parameters used for both generate and view
ap.add_argument('-r', '--rawdata', help="csv indices for raw data to keep, or +step ('n' if skip in view mode)", default='0,-1')
ap.add_argument('-c', '--comment', help="append a comment or includes comments in keys printout if set", nargs='?', const='view', default=None)

# parameters just used for generate
ap.add_argument('--id', help="can be a specific id name or 'all'", default='all')
ap.add_argument('--ptype', help="data processing file type (if known):  'spfile'/'poco' (None)", default=None)
ap.add_argument('--keep-data', dest='keep_data', help="keep data after processing", action='store_true')
ap.add_argument('-#', '--sets-per-pol', dest='sets_per_pol', help="number of sets per pol per file", default=10000)
ap.add_argument('--share-freq', dest='share_freq', help="invoke if you know all spectra have same freq axis", action="store_true")
ap.add_argument('--peak-on', dest='peak_on', help="Peak on event component (if other than max->min->val)", default=None)
ap.add_argument('--view_peaks_ongoing', help="view all peaks in process (diagnostic only!)", action="store_true")
ap.add_argument('--show-progress', dest='show_progress', help="print file name as processed", action='store_true')
ap.add_argument('--data-only', dest='data_only', help="flag to only store data and not peaks", action='store_true')
ap.add_argument('--ecal', help="E-pol cal filename", default=None)
ap.add_argument('--ncal', help="N-pol cal filename", default=None)

# parameters only used in viewing info on existing
ap.add_argument('--show_fc', help="csv list of feature components to show (if different)", default='all')
ap.add_argument('--threshold_view', help="new threshold for viewing (if possible)", default=None)

args = ap.parse_args()
if args.rid_filename == 'fnhelp':
    print(features.spectrum_peak._peel_filename(v='filename_format_help'))
    raise SystemExit

if args.archive_data:
    args.data_only = True
    args.rawdata = '+1'
if not args.keep_data and args.data_only and args.rawdata != '+1' and not args.data_only_override:
    print("Warning:  This will delete the data but not save all of the raw or processed data.")
    raise ValueError("If this is desired then rerun as same but adding flag '--data_only_override'")

args.sets_per_pol = int(args.sets_per_pol)
if args.view:
    if args.rawdata[0].lower() == 'n':
        args.rawdata = False
    else:
        args.rawdata = True
elif args.rawdata[0] == '+':
    step = int(args.rawdata[1:])
    args.rawdata = list(range(0, args.sets_per_pol, step))
else:
    a = args.rawdata.split(',')
    args.rawdata = [int(x) for x in a]
if args.show_fc.lower() != 'all':
    args.show_fc = args.show_fc.split(',')
if args.threshold_view is not None:
    args.threshold_view = float(args.threshold_view)
full_filename = os.path.join(args.directory, args.rid_filename)

r = features.spectrum_peak.SpectrumPeak(share_freq=args.share_freq, view_ongoing=args.view_peaks_ongoing)
r.reader(full_filename, reset=False)
if args.info:
    r.info()
elif args.view:
    r.info()
    r.viewer(threshold=args.threshold_view, show_components=args.show_fc, show_data=args.rawdata)
elif args.keys:
    print("\nFeature set keys:")
    for k in r.feature_sets.keys():
        if args.comment is None:
            s = '['
            for x in dir(r.feature_sets[k]):
                if x[0] != '_':
                    s += '{}, '.format(x)
            s = s.strip().strip(',') + ']'
            print("\t{}  {}".format(k, s))
        else:
            print('---{}---'.format(k))
            if len(r.feature_sets[k].comment):
                print('{}'.format(r.feature_sets[k].comment))
else:
    r.append_comment(args.comment)
    if args.ecal is not None:
        r.read_cal(args.ecal, 'E')
    if args.ncal is not None:
        r.read_cal(args.ncal, 'N')
    r.process_files(ident=args.id,
                    ptype=args.ptype,
                    directory=args.directory,
                    data=args.rawdata,
                    peak_on=args.peak_on,
                    data_only=args.data_only,
                    sets_per_pol=args.sets_per_pol,
                    keep_data=args.keep_data,
                    show_progress=args.show_progress)
