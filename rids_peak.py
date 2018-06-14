#! /usr/bin/env python
from __future__ import print_function, division, absolute_import

import argparse
import os.path

import spectrum_peak as sp

ap = argparse.ArgumentParser()
ap.add_argument('parameters', help="parameter/header json filename", default=None)
ap.add_argument('-e', '--ecal', help="E-pol cal filename", default=None)
ap.add_argument('-n', '--ncal', help="N-pol cal filename", default=None)
ap.add_argument('-#', '--nsets', help="number of sets per pol per file", default=100)
ap.add_argument('-i', '--ident', help="can be a specific ident or 'all'", default='all')
ap.add_argument('-c', '--comment', help="append a comment", default=None)
ap.add_argument('-^', '--peak_on', help="Peak on event component (if other than default)", default=None)
ap.add_argument('-r', '--rawdata', help="csv indices for raw data to keep, or +step ('n' to stop if view)", default='0,-1')
ap.add_argument('-s', '--show_info', help="show the info for provided filename", action="store_true")
ap.add_argument('-v', '--view', help="show plot for provided filename", action="store_true")
ap.add_argument('-f', '--show_fc', help="csv list of event components to show (if different)", default='all')
ap.add_argument('--view_peaks_ongoing', help="view all peaks in process (diagnostic)", action="store_true")
ap.add_argument('--directory', help="directory for process files and where parameter file lives", default='.')
ap.add_argument('--threshold_view', help="new threshold for viewing (if possible)", default=None)
ap.add_argument('--max_loops', help="maximum number of iteration loops", default=1000)

args = ap.parse_args()
args.max_loops = int(args.max_loops)
args.nsets = int(args.nsets)
if args.view:
    if args.rawdata[0].lower() == 'n':
        args.rawdata = False
    else:
        args.rawdata = True
elif args.rawdata[0] == '+':
    step = int(args.rawdata[1:])
    args.rawdata = range(0, args.nsets, step)
    if (args.nsets - args.rawdata[-1]) >= step:
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
    r = sp.SpectrumPeak(view_ongoing=args.view_peaks_ongoing)
    if args.show_info:
        r.rids.reader(full_filename)
        r.rids.info()
    elif args.view:
        import matplotlib.pyplot as plt
        r.rids.reader(full_filename)
        r.rids.info()
        r.viewer(threshold=args.threshold_view, show_components=args.show_fc, show_rawdata=args.rawdata)
        plt.show()
    else:
        r.rids.reader(full_filename)
        r.rids.append_comment(args.comment)
        if args.ecal is not None:
            r.read_cal(args.ecal, 'E')
        if args.ncal is not None:
            r.read_cal(args.ncal, 'N')
        r.process_files(args.directory, args.ident, args.rawdata, args.peak_on, args.nsets, args.max_loops)
