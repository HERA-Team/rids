#! /usr/bin/env python

import rids
import argparse
import os.path

ap = argparse.ArgumentParser()
ap.add_argument('parameters', help="parameter/header json filename", default=None)
ap.add_argument('-e', '--ecal', help="E-pol cal filename", default=None)
ap.add_argument('-n', '--ncal', help="N-pol cal filename", default=None)
ap.add_argument('-#', '--nevents', help="number of events per pol per file", default=100)
ap.add_argument('-d', '--directory', help="directory for process files and where parameter file lives", default='.')
ap.add_argument('-i', '--ident', help="can be a specific ident or 'all'", default='all')
ap.add_argument('-c', '--comment', help="append a comment", default=None)
ap.add_argument('-m', '--max_loops', help="maximum number of iteration loops", default=1000)
ap.add_argument('-^', '--peak_on', help="Peak on event component (if other than default)", default=None)
ap.add_argument('-b', '--baselines', help="csv indices for baselines, or +step ('n' to stop if view)", default='0,-1')
ap.add_argument('-s', '--show_info', help="show the info for provided filename", action="store_true")
ap.add_argument('-v', '--view', help="show plot for provided filename", action="store_true")
ap.add_argument('-t', '--threshold_view', help="new threshold for viewing (if possible)", default=None)
ap.add_argument('-@', '--show_ec', help="csv list of event components to show (if different)", default='all')
ap.add_argument('-!', '--view_peaks_on_event', help="view all peaks in process (diagnostic)", action="store_true")

args = ap.parse_args()
args.max_loops = int(args.max_loops)
args.nevents = int(args.nevents)
if args.view:
    if args.baselines[0].lower() == 'n':
        args.baselines = False
    else:
        args.baseline = True
elif args.baselines[0] == '+':
    step = int(args.baselines[1:])
    args.baselines = range(0, args.nevents, step)
    if (args.nevents - args.baselines[-1]) >= step:
        args.baselines.append(-1)
else:
    a = args.baselines.split(',')
    args.baselines = [int(x) for x in a]
if args.show_ec.lower() != 'all':
    args.show_ec = args.show_ec.split(',')
if args.threshold_view is not None:
    args.threshold_view = float(args.threshold_view)

if __name__ == '__main__':
    r = rids.Rids(view_peaks_on_event=args.view_peaks_on_event)
    if args.show_info:
        r.reader(args.parameters)
        r.info()
    elif args.view:
        import matplotlib.pyplot as plt
        r.reader(args.parameters)
        r.info()
        r.viewer(threshold=args.threshold_view, show_components=args.show_ec, show_baseline=args.baselines)
        plt.show()
    else:
        r.reader(os.path.join(args.directory, args.parameters))
        r.append_comment(args.comment)
        if args.ecal is not None:
            r.cal['E'] = Rids.Spectral('E')
            rids.spectrum_reader(args.ecal, r.cal['E'])
        if args.ncal is not None:
            r.cal['N'] = Rids.Spectral('N')
            rids.spectrum_reader(args.ncal, r.cal['N'])
        r.process_files(args.directory, args.ident, args.baselines, args.peak_on, args.nevents, args.max_loops)
