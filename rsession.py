#! /usr/bin/env python

import rids
import argparse
import os.path

ap = argparse.ArgumentParser()
ap.add_argument('-e', '--ecal', help="E-pol cal filename", default=None)
ap.add_argument('-n', '--ncal', help="N-pol cal filename", default=None)
ap.add_argument('-p', '--parameters', help="parameter/header json filename", default=None)
ap.add_argument('-f', '--events_per_file', help="number of events per file", default=100)
ap.add_argument('-d', '--directory', help="directory for process files and where parameter file lives", default='.')
ap.add_argument('-i', '--ident', help="can be a specific ident or 'all'", default='all')
ap.add_argument('-c', '--comment', help="append a comment", default=None)
ap.add_argument('-m', '--max_loops', help="maximum number of iteration loops", default=1000)
args = ap.parse_args()
args.events_per_file = int(args.events_per_file)

if __name__ == '__main__':
    r = rids.Rids()
    r.reader(os.path.join(args.directory, args.parameters))
    r.append_comment(args.comment)
    if args.ecal is not None:
        rids.spectrum_reader(args.ecal, r.cal['E'])
    if args.ncal is not None:
        rids.spectrum_reader(args.ncal, r.cal['N'])
    r.process_files(args.directory, args.ident, args.events_per_file, args.max_loops)
