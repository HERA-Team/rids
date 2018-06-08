#! /usr/bin/env python

import rids
import argparse
import os.path

ap = argparse.ArgumentParser()
ap.add_argument('-e', '--ecal', help="E-pol cal filename", default=None)
ap.add_argument('-n', '--ncal', help="N-pol cal filename", default=None)
ap.add_argument('-p', '--parameters', help="parameter/header json filename", default=None)
ap.add_argument('-#', '--nevents', help="number of events per pol per file", default=100)
ap.add_argument('-d', '--directory', help="directory for process files and where parameter file lives", default='.')
ap.add_argument('-i', '--ident', help="can be a specific ident or 'all'", default='all')
ap.add_argument('-c', '--comment', help="append a comment", default=None)
ap.add_argument('-m', '--max_loops', help="maximum number of iteration loops", default=1000)
ap.add_argument('-s', '--show_info', help="show the info for provided filename", default=None)
ap.add_argument('-v', '--view', help="show plot for provided filename", default=None)
ap.add_argument('-b', '--baselines', help="Indices for baseline in csv-list", default='0,-1')
args = ap.parse_args()
args.nevents = int(args.nevents)
a = args.baselines.split(',')
args.baselines = [int(x) for x in a]

if __name__ == '__main__':
    r = rids.Rids()
    if args.show_info is not None:
        r.reader(args.show_info)
        r.info()
    elif args.view is not None:
        import matplotlib.pyplot as plt
        r.reader(args.view)
        r.info()
        r.viewer()
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
        r.process_files(args.directory, args.ident, args.baselines, args.nevents, args.max_loops)
