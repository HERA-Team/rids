#! /usr/bin/env python

import rids
import rids_utils as utils
import argparse
import os.path

ap = argparse.ArgumentParser()
ap.add_argument('-e', '--ecal', help="E-pol cal filename", default=None)
ap.add_argument('-n', '--ncal', help="N-pol cal filename", default=None)
ap.add_argument('-p', '--parameters', help="parameter/header json filename", default=None)
ap.add_argument('-o', '--obs_per_file', help="number of obs groups per file", default=100)
ap.add_argument('-d', '--directory', help="directory for process files and where parameter file lives", default='.')
ap.add_argument('-k', '--keep_completed', help="keep files after processing them", action='store_true')
ap.add_argument('-c', '--comment', help="append a comment", default=None)
args = ap.parse_args()

if __name__ == '__main__':
    r = rids.Rids()
    r.json_reader(os.path.join(args.directory, args.parameters))
    r.append_comment(args.comment)
    if args.ecal is not None:
        utils.spectrum_reader(args.ecal, r.cal['E'])
    if args.ncal is not None:
        utils.spectrum_reader(args.ncal, r.cal['N'])
    r.process_files(args.directory, args.obs_per_file, args.keep_completed)
