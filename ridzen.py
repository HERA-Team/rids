#! /usr/bin/env python
from __future__ import print_function
import rids
import argparse
import os
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('-d', '--day_to_use', help="If just do one day.", default=None)
ap.add_argument('-p', '--path_to_save', help="Path for output files. Will create if doesn't exist.", default='data_ridzen')
ap.add_argument('-X', '--dontwrite', help="Flag to turn off file writing", action="store_true")
ap.add_argument('-n', '--num_chan', help="Number of channels.", default=1024)
args = ap.parse_args()

if not os.path.exists(args.path_to_save):
    os.mkdir(args.path_to_save)

if __name__ == '__main__':
    available_files = sorted(os.listdir('.'))
    freq = (250.0 / args.num_chan) * np.arange(num_chan)

    for af in available_files:
        ident = af.split('.')[0]
        if ident != 'zen':
            continue
        day = af.split('.')[1]
        time_stamp = '.'.join(af.split('.')[1:3])
        if args.day_to_use is not None and day != args.day_to_use:
            continue
        print(ident, time_stamp)
        df = np.load(af)
        c = np.zeros(num_chan)
        right_num_chan = True
        for a, v in df.iteritems():
            if a == 'times':
                continue
            if len(v) != args.num_chan:
                right_num_chan = False
                print("Channel number is {}, expected {}".format(len(v), num_chan))
                break
            c += v
        if not args.dontwrite and right_num_chan:
            c /= len(df['times'])
            nfn = "{}.{}.val.I".format(ident, time_stamp)
            nfn = os.path.join(args.path_to_save, nfn)
            with open(nfn, 'w') as fp:
                for fc, cc in zip(freq, c):
                    fp.write("{} {}\n".format(fc, cc))
