#! /usr/bin/env python
from __future__ import print_function
import rids
import argparse
import os
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('-d', '--day_to_use', help="If just do one day.", default=None)
ap.add_argument('-x', '--dontwrite', help="Flag to turn off file writing", action="store_true")
args = ap.parse_args()

if __name__ == '__main__':
    available_files = sorted(os.listdir('.'))
    num_chan = 1024
    freq = (250.0 / num_chan) * np.arange(num_chan)

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
            if len(v) != num_chan:
                right_num_chan = False
                print("Channel number is {}, expected {}".format(len(v), num_chan))
                break
            c += v
        if not args.dontwrite and right_num_chan:
            c /= len(df['times'])
            nfn = "dataout/{}.{}.val.I".format(ident, time_stamp)
            with open(nfn, 'w') as fp:
                for fc, cc in zip(freq, c):
                    fp.write("{} {}\n".format(fc, cc))
