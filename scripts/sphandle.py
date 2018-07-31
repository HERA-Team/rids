#! /usr/bin/env python
# _*_ coding: utf-8 _*_
# Copy 2018 the HERA Project
# Licensed under the 2-clause BSD license
from __future__ import print_function, division, absolute_import

import argparse
import os.path
import matplotlib.pyplot as plt

from rids import spectrum_peak as sp

ap = argparse.ArgumentParser()
ap.add_argument('file', help="file(s) to use", default=None)
ap.add_argument('--rdwf', help="plot raw_data as waterfall in that file ('val', 'maxhold', or 'minhold')", default=None)
ap.add_argument('--rdstack', help="plot raw_data as stack in that file ('val', 'maxhold', or 'minhold')", default=None)

args = ap.parse_args()

if __name__ == '__main__':
    r = sp.spectrum_peak.SpectrumPeak()
    r.reader(args.file)
    s = sp.sp_handling.SPHandling()
    if args.rdwf is not None:
        s.raw_data_plot(r, args.rdwf, plot_type='waterfall')
        plt.show()
    if args.rdstack is not None:
        s.raw_data_plot(r, args.rdstack, plot_type='stack')
        plt.show()
