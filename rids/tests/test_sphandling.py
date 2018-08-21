# -*- mode: python; coding: utf-8 -*-
# Copyright 2016 the HERA Collaboration
# Licensed under the 2-clause BSD license.

from __future__ import print_function, division, absolute_import

from .. import features
import matplotlib.pyplot as plt


def test_reconstitute_params():
    rid = features.spectrum_peak.SpectrumPeak()
    rid.reader('sa_Spectrum_Peak.20180526-1033.n40.maxT-20.ridz')
    fs = '20180527-0116.E'
    rcf = features.sp_handling.SPHandling()
    fc2use = 'maxhold'

    print("<defaults>")
    rcf.reconstitute_params(rid, fs, fc2use)
    print("{}\n\t{}".format(fc2use, rcf.reconstituted_info))

    # dfill
    print('\n<dfill>')
    dfill_types = ['feature_set_min', 'component_min', 'maxhold', 'minhold', 'val', -20.0, '-30.0']
    for dfy in dfill_types:
        rcf.reconstitute_params(rid, fs, fc2use, dfill=dfy)
        print("{}, {}\n\t{}".format(dfy, fc2use, rcf.reconstituted_info))
    # fstep
    print('\n<fstep>')
    fstep_types = ['channel', 0.1, '0.1']
    for ffy in fstep_types:
        rcf.reconstitute_params(rid, fs, fc2use, fstep=ffy)
        print("{}\n\t{}".format(ffy, rcf.reconstituted_info))


def test_reconstitute_features():
    rid = features.spectrum_peak.SpectrumPeak()
    rid.reader('sa_Spectrum_Peak.20180526-1033.n40.maxT-20.ridz')
    rcf = features.sp_handling.SPHandling()
    fs = '20180526-1033.E'
    fc2use = 'maxhold'
    rcf.reconstitute_features(rid, fs, fc2use, dfill='component_min')
    rcf.reconstitute_plot(fs)
    dfs = 'data.20180526-1033.E'
    rcf.reconstitute_features(rid, dfs, fc2use, dfill='component_min')
    rcf.reconstitute_plot(fs, ptype='spec')
    plt.show()
