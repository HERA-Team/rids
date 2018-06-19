# -*- mode: python; coding: utf-8 -*-
# Copyright 2016 the HERA Collaboration
# Licensed under the 2-clause BSD license.

from __future__ import print_function, division, absolute_import

from rids import spectrum_peak as sp


def test_reconstitute_params():
    rid = sp.spectrum_peak.SpectrumPeak()
    rid.reader('sa_Spectrum_Peak.20180526-1033.n40.maxT-20.ridz')
    fs = '20180527-0116.E'
    rcf = sp.sp_handling.SPHandling()
    fc2use = 'maxhold'

    print("<defaults>")
    rfpar = rcf.reconstitute_params(rid.rids, fs, fc2use)
    print("{}\n\t{}".format(fc2use, rfpar))

    # dfill
    print('\n<dfill>')
    dfill_types = ['feature_set_min', 'component_min', 'maxhold', 'minhold', 'val', -20.0, '-30.0']
    for dfy in dfill_types:
        rfpar = rcf.reconstitute_params(rid.rids, fs, fc2use, dfill=dfy)
        print("{}, {}\n\t{}".format(dfy, fc2use, rfpar))
    # fstep
    print('\n<fstep>')
    fstep_types = ['channel', 0.1, '0.1']
    for ffy in fstep_types:
        rfpar = rcf.reconstitute_params(rid.rids, fs, fc2use, fstep=ffy)
        print("{}\n\t{}".format(ffy, rfpar))


def test_reconstitute_features():
    rid = sp.spectrum_peak.SpectrumPeak()
    rid.reader('sa_Spectrum_Peak.20180526-1033.n40.maxT-20.ridz')
    fs = '20180527-0116.E'
    rcf = sp.sp_handling.SPHandling()
    fc2use = 'maxhold'
    rcf.reconstitute_features(rid.rids, fs, fc2use, reset_params=False)
    print(rcf.rfpar)
