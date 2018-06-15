# _*_ coding: utf-8 _*_
# Copy 2018 the HERA Project
# Licensed under the 2-clause BSD license


"""
SPHandle:  Spectrum_Peak Handling

This has various modules to handle and view ridz files.

"""
from __future__ import print_function, absolute_import, division
from . import spectrum_peak as sp


class SPHandling:

    def __init__(self):
        self.rid = sp.spectrum_peak.SpectrumPeak()

    def reconstitute_feature(self, feature_set, feature_component, data=None):
        """
        From a peak feature_set, reconstitute a spectrum (imperfect, just use bw).
        """
        try:
            data = feature_set[feature_component]
        except KeyError:
            print("{} not in set {}".format(feature_component, tag))
        # data is the nearest "baseline", or outside_bw_value or None
        #  if None takes smallest value in feature_set

    def reconstitute_waterfall(self, start_time, stop_time):
        """
        Give a date range to go over to make a reconstituted waterfall plot.
        """

    def raw_data_waterfall(self, start_time, stop_time):
        """
        Give a date range to make a waterfall with whatever raw data is in
        """
