# _*_ coding: utf-8 _*_
# Copy 2018 the HERA Project
# Licensed under the 2-clause BSD license


"""
SPHandle:  Spectrum_Peak Handling

This has various modules to handle and view ridz files.

"""
from __future__ import print_function, absolute_import, division
from . import spectrum_peak
from argparse import Namespace


class SPHandling:

    def __init__(self):
        self.sp = spectrum_peak.SpectrumPeak()
        self.rfpar = None

    def reconstitute_params(self, rid, feature_key, feature_component, **param):
        """
        From a peak feature_set, get parameters for reconstituting a spectrum (imperfect, just use bw).

        Parameters:
        ------------
        rid:  the object containing the feature_set
        feature_key:  key of feature set
        feature_component:  feature_component to reconstitute
        data:  reconstitute feature parameters "rfpar" (changes from default)
            fmin/fmax:  min/max frequencies [from ridz file]
            fstep:  'channel' uses the channel_width
                    value:  use this value
            dfill:  value to use where no feature ['feature_set_min']
                    'feature_set_min':  minimum over the features_set
                    'component_min': minimum over the component being reconstituted
                    <'specific_component'>:  minimum over specified component
                    value:  use this value
        """
        try:
            data = getattr(rid.feature_sets[feature_key], feature_component)
        except AttributeError:
            print("{} not in set {}".format(feature_component, feature_key))
        # Set parameters
        rfpar = Namespace()
        # ...defaults
        rfpar.fstep = 'channel'
        rfpar.dfill = 'feature_set_min'
        for x in ['fmin', 'fmax']:
            setattr(rfpar, x, getattr(rid.features, x))
        # ...final values
        for x in param:
            setattr(rfpar, x, param[x])
        if isinstance(rfpar.dfill, (str, unicode)):
            if rfpar.dfill == 'component_min':
                rfpar.dfill = min(data)
            elif rfpar.dfill in self.sp.feature_components:
                rfpar.dfill = min(getattr(rid.feature_sets[feature_key], rfpar.dfill))
            elif rfpar.dfill == 'feature_set_min':
                rfpar.dfill = 1E9
                for mf in rid.feature_sets[feature_key].measured_spectral_fields:
                    v = min(getattr(rid.feature_sets[feature_key], mf))
                    if v < rfpar.dfill:
                        rfpar.dfill = v
            else:
                rfpar.dfill = float(rfpar.dfill)
        if isinstance(rfpar.fstep, (str, unicode)):
            if rfpar.fstep == 'channel':
                rfpar.fstep = rid.channel_width
            else:
                rfpar.fstep = float(rfpar.fstep)
        return rfpar

    def reconstitute_features(self, rid, feature_key, feature_component, reset_params=False, **param):
        """
        From a peak feature_set, reconstitute a spectrum (imperfect, just use bw).

        Parameters:
        ------------
        rid:  the object containing the feature_set
        feature_key:  key of feature set
        feature_component:  feature_component to reconstitute
        data:  reconstitute feature parameters "rfpar" (changes from default)
            fmin/fmax:  min/max frequencies [from ridz file]
            fstep:  'channel' uses the channel_width
                    value:  use this value
            dfill:  value to use where no feature ['feature_set_min']
                    'feature_set_min':  minimum over the features_set
                    'component_min': minimum over the component being reconstituted
                    <'specific_component'>:  minimum over specified component
                    value:  use this value
        """
        try:
            data = getattr(rid.feature_sets[feature_key], feature_component)
        except AttributeError:
            print("{} not in set {}".format(feature_component, feature_key))
        if self.rfpar is None or reset_params:
            self.rfpar = self.reconstitute_params(rid=rid, feature_key=feature_key,
                                                  feature_component=feature_component, **param)

    def reconstitute_waterfall(self, start_time, stop_time):
        """
        Give a date range to go over to make a reconstituted waterfall plot.
        """

    def raw_data_waterfall(self, start_time, stop_time):
        """
        Give a date range to make a waterfall with whatever raw data is in
        """
