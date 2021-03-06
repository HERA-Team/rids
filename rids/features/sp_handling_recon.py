# _*_ coding: utf-8 _*_
# Copy 2018 the HERA Project
# Licensed under the 2-clause BSD license


"""
SPHandle:  Spectrum_Peak Handling

This has various modules to handle and view ridz files with the SpectrumPeak feature module.

"""
from __future__ import print_function, absolute_import, division
from . import spectrum_peak
from argparse import Namespace
import numpy as np
import matplotlib.pyplot as plt
import six


class SPHandling:

    def __init__(self):
        self.sp = spectrum_peak.SpectrumPeak()
        self.reconstituted_info = None

    def reconstitute_params(self, rid, feature_key, feature_component, **param):
        """
        From a peak feature_set, get parameters for reconstituting a spectrum (imperfect, just use bw).
        This populates a 'reconstituted_info' Namespace with the relavent information

        Parameters:
        ------------
        rid:  the object containing the feature_set
        feature_key:  key of feature set
        feature_component:  feature_component to reconstitute
        data:  reconstitute feature parameters "self.reconstituted_info" (changes from default)
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
        self.reconstituted_info = Namespace()
        self.reconstituted_info.feature_key = feature_key
        self.reconstituted_info.feature_component = feature_component
        self.reconstituted_info.rid = rid
        # ...defaults
        self.reconstituted_info.fstep = 'channel'
        self.reconstituted_info.dfill = 'component_min'
        for x in ['fmin', 'fmax']:
            setattr(self.reconstituted_info, x, getattr(rid, x))
        # ...final values
        for x in param:
            setattr(self.reconstituted_info, x, param[x])
        if isinstance(self.reconstituted_info.dfill, (str)):
            if self.reconstituted_info.dfill == 'component_min':
                self.reconstituted_info.dfill = min(data)
            elif self.reconstituted_info.dfill in self.sp.feature_components:
                self.reconstituted_info.dfill = min(getattr(rid.feature_sets[feature_key],
                                                            self.reconstituted_info.dfill))
            elif self.reconstituted_info.dfill == 'feature_set_min':
                self.reconstituted_info.dfill = 1E9
                for mf in rid.feature_sets[feature_key].spectral_fields:
                    try:
                        v = min(getattr(rid.feature_sets[feature_key], mf))
                    except AttributeError:
                        continue
                    except ValueError:
                        continue
                    if not isinstance(v, (six.integer_types, float)):
                        continue
                    if v < self.reconstituted_info.dfill:
                        self.reconstituted_info.dfill = v
            else:
                self.reconstituted_info.dfill = float(self.reconstituted_info.dfill)
        if isinstance(self.reconstituted_info.fstep, (str)):
            if self.reconstituted_info.fstep == 'channel':
                self.reconstituted_info.fstep = rid.channel_width
            else:
                self.reconstituted_info.fstep = float(self.reconstituted_info.fstep)

    def reconstitute_features(self, rid, feature_key, feature_component, **param):
        """
        From a peak feature_set, reconstitute a spectrum (imperfect, just use bw).
        This finishes populating the 'reconstituted_info' Namespace

        Parameters:
        ------------
        rid:  the object containing the feature_set
        feature_key:  key of feature set
        feature_component:  feature_component to reconstitute
        data:  reconstitute feature parameters "self.reconstituted_info" (changes from default)
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
            freq = rid.feature_sets[feature_key].freq
        except AttributeError:
            print("{} not in set {}".format(feature_component, feature_key))
        self.reconstitute_params(rid=rid, feature_key=feature_key,
                                 feature_component=feature_component, **param)
        if spectrum_peak.is_spectrum(feature_key):
            self.reconstituted_info.freq, self.reconstituted_info.spec =\
                spectrum_peak._spectrum_plotter(freq, data, fmt=None)
            return

        self.reconstituted_info.freq = np.arange(self.reconstituted_info.fmin,
                                                 self.reconstituted_info.fmax,
                                                 self.reconstituted_info.fstep)
        self.reconstituted_info.spec = []
        for f in self.reconstituted_info.freq:
            val = 0.0
            num = 0
            for s, v, bw in zip(freq, data, rid.feature_sets[feature_key].bw):
                if f >= s + bw[0] and f < s + bw[1]:
                    val += v
                    num += 1
            if num:
                self.reconstituted_info.spec.append(val / num)
            else:
                self.reconstituted_info.spec.append(self.reconstituted_info.dfill)

    def reconstitute_plot(self, figname=None, ptype='all'):
        if not self.reconstituted_info.spec:
            print("Need to generate a reconstituted spectrum")
            return
        if figname is None:
            print(figname)
            figname = self.reconstituted_info.feature_key
        plt.figure(figname)
        if ptype in ['all', 'spec']:
            plt.plot(self.reconstituted_info.freq, self.reconstituted_info.spec)
        if ptype in ['all', 'points']:
            recon_fk = self.reconstituted_info.rid.feature_sets[self.reconstituted_info.feature_key]
            spec = getattr(recon_fk, self.reconstituted_info.feature_component)
            plt.plot(recon_fk.freq, spec, 'o')

    def reconstitute_waterfall(self, start_time, stop_time):
        """
        Give a date range to go over to make a reconstituted waterfall plot.
        """
