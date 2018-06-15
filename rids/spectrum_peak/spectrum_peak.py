# _*_ coding: utf-8 _*_
# Copy 2018 the HERA Project
# Licensed under the 2-clause BSD license

"""Feature module:  Spectrum_Peak
This module defines the format of the Spectrum_Peak feature_sets.
"""

from __future__ import print_function, absolute_import, division
import os
import numpy as np
from rids import rids
# import peaks  # Scipy option
import peak_det  # Another option...
import bw_finder


class Spectral:
    """
    For now, the order matters since used to order peak finding priority
    """
    spectral_fields = ['maxhold', 'minhold', 'val', 'comment', 'polarization', 'freq', 'bw']
    measured_spectral_fields = ['maxhold', 'minhold', 'val']

    def __init__(self, polarization='', comment=''):
        self.comment = comment
        self.polarization = polarization
        self.freq = []
        self.val = []


class SpectrumPeak:
    """
    This feature module searches a directory for spectrum files and saves feature sets based
    peak finding (or saves rawdata).

    The feature_module_name is "Spectrum_Peak"

    Direct attributes are:
        comment:
        peaked_on:  event_component on which peaks were found
        delta:
        bw_range:
        delta_values:
    Unit attributes are:
        threshold:  value used to threshold peaks
        threshold_unit: unit of threshold
        vbw:
        vbw_unit:
        rbw:
        rbw_unit:
    """
    direct_attributes = ['comment', 'peaked_on', 'delta', 'bw_range', 'delta_values', 'fmin', 'fmax']
    unit_attributes = ['threshold', 'rbw', 'vbw']
    feature_module_name = 'Spectrum_Peak'
    polarizations = ['E', 'N', 'I']

    def __init__(self, comment='', view_ongoing=False):
        for d in self.direct_attributes:
            setattr(self, d, None)
        for d in self.unit_attributes:
            setattr(self, d, None)
            setattr(self, d + '_unit', None)
        # Re-initialize attributes
        self.comment = comment
        self.bw_range = []  # Range to search for bandwidth
        self.delta_values = {'zen': 0.1, 'sa': 1.0}
        # Copy rids to local
        self.rids = rids.Rids(self)
        self.reader = self.rids.reader
        self.writer = self.rids.writer
        self.info = self.rids.info
        # Other attributes
        feature_set_info = Spectral()
        self.feature_components = feature_set_info.spectral_fields
        self.hipk = None
        self.hipk_bw = None
        self.view_ongoing_features = view_ongoing

    def reset(self):
        self.__init__()

    def set(self, **kwargs):
        for k in kwargs:
            if k in self.direct_attributes:
                setattr(self, k, kwargs[k])
            elif k in self.unit_attributes:
                self.rids.set_unit_values(self, k, kwargs[k])

    def read_feature_set_dict(self, fs):
        feature_set = Spectral()
        for v, Y in fs.iteritems():
            if v not in self.feature_components:
                print("Unexpected field {}".format(v))
                continue
            setattr(feature_set, v, Y)
        return feature_set

    def get_feature_sets(self, fset_tag, polarization, peak_on=None, **fnargs):
        """
        Get the feature sets for current iteration
        **fnargs are filenames for self.feature_components {"feature_component": <filename>}
        """
        fset_name = fset_tag + polarization
        is_spectrum = 'data' in fset_tag.lower() or 'cal' in fset_tag.lower()
        self.rids.feature_sets[fset_name] = Spectral(polarization=polarization)
        self.rids.feature_sets[fset_name].freq = []
        spectra = {}
        for fc, sfn in fnargs.iteritems():
            if fc not in self.feature_components or sfn is None:
                continue
            spectra[fc] = Spectral()
            spectrum_reader(sfn, spectra[fc], polarization)
            if is_spectrum:
                if len(spectra[fc].freq) > len(self.rids.feature_sets[fset_name].freq):
                    self.rids.feature_sets[fset_name].freq = spectra[fc].freq
                setattr(self.rids.feature_sets[fset_name], fc, spectra[fc].val)
        self.rids.nsets += 1
        if is_spectrum:
            return

        # Get the feature_component to use and find peaks/bw
        if peak_on is not None:
            fc = peak_on
        else:
            for fc in self.feature_components:
                if fc in fnargs.keys() and fnargs[fc] is not None:
                    break
            else:
                return
        if self.peaked_on is None:
            self.peaked_on = fc
        elif fc != self.peaked_on:
            spectra[fc].comment += 'Peaked on different component: {} rather than {}'.format(fc, self.peaked_on)
        # self.peak_finder(spectra[ec], view_peaks=self.view_peaks_on_event)
        self.peak_det(spectra[fc], delta=self.delta)
        self.find_bw()

        # put values in feature set dictionary
        self.rids.feature_sets[fset_name].freq = list(np.array(self.hipk_freq)[self.hipk])
        for fc, sfn in fnargs.iteritems():
            if sfn is None:
                continue
            try:
                setattr(self.rids.feature_sets[fset_name], fc, list(np.array(spectra[fc].val)[self.hipk]))
            except IndexError:
                pass
        self.rids.feature_sets[fset_name].bw = self.hipk_bw
        ftr_fmin = min(self.rids.feature_sets[fset_name].freq)
        ftr_fmax = max(self.rids.feature_sets[fset_name].freq)
        if ftr_fmin < self.fmin:
            self.fmin = ftr_fmin
        if ftr_fmax > self.fmax:
            self.fmax = ftr_fmax

    def peak_det(self, spec, delta=0.1):
        self.hipk_freq = spec.freq
        self.hipk_val = spec.val
        self.hipk = peak_det.peakdet(spec.val, delta=delta, threshold=self.threshold)
        self.hipk = list(self.hipk)

    def find_bw(self):
        self.hipk_bw = bw_finder.bw_finder(self.hipk_freq, self.hipk_val, self.hipk, self.bw_range)
        if self.view_ongoing_features:
            self.peak_viewer()

    def peak_finder(self, spec, cwt_range=[1, 3], rc_range=[4, 4]):
        self.hipk_freq = spec.freq
        self.hipk_val = spec.val
        self.hipk = peaks.fp(spec.val, self.threshold, cwt_range, rc_range)

    def peak_viewer(self):
        if self.hipk is None:
            return
        import matplotlib.pyplot as plt
        plt.plot(self.hipk_freq, self.hipk_val)
        plt.plot(np.array(self.hipk_freq)[self.hipk], np.array(self.hipk_val)[self.hipk], 'kv')
        if self.hipk_bw is not None:
            vv = np.array(self.hipk_val)[self.hipk]
            if 'dB' in self.val_unit:
                vv2 = vv - 6.0
            else:
                vv2 = vv / 4.0
            fl = np.array(self.hipk_freq)[self.hipk] + np.array(self.hipk_bw)[:, 0]
            fr = np.array(self.hipk_freq)[self.hipk] + np.array(self.hipk_bw)[:, 1]
            plt.plot([fl, fl], [vv, vv2], 'm')
            plt.plot([fr, fr], [vv, vv2], 'c')
        plt.show()

    def viewer(self, threshold=None, show_components='all', show_data=True):
        """
        Parameters:
        ------------
        show_components:  event_components to show (list) or 'all'
        show_data:  include data spectra (Boolean)
        """
        import matplotlib.pyplot as plt
        import collections
        if threshold is not None and threshold < self.threshold:
            print("Below threshold - using {}".format(self.threshold))
            threshold = None
        if threshold is not None:
            print("Threshold view not implemented yet.")
            threshold = None
        fc_list = collections.OrderedDict([('maxhold', ('r', 'v')), ('minhold', ('b', '^')), ('val', ('k', '_'))])
        if isinstance(show_components, (str, unicode)):
            show_components = fc_list.keys()
        color_list = ['r', 'b', 'k', 'g', 'm', 'c', 'y', '0.25', '0.5', '0.75']
        line_list = ['-', '--', ':']
        c = 0
        bl = 0
        for f, v in self.rids.feature_sets.iteritems():
            is_data = 'data' in f.lower() or 'cal' in f.lower() or 'baseline' in f.lower()
            if is_data and not show_data:
                continue
            if is_data:
                clr = [fc_list[x][0] for x in show_components]
                ls = [line_list[bl % len(line_list)]] * len(show_components)
                fmt = [None] * len(show_components)
                bl += 1
            else:
                clr = [color_list[c % len(color_list)]] * len(show_components)
                ls = [None] * len(show_components)
                fmt = [fc_list[x][1] for x in fc_list]
                c += 1
            for fc in show_components:
                try:
                    i = show_components.index(fc)
                    spectrum_plotter(self.rids.rid_file, is_data, v.freq, getattr(v, fc), fmt[i], clr[i], ls[i], plt)
                except AttributeError:
                    pass
            if is_data:
                continue
            # Now plot bandwidth
            try:
                vv = np.array(getattr(v, self.peaked_on))
                fv = np.array(v.freq)
                bw = np.array(v.bw)
                if 'dB' in self.rids.val_unit:
                    vv2 = vv - 6.0
                else:
                    vv2 = vv / 4.0
                fl = fv + bw[:, 0]
                if self.fmin is not None:
                    ifl = np.where(fl < self.fmin)
                    fl[ifl] = self.fmin
                fr = fv + bw[:, 1]
                if self.fmax is not None:
                    ifr = np.where(fr > self.fmax)
                    fr[ifr] = self.fmax
                for j in range(len(fl)):
                    plt.plot([fl[j], fl[j]], [vv[j], vv2[j]], clr[0])
                    plt.plot([fr[j], fr[j]], [vv[j], vv2[j]], clr[0])
            except AttributeError:
                continue
        plt.show()

    def read_cal(self, filename, polarization):
        tag = 'cal.' + filename + '_'
        self.get_feature_sets(tag, polarization, val=filename)

    def apply_cal(self):
        print("NOT IMPLEMENTED YET: Apply the calibration, if available.")

    def process_files(self, directory='.', ident='all', data=[0, -1], peak_on=False, sets_per_pol=100, max_loops=1000):
        """
        This is the standard method to process spectrum files in a directory to
        produce ridz files.

        Format of the spectrum filename (peel_filename below):
        <path>/identifier.time-stamp.feature_component.polarization

        Parameters:
        ------------
        directory:  directory where files reside and ridz files get written
        idents:  idents to do.  If 'all' processes all, picking first for overall ident
        data:  indices of events to be saved as data spectra
        sets_per_pol:  number of events (see event_components) per pol per ridz file
        max_loops:  will stop after this number
        """
        loop = True
        loop_ctr = 0
        self.ident = None
        self.fmin = 1E9
        self.fmax = -1E9
        while (loop):
            loop_ctr += 1
            if loop_ctr > max_loops:
                break
            # Set up the feature_component file dictionary
            ftrfiles = {}
            max_pol_cnt = {}
            for pol in self.polarizations:
                ftrfiles[pol] = {}
                max_pol_cnt[pol] = 0
                for fc in self.feature_components:
                    ftrfiles[pol][fc] = []
            # Go through and sort the available files into file dictionary
            loop = False
            available_files = sorted(os.listdir(directory))
            file_times = []
            for af in available_files:
                if 'rid' in af.split('.')[-1]:
                    continue
                fnd = peel_filename(af, self.feature_components)
                if not len(fnd) or\
                        fnd['polarization'] not in self.polarizations or\
                        fnd['feature_component'] not in self.feature_components:
                    continue
                if ident == 'all' or ident == fnd['ident']:
                    loop = True
                    file_list = ftrfiles[fnd['polarization']][fnd['feature_component']]
                    if len(file_list) > sets_per_pol:
                        continue
                    file_times.append(fnd['timestamp'])
                    file_list.append(os.path.join(directory, af))
                    if self.rids.ident is None:
                        self.rids.ident = fnd['ident']
            if self.rids.ident not in self.delta_values:
                self.delta = 0.1
            else:
                self.delta = self.delta_values[self.rids.ident]
            if not loop:
                break
            # Go through and count the sorted available files
            num_to_read = {}
            for pol in ftrfiles:
                for fc in ftrfiles[pol]:
                    if len(ftrfiles[pol][fc]) > max_pol_cnt[pol]:
                        max_pol_cnt[pol] = len(ftrfiles[pol][fc])
                for fc in ftrfiles[pol]:
                    diff_len = max_pol_cnt[pol] - len(ftrfiles[pol][fc])
                    if diff_len > 0:
                        ftrfiles[pol][fc] = ftrfiles[pol][fc] + [None] * diff_len
                    num_to_read[pol] = len(ftrfiles[pol][fc])  # Yes, does get reset
            file_times = sorted(file_times)
            self.rids.timestamp_first = file_times[0]
            self.rids.timestamp_last = file_times[-1]
            # Process the files
            self.rids.feature_sets = {}
            self.rids.nsets = 0
            for pol in self.polarizations:
                if not max_pol_cnt[pol]:
                    continue
                self.rids.nsets += 1
                # Get the data spectrum files
                for i in data:
                    bld = {}
                    for fc, fcfns in ftrfiles[pol].iteritems():
                        if abs(i) >= len(fcfns) or fcfns[i] is None:
                            continue
                        fnd = peel_filename(fcfns[i], self.feature_components)
                        bld[fc] = fcfns[i]
                    if not len(bld):
                        break
                    fvn = 'data.{}.{}.'.format(i, fnd['timestamp'])
                    self.get_feature_sets(fvn, pol, **bld)
                # Get the feature_sets
                for i in range(num_to_read[pol]):
                    fcd = {}
                    for fc, fcfns in ftrfiles[pol].iteritems():
                        fnd = peel_filename(fcfns[i], self.feature_components)
                        if 'timestamp' in fnd:
                            fnd_ts = fnd['timestamp']
                        if len(fnd):
                            fcd[fc] = fcfns[i]
                    if not len(fcd):
                        continue
                    self.get_feature_sets(fnd_ts, pol, peak_on=peak_on, **fcd)
                    for x in fcd.itervalues():
                        if x is not None:
                            os.remove(x)
            # Write the ridz file
            th = self.threshold
            if abs(self.threshold) < 1.0:
                th *= 100.0
            pk = self.peaked_on[:3]
            fn = "{}_{}.{}.n{}.{}T{:.0f}.ridz".format(self.rids.ident, self.feature_module_name,
                                                      self.rids.timestamp_first, self.rids.nsets,
                                                      pk, th)
            self.filename = os.path.join(directory, fn)
            self.rids.writer(self.filename)


def spectrum_reader(filename, spec, polarization=None):
    """
    This reads in an ascii spectrum file.
    """
    if filename is None:
        return
    if polarization is not None:
        spec.polarization = polarization
    with open(filename, 'r') as f:
        for line in f:
            data = [float(x) for x in line.split()]
            spec.freq.append(data[0])
            spec.val.append(data[1])


def spectrum_plotter(figure_name, is_data, x, y, fmt, clr, ls, plt):
    if not len(x) or not len(y):
        return
    try:
        plt.figure(figure_name)
        _X = x[:len(y)]
        if is_data:
            plt.plot(_X, y, clr, linestyle=ls)
        else:
            plt.plot(_X, y, fmt, color=clr)
    except ValueError:
        _Y = y[:len(x)]
        if is_data:
            plt.plot(x, _Y, clr, linestyle=ls)
        else:
            plt.plot(x, _Y, fmt, color=clr)


def peel_filename(v, fclist=None):
    if v is None:
        return {}
    s = v.split('/')[-1].split('.')
    if len(s) not in [4, 5]:  # timestamp may have one '.' in it
        return {}
    fnd = {'ident': s[0]}
    fnd['timestamp'] = '.'.join(s[1:-2])
    fnd['feature_component'] = s[-2].lower()
    fnd['polarization'] = s[-1].upper()
    if fclist is None:
        return fnd
    fclist = [x.lower() for x in fclist]
    for fc in fclist:
        if fnd['feature_component'] in fc:
            fnd['feature_component'] = fc
            return fnd
    return {}
