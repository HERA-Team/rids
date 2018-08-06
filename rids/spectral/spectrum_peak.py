# _*_ coding: utf-8 _*_
# Copy 2018 the HERA Project
# Licensed under the 2-clause BSD license

"""Feature module:  SpectrumPeak
This module defines the format of the SpectrumPeak feature_sets.
"""

from __future__ import print_function, absolute_import, division
import os
import copy
import numpy as np
import matplotlib.pyplot as plt
from .. import rids
# from . import peaks  # Scipy option
from . import peak_det  # Another option that seems to do better.
from . import bw_finder


class Spectral:
    """
    Generic spectral class, with the 'basics' initialized.  The other fields may get added
    as appropriate.
    For now, the order matters for 'spectral_fields' since used to order peak-finding priority
    """
    spectral_fields = ['maxhold', 'minhold', 'val', 'comment', 'polarization', 'freq', 'bw']

    def __init__(self, polarization='', comment=''):
        self.comment = comment
        self.polarization = polarization
        self.freq = []
        self.val = []


def is_spectrum(tag, prefixes=['data', 'cal', 'baseline']):
    """
    This defines prefix "tags" that denote "spectra" - i.e. meant to be complete spectra
    and not a subset of features (e.g. peaks in this case)
    """
    t = tag.lower()
    for s in prefixes:
        if t[:len(s)] == s:
            return True
    return False


class SpectrumPeak(rids.Rids):
    """
    Feature module:
    This feature module searches a directory for spectrum files and saves feature sets based
    peak finding (or saves rawdata).

    The feature_module_name is "SpectrumPeak"

    Direct attributes are:
        feature_module_name:  SpectrumPeak
        comment:  generic comments
        peaked_on:  event_component on which peaks were found
        delta:  delta value used in the peak-finding routine
        bw_range:  +/- bw_range (list) that gets searched over (and is then max)
        delta_values:  a dictionary with some default values
        freq: can be used for is_spectrum feature_sets if desired
        freq_unit:  (even though a _unit, handled differently than unit attribute)
        fmin: minimum frequency
        fmax: maximum frequency
        val_unit:  unit for data
    Unit attributes are:
        threshold:  value used to threshold peaks
        threshold_unit:
        vbw:  if spectrum analyzer, video bandwidth
        vbw_unit:
        rbw:  if spectrum analyzer, resolution bandwidth (probably channel_width)
        rbw_unit:
    """
    sp__direct_attributes = ['peaked_on', 'delta', 'bw_range', 'delta_values', 'feature_module_name',
                             'freq', 'freq_unit', 'fmin', 'fmax', 'val_unit']
    sp__unit_attributes = ['threshold', 'rbw', 'vbw']
    polarizations = ['E', 'N', 'I']

    def __init__(self, comment='', share_freq=False, view_ongoing=False):
        # Initialize base attributes
        super(SpectrumPeak, self).__init__(comment)

        # Initialize feature_module attributes
        for d in self.sp__direct_attributes:
            setattr(self, d, None)
        for d in self.sp__unit_attributes:
            setattr(self, d, None)
            setattr(self, d + '_unit', None)

        # Re-initialize attributes
        self.feature_module_name = 'SpectrumPeak'
        self.bw_range = []  # Range to search for bandwidth
        self.delta_values = {'zen': 0.1, 'sa': 1.0}
        self.feature_sets = {}
        # Other attributes
        feature_set_info = Spectral()
        self.feature_components = feature_set_info.spectral_fields
        self.hipk = None
        self.hipk_bw = None
        self.view_ongoing_features = view_ongoing
        self.share_freq = share_freq

    # Redefine the reader/writer/info base modules
    def reader(self, filename, reset=True):
        print("Reading {}".format(filename))
        if reset:
            self.reset()
        self._reader(filename, feature_direct=self.sp__direct_attributes,
                     feature_unit=self.sp__unit_attributes)
        # now need to fill in the share_freq feature values, if any
        for ftr in self.feature_sets:
            if self.feature_sets[ftr].freq == '@':
                self.feature_sets[ftr].freq = self.freq

    def writer(self, filename, fix_list=True):
        self._writer(filename, feature_direct=self.sp__direct_attributes,
                     feature_unit=self.sp__unit_attributes, fix_list=fix_list)

    def info(self):
        self._info(feature_direct=self.sp__direct_attributes,
                   feature_unit=self.sp__unit_attributes)

    def reset(self):
        self.__init__(share_freq=self.share_freq, view_ongoing=self.view_ongoing_features)

    def read_feature_set_dict(self, fs):
        """
        This is used in RIDS parent class for base reader.
        """
        feature_set = Spectral()
        for v, Y in fs.items():
            if v not in self.feature_components:
                print("Unexpected field {}".format(v))
                continue
            setattr(feature_set, v, Y)
        return feature_set

    def get_feature_sets(self, fset_tag, polarization, peak_on=None, **fnargs):
        """
        Get the feature sets for current iteration from individual spectrum files
        with naming convention as realized in module 'peel_filename' below
        **fnargs are filenames for self.feature_components {"feature_component": <filename>}
        """
        fset_name = fset_tag + polarization
        self.feature_sets[fset_name] = Spectral(polarization=polarization)
        self.feature_sets[fset_name].freq = []
        spectra = {}
        for fc, sfn in fnargs.items():
            if fc not in self.feature_components or sfn is None:
                continue
            spectra[fc] = Spectral()
            spectrum_reader(sfn, spectra[fc], polarization)
            ftr_fmin = min(spectra[fc].freq)
            ftr_fmax = max(spectra[fc].freq)
            if ftr_fmin < self.fmin:
                self.fmin = ftr_fmin
            if ftr_fmax > self.fmax:
                self.fmax = ftr_fmax
            if is_spectrum(fset_tag):
                if self.share_freq:
                    self.feature_sets[fset_name].freq = '@'
                    if self.freq is None:
                        self.freq = copy.copy(spectra[fc].freq)
                elif len(spectra[fc].freq) > len(self.feature_sets[fset_name].freq):
                    self.feature_sets[fset_name].freq = spectra[fc].freq
                setattr(self.feature_sets[fset_name], fc, spectra[fc].val)
            if len(spectra[fc].comment):
                self.feature_sets[fset_name].comment += spectra[fc].comment

        self.nsets += 1
        if is_spectrum(fset_tag):
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
        self.feature_sets[fset_name].freq = list(np.array(self.hipk_freq)[self.hipk])
        if len(spectra[fc].comment):
            self.feature_sets[fset_name].comment += spectra[fc].comment
        for fc, sfn in fnargs.items():
            if sfn is None:
                continue
            try:
                setattr(self.feature_sets[fset_name], fc, list(np.array(spectra[fc].val)[self.hipk]))
            except IndexError:
                pass
        self.feature_sets[fset_name].bw = self.hipk_bw

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
        for f, v in self.feature_sets.items():
            issp = is_spectrum(f)
            if issp and not show_data:
                continue
            use_freq = v.freq
            if issp:
                clr = [fc_list[x][0] for x in show_components]
                ls = [line_list[bl % len(line_list)]] * len(show_components)
                fmt = zip(clr, ls)
                bl += 1
                if use_freq == '@':
                    use_freq = self.freq
            else:
                clr = [color_list[c % len(color_list)]] * len(show_components)
                mkr = [fc_list[x][1] for x in fc_list]
                fmt = zip(clr, mkr)
                c += 1
            for fc in show_components:
                try:
                    i = show_components.index(fc)
                    spectrum_plotter(use_freq, getattr(v, fc), fmt=fmt[i], is_spectrum=issp, figure_name=self.rid_file)
                except AttributeError:
                    pass
            if issp:
                continue
            # Now plot bandwidth
            try:
                vv = np.array(getattr(v, self.peaked_on))
                fv = np.array(use_freq)
                bw = np.array(v.bw)
                if 'dB' in self.val_unit:
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
        tag = 'cal.' + filename + '.'
        self.get_feature_sets(tag, polarization, val=filename)

    def apply_cal(self):
        print("NOT IMPLEMENTED YET: Apply the calibration, if available.")

    def process_files(self, directory='.', ident='all', data=[0, -1], peak_on=None,
                      data_only=False, sets_per_pol=10000, max_loops=1000):
        """
        This is the standard method to process spectrum files in a directory to
        produce ridz files.  The module has an "outer loop" that is meant to handle
        ongoing file-writing along with file reading.  It will quit when either it
        has run out of appropriate files or the max_loops has been exceeded (unlikely
        but just in case)

        Format of the spectrum filename (peel_filename below):
        identifier.{time-stamp}.feature_component.polarization

        Parameters:
        ------------
        directory:  directory where files reside and ridz files get written
        ident:  idents to do.  If 'all' processes all, picking first for overall ident
        data:  indices of events to be saved as data spectra
        peak_on:  feature_component on which to find peaks, default order if None
        data_only:  flag if only saving data and not peaks
        sets_per_pol:  number of feature_sets per pol per ridz file
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
                pol = fnd['polarization']
                feco = fnd['feature_component']
                if ident == 'all' or ident == fnd['ident']:
                    loop = True
                    if len(ftrfiles[pol][feco]) > sets_per_pol:
                        continue
                    file_times.append(fnd['timestamp'])
                    ftrfiles[pol][feco].append(os.path.join(directory, af))
                    if self.ident is None:
                        self.ident = fnd['ident']
            if self.ident not in self.delta_values:
                self.delta = 0.1
            else:
                self.delta = self.delta_values[self.ident]
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
                    num_to_read[pol] = len(ftrfiles[pol][fc])  # Yes, does get reset many times
            file_times = sorted(file_times)
            self.timestamp_first = file_times[0]
            self.timestamp_last = file_times[-1]
            # Process the files
            self.feature_sets = {}
            self.nsets = 0
            for pol in self.polarizations:
                if not max_pol_cnt[pol]:
                    continue
                processed_pol_data = []
                # Get the data spectrum files
                for i in data:
                    bld = {}
                    for fc, fcfns in ftrfiles[pol].items():
                        if abs(i) >= len(fcfns) or fcfns[i] is None:
                            continue
                        fnd = peel_filename(fcfns[i], self.feature_components)
                        bld[fc] = fcfns[i]
                    if not len(bld):
                        break
                    feature_tag = 'data.{}.'.format(fnd['timestamp'])
                    self.get_feature_sets(feature_tag, pol, **bld)
                # Get the feature_sets, write unless data_only and remove processed files
                for i in range(num_to_read[pol]):
                    fcd = {}
                    for fc, fcfns in ftrfiles[pol].items():
                        fnd = peel_filename(fcfns[i], self.feature_components)
                        if 'timestamp' in fnd:
                            feature_tag = fnd['timestamp'] + '.'
                        if len(fnd):
                            fcd[fc] = fcfns[i]
                    if not len(fcd):
                        continue
                    if not data_only:
                        self.get_feature_sets(feature_tag, pol, peak_on=peak_on, **fcd)
                    # Delete processed files
                    for x in fcd.itervalues():
                        if x is not None:
                            os.remove(x)
            # Write the ridz file
            if self.threshold is not None:
                th = self.threshold
                if abs(self.threshold) < 1.0:
                    th *= 100.0
                th = 'T{:.0f}'.format(th)
            else:
                th = ''
            if self.peaked_on is not None:
                pk = self.peaked_on[:3]
            else:
                pk = 'None'
                th = ''
            fn = "{}_{}.{}.n{}.{}{}.ridz".format(self.ident, self.feature_module_name,
                                                 self.timestamp_first, self.nsets,
                                                 pk, th)
            self.filename = os.path.join(directory, fn)
            self.writer(self.filename)


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
            if line.strip()[0] == ':':
                spec.comment += (line.strip()[1:] + '\n')
                continue
            data = [float(x) for x in line.split()]
            spec.freq.append(data[0])
            spec.val.append(data[1])


def spectrum_plotter(x, y, fmt=None, is_spectrum=False, figure_name=None):
    if not len(x) or not len(y):
        return
    if len(x) > len(y):
        _X = x[:len(y)]
        _Y = y[:]
    else:
        _X = x[:]
        _Y = y[:len(x)]
    if fmt is None:
        return _X, _Y

    if figure_name is not None:
        plt.figure(figure_name)
    if is_spectrum:
        plt.plot(_X, _Y, color=fmt[0], linestyle=fmt[1])
    else:
        plt.plot(_X, _Y, color=fmt[0], marker=fmt[1], linestyle="None")


def peel_filename(v, fclist=None):
    if v == 'filename_format_help':
        s = "The filename format convention is:"
        s += "\tidentifier.{time-stamp}.feature_component.polarization"
        s += "\ni.e. x=filename.split(.) has:"
        s += "\tx[0]:  arbitrary identifier"
        s += "\tx[...]: time-stamp (may have .'s in it)"
        s += "\tx[-2]:  feature_component ('maxhold', 'minhold', 'val')"
        s += "\tx[-1]:  polarization"
        return s
    if v is None:
        return {}
    s = v.split('/')[-1].split('.')
    if len(s) < 4:
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
