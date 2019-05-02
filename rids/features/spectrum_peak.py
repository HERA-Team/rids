# _*_ coding: utf-8 _*_
# Copy 2018 the HERA Project
# Licensed under the 2-clause BSD license

"""Feature module:  SpectrumPeak
This module defines the format of the SpectrumPeak feature_sets.
"""

from __future__ import print_function, absolute_import, division
import os
import copy
import six
import numpy as np
import matplotlib.pyplot as plt
import datetime
from .. import rids
# from . import peaks  # Scipy option
from . import peak_det  # Another option that seems to do better.
from . import bw_finder


class Spectral:
    """
    Generic spectral class, with the 'basics' initialized.  The other fields may get added
    as appropriate.
    For now, the order matters since used to order peak-finding priority
    """
    data_spectral_class_fields = ['maxhold', 'minhold', 'val']
    other_spectral_class_fields = ['comment', 'polarization', 'freq', 'bw']
    all_spectral_class_fields = data_spectral_class_fields + other_spectral_class_fields

    def __init__(self, polarization='', comment=''):
        self.comment = comment
        self.polarization = polarization
        self.freq = []


class FileSet:
    """
    Class used to generate the feature_set function group by id
    and act on it (chunk it, etc)
    """

    def __init__(self, id):
        self.id = id
        self.included_feature_components = set()
        # Each of these below is per polarization
        self.timestamps = {}
        spec = Spectral()
        for dscf in spec.data_spectral_class_fields:
            setattr(self, dscf, {})

    def chunk_it(self, chunk_set):
        self.chunked = {}
        self.biggest_pol = {'pol': '-', 'len': -1, 'size': -1}
        for pol in self.timestamps:
            self.chunked[pol] = []
            _chnk = []
            _cize = 0
            for i, ts in enumerate(sorted(self.timestamps[pol])):
                if i and not i % chunk_set:
                    self.chunked[pol].append(_chnk)
                    _chnk = []
                _chnk.append(ts)
                _cize += 1
            self.chunked[pol].append(_chnk)
            if _cize > self.biggest_pol['size']:
                self.biggest_pol['pol'] = pol
                self.biggest_pol['len'] = len(self.chunked[pol])
                self.biggest_pol['size'] = _cize
        self.chunked_time_limits = [[] for x in range(self.biggest_pol['len'])]
        for p, ts_chunked in six.iteritems(self.chunked):
            for i, ts_list in enumerate(ts_chunked):
                self.chunked_time_limits[i].extend((ts_list[0], ts_list[-1]))
        for i in range(len(self.chunked_time_limits)):
            x = sorted(self.chunked_time_limits[i])
            self.chunked_time_limits[i] = [x[0], x[-1]]


def is_spectrum(tag, prefixes=('data', 'cal', 'baseline')):
    """
    This defines prefix "tags" that denote "spectra" - i.e. meant to be complete spectra
    (so plot with lines) and not a subset of features (e.g. peaks in this case, plot with points).
    """
    t = tag.lower()
    return t.startswith(prefixes)


class SpectrumPeak(rids.Rids):
    """
    Feature module:
    This feature module searches a directory for spectrum files and saves feature sets based
    peak finding (or saves rawdata).

    The feature_module_name is "SpectrumPeak"

    Direct attributes are (on top of base rids attributes):
        feature_module_name:  SpectrumPeak
        comment:  generic comments
        peaked_on:  event_component on which peaks were found
        delta:  delta value used in the peak-finding routine
        bw_range:  +/- bw_range (list) that gets searched over (and is then max)
        val_unit:  unit for data
        freq_unit:  (even though a _unit, handled differently than unit attribute)
        ... these are derived from data
        freq: can be used for is_spectrum feature_sets if desired
        fmin: minimum frequency
        fmax: maximum frequency
    Unit attributes are:
        threshold:  value used to threshold peaks
        threshold_unit:
        vbw:  if spectrum analyzer, video bandwidth
        vbw_unit:
        rbw:  if spectrum analyzer, resolution bandwidth (probably channel_width)
        rbw_unit:
    """
    sp__direct_attributes = ['peaked_on', 'delta', 'bw_range', 'feature_module_name',
                             'freq', 'freq_unit', 'fmin', 'fmax', 'val_unit']
    sp__unit_attributes = ['threshold', 'rbw', 'vbw']
    sp__polarizations = ['E', 'N', 'I', 'Q', 'U', 'V', 'H', 'X', 'Y', 'XX', 'YY', 'XY', 'YX', 'unk', 'none']

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
        self.delta = 10.0  # Fairly random default value
        self.bw_range = [-10.0, 10.0]  # "
        self.feature_sets = {}
        # Other attributes
        feature_set_info = Spectral()
        self.feature_components = feature_set_info.all_spectral_class_fields
        self.hipk = None
        self.hipk_bw = None
        self.view_ongoing_features = view_ongoing
        self.share_freq = share_freq
        self.all_lowercase_polarization_names = [x.lower() for x in self.sp__polarizations]

    # Redefine the reader/writer/info base modules
    def reader(self, filename, reset=True):
        """
        This will read RID files with a full or subset of structure entities.
        If 'filename' is a RID file, it will read that file.
        If 'filename' is a list, it will read in that list of files.
        If 'filename' is a non-RID file it will assume it is a list of files and
            1 - reset the current feature set
            2 - read in the files listed in 'filename'

        Parameters:
        ------------
        filename:  if single RID file or list read it/them
                   if non-RID file, reset feature set and read list of files in it
        reset:  If true, resets all elements.
                If false, will overwrite headers etc but add events
                    (i.e. things with a unique key won't get overwritten)
        """
        file_list, reset = rids._get_file_list_and_reset(filename, reset)
        if reset:
            self.reset()
        for filename in file_list:
            print("Reading {}".format(filename))
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
        for v, Y in six.iteritems(fs):
            if v not in self.feature_components:
                print("Unexpected field {}".format(v))
                continue
            setattr(feature_set, v, Y)
        return feature_set

    def peak_det(self, spec, delta=0.1):
        self.hipk_freq = spec.freq
        self.hipk_val = spec.val
        self.hipk = peak_det.peakdet(spec.val, delta=delta, threshold=self.threshold)
        self.hipk = list(self.hipk)

    def find_bw(self):
        self.hipk_bw = bw_finder.bw_finder(self.hipk_freq, self.hipk_val, self.hipk, self.bw_range)
        if self.view_ongoing_features:
            self.peak_viewer()

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

    def showtimes(self):
        spectimes = {}
        othertimes = {}
        for f, v in six.iteritems(self.feature_sets):
            ts = datetime.datetime.strptime(_get_timestr_from_ftr_key(f), "%Y%m%d-%H%M%S")
            if is_spectrum(f):
                if v.polarization in spectimes.keys():
                    spectimes[v.polarization].append(ts)
                else:
                    spectimes[v.polarization] = [ts]
            else:
                if v.polarization in othertimes.keys():
                    othertimes[v.polarization].append(ts)
                else:
                    othertimes[v.polarization] = [ts]

        delta_st = {}
        delta_ot = {}
        plt.figure("Sample Times/Steps")
        for p, tval in six.iteritems(spectimes):
            delta_st[p] = [0.0]
            for i in range(len(tval)):
                if i:
                    dt = (tval[i] - tval[i - 1]).total_seconds()
                    if dt > 3600.0:
                        print("Resetting {:.2f} hours to 0 at {}".format(dt / 3600.0, tval[i]))
                        dt = 0.0
                    delta_st[p].append(dt)
            plt.plot(tval, delta_st[p], '.', label=p + ': raw')
        for p, tval in six.iteritems(othertimes):
            delta_ot[p] = []
            for i in range(len(tval)):
                if i:
                    dt = (tval[i] - tval[i - 1]).total_seconds()
                    if dt > 3600.0:
                        print("Resetting {:.2f} hours to 0 at {}".format(dt / 3600.0, tval[i]))
                        dt = 0.0
                    delta_ot[p].append(dt)
            plt.plot(tval, delta_ot[p], '.', label=p + ': peaks')

        plt.legend()
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
        if isinstance(show_components, six.string_types):
            show_components = list(fc_list.keys())
        color_list = ['r', 'b', 'k', 'g', 'm', 'c', 'y', '0.25', '0.5', '0.75']
        line_list = ['-', '--', ':']
        c = 0
        bl = 0
        for f, v in six.iteritems(self.feature_sets):
            issp = is_spectrum(f)
            if issp and not show_data:
                continue
            use_freq = v.freq
            if issp:
                clr = [fc_list[x][0] for x in show_components]
                ls = [line_list[bl % len(line_list)]] * len(show_components)
                fmt = list(zip(clr, ls))
                bl += 1
                if use_freq == '@':
                    use_freq = self.freq
            else:
                clr = [color_list[c % len(color_list)]] * len(show_components)
                mkr = [fc_list[x][1] for x in fc_list]
                fmt = list(zip(clr, mkr))
                c += 1
            for fc in show_components:
                try:
                    i = show_components.index(fc)
                    _spectrum_plotter(use_freq, getattr(v, fc), fmt=fmt[i], is_spectrum=issp, figure_name=self.rid_file)
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

    def read_cal(self, filename, time_stamp, polarization):
        """
        This would allow you to specify an "external" calibration
        """
        tag = 'cal:' + time_stamp + ':'
        self.get_feature_set_from_files(tag, polarization, val=filename)

    def get_cal_feature_keys(self):
        cals = {}
        for ftr_key in self.feature_sets.keys():
            des, ts, pol = sp_utils.parse_feature_key(ftr_key)
            if des.startswith('cal'):
                caltime = datetime.datetime.strptime(ts, self.time_format)
                cals[caltime] = ftr_key

    def apply_cal(self):
        """
        This assumes that all of the cal info is in feature_sets.  It finds all of
        the cal info, then goes through and applies the cal file preceding it in time.
        If one doesn't exist, it doesn't modify.
        """
        cals = self.get_cal_feature_keys()

        cals_sorted = sorted(list(cals.keys()))
        for ftr_key in self.feature_sets.keys():
            des, ts, pol = sp_utils.parse_feature_key(ftr_key)
            if des.startswith('dat'):
                ftrtime = datetime.datetime.strptime(ts, self.time_format)
                for dtype in self.feature_sets[ftr_key].data_spectral_class_fields:
                    this_cal = None
                    try:
                        this_data = np.array(getattr(self.feature_sets[ftr_key], dtype))
                    except AttributeError:
                        continue
                    for ct in cals_sorted:
                        cdes, cts, cpol = sp_utils.parse_feature_key(cals[ct])
                        if ftrtime > ct and cpol == pol:
                            try:
                                this_cal = np.array(getattr(self.feature_sets[cals[ct]], dtype))
                            except AttributeError:
                                continue
                            break
                    # this_cal will either hold array or string saying something like 'stop' or 'none'
                    if this_cal is not None and not isinstance(this_call, six.string_types):
                        setattr(self.feature_sets[ftr_key], dtype, list(this_data + this_cal))

    def ridz_out_filename(self, idkey, directory='./'):
        # Get overall meta-data for filename
        th_for_fn = ''
        if self.threshold is not None:
            th_for_fn = self.threshold
            if abs(self.threshold) < 1.0:
                th_for_fn *= 100.0
            th_for_fn = 'T{:.0f}'.format(th_for_fn)
        if self.peaked_on is not None:
            pk_for_fn = self.peaked_on[:3]
        else:
            pk_for_fn, th_for_fn = 'None', ''
        if self.data_only:
            pk_for_fn, th_for_fn = 'None', ''
        fn = "{}_{}.{}.n{}.{}{}.ridz".format(idkey, self.feature_module_name, self.timestamp_first,
                                             self.nsets, pk_for_fn, th_for_fn)
        fn = os.path.join(directory, fn)
        return fn

    def process_files(self, ident, ptype=None, directory='.', data=[0, -1], peak_on=None,
                      data_only=False, sets_per_pol=10000, keep_data=False, show_progress=False):
        """
        Sets up for processing spfiles and or pocofiles.

        Parameters:
        ------------
        ident:  identifier for poco files to process
        ptype:  spfile or pocofile if known
        directory:  directory where files reside and ridz files get written
        data:  indices of events to be saved as data spectra
        peak_on:  feature_component on which to find peaks, default order if None
        data_only:  flag if only saving data and not peaks
        sets_per_pol:  number of feature_sets per pol per ridz file (i.e. number of timestamps/pol/file)
        keep_data:  if False (default) it will delete the processed files, otherwise it will keep
        show_progress:  if True it will display some progress cues, default is False
        """
        if ptype is None:
            print("NEED TO FIND OUT IF POCO OR SPECTRUM FILES")
            print("if ident =='all' and both in there set ptype to 'all'")
            print("can also set ident to appropriate list of idents...(?)")
            raise ValueError('But for now just fail.')
        self.data_only = data_only
        if ptype.lower() in ('all', 'spfile'):
            self.spfile_proc(ident=ident, directory=directory, data=data, peak_on=peak_on, data_only=data_only,
                             sets_per_pol=sets_per_pol, keep_data=keep_data, show_progress=show_progress)
        if ptype.lower() in ('all', 'poco'):
            self.poco_proc(ident=ident, directory=directory, data=data, peak_on=peak_on, data_only=data_only,
                           sets_per_pol=sets_per_pol, keep_data=keep_data, show_progress=show_progress)

    # ##############################################################################################
    # ################################   POCO processing section   #################################
    # ##############################################################################################
    def poco_proc(self, ident, directory='.', data=[0, -1], peak_on=None,
                  data_only=False, sets_per_pol=10000, keep_data=False, show_progress=False):
        """
        This processes the poco npz files.

        Parameters:
        ------------
        ident:  identifier for poco files to process
        directory:  directory where files reside and ridz files get written
        data:  indices of events to be saved as data spectra
        peak_on:  feature_component on which to find peaks, default order if None
        data_only:  flag if only saving data and not peaks
        sets_per_pol:  number of feature_sets per pol per ridz file (i.e. number of timestamps/pol/file)
        keep_data:  if False (default) it will delete the processed files, otherwise it will keep
        show_progress:  if True it will display some progress cues, default is False
        """
        import glob
        if show_progress:
            import tabulate
        self.show_progress = show_progress
        self.time_format = "%Y%m%d-%H%M%S"
        files = glob.glob('{}/{}*.npz'.format(directory, ident))
        if not len(files):
            raise ValueError('{} identifier yielded no files.'.format(ident))
        files.sort()
        self.reader('{}/{}.ridm'.format(directory, ident))
        antennas = self.antenna.split(',')
        poco_pols = ['E', 'N']
        cadence = len(antennas) * len(poco_pols)

        ant = {}
        data_file = np.load(files[0])
        self.freq = list(data_file['frequencies'])
        self.fmin = self.freq[0]
        self.fmax = self.freq[-1]
        self.receiver = str(data_file['source'])
        num_antennas_in_npz = max(data_file['ant1_array']) + 1
        if num_antennas_in_npz != len(antennas):
            raise ValueError("npz and ridm antenna count doesn't match.")
        print("Receiver {} with {} freqs from {} - {} {}".format(self.receiver, len(self.freq), self.fmin, self.fmax, self.freq_unit))
        amap = []
        for a in antennas:
            ant[a] = {}
            for p in poco_pols:
                ant[a][p] = {'header': {}, 'times': [], 'len': 0, 'val': [], 'maxhold': []}
                amap.append((a, p))
        files_this_pass = set()
        if self.show_progress:
            print("Reading in npz files.")
        for fname in files:
            files_this_pass.add(fname)
            data_file = np.load(fname)
            if str(data_file['source']) != self.receiver:
                raise ValueError("{} does not match {}.".format(str(data_file['source']), self.receiver))
            n = -1
            for k, t in enumerate(data_file['times']):
                ap = k % cadence
                if not ap:
                    n += 1
                a, p = amap[ap]
                tstamp = datetime.datetime.utcfromtimestamp(t)
                try:
                    ant[a][p]['val'].append(list(data_file['average'][n, :, ap]))
                    ant[a][p]['maxhold'].append(list(data_file['max_hold'][n, :, ap]))
                    ant[a][p]['times'].append(tstamp)
                except IndexError:
                    continue
        # Set antenna-based chunking
        chunks = {}
        for a in antennas:
            amax = 0
            for p in poco_pols:
                lnap = len(ant[a][p]['times'])
                ant[a][p]['len'] = lnap
                if lnap > amax:
                    amax = lnap
            chunks[a] = list(range(0, amax, sets_per_pol))
            if chunks[a][-1] != amax:
                chunks[a].append(amax)
                # Below fails for 'data' argument
                # if len(chunks[a]) > 1 and float(amax - chunks[a][-1]) / amax < 0.15:
                #     chunks[a][-1] = amax
                # else:
                #     chunks[a].append(amax)

        if self.show_progress:
            print("Processing npz into ridz.")
            headers = ['ant']
            for p in poco_pols:
                headers.append(p)
            headers.append('chunks')
            tabl = []
            for a in antennas:
                row = [a]
                for p in poco_pols:
                    row.append(ant[a][p]['len'])
                row.append(len(chunks[a]) - 1)
                tabl.append(row)
            print(tabulate.tabulate(tabl, headers=headers, tablefmt='grid'))

        if self.peaked_on is None:
            self.peaked_on = 'maxhold'
        if peak_on is not None and peak_on in self.feature_components:
            if self.peaked_on != peak_on:
                self.comment += "\nPeaked on user-supplied {} rather than {}.".format(peak_on, self.peaked_on)
            self.peaked_on = peak_on
        for a in antennas:  # Now write the data to ridz files...
            self.antenna = a
            if self.show_progress:
                print("\nProcessing antenna {}:".format(a), end=' ')
            for ctr in range(len(chunks[a]) - 1):
                L0, L1 = chunks[a][ctr], chunks[a][ctr + 1]
                if self.show_progress:
                    print("{}/{}".format(ctr + 1, len(chunks[a]) - 1), end=' ')
                self.feature_sets = {}  # Reset the features sets for new files.
                self.nsets = 0
                self.timestamp_first = datetime.datetime(year=2030, month=1, day=1)
                self.timestamp_last = datetime.datetime(year=2001, month=12, day=31)
                for p in poco_pols:
                    if isinstance(data, list):
                        for i in data:
                            if L0 > len(ant[a][p]['times']):
                                break
                            if i > len(ant[a][p]['times'][L0:L1]) - 1:
                                break
                            if L1 > len(ant[a][p]['times']):
                                L1 = len(ant[a][p]['times'])
                            if i < 0:
                                IL = L1 + i
                            else:
                                IL = L0 + i
                            if ant[a][p]['times'][IL] < self.timestamp_first:
                                self.timestamp_first = ant[a][p]['times'][IL]
                            if ant[a][p]['times'][IL] > self.timestamp_last:
                                self.timestamp_last = ant[a][p]['times'][IL]
                            timestamp = datetime.datetime.strftime(ant[a][p]['times'][IL], self.time_format)
                            feature_tag = 'data:{}:{}'.format(timestamp, p)
                            self.feature_sets[feature_tag] = Spectral(polarization=p)
                            self.feature_sets[feature_tag].freq = '@'
                            self.feature_sets[feature_tag].val = copy.copy(ant[a][p]['val'][IL])
                            self.feature_sets[feature_tag].maxhold = copy.copy(ant[a][p]['maxhold'][IL])
                            self.nsets += 1
                    if not data_only:
                        for i, t in enumerate(ant[a][p]['times'][L0:L1]):
                            IL = L0 + i
                            if t < self.timestamp_first:
                                self.timestamp_first = t
                            if t > self.timestamp_last:
                                self.timestamp_last = t
                            timestamp = datetime.datetime.strftime(t, self.time_format)
                            feature_tag = '{}:{}'.format(timestamp, p)
                            self.feature_sets[feature_tag] = Spectral(polarization=p)
                            find_peaks = Spectral()
                            find_peaks.freq = self.freq
                            find_peaks.val = ant[a][p][self.peaked_on][IL]
                            self.peak_det(find_peaks, delta=self.delta)
                            self.find_bw()
                            self.feature_sets[feature_tag].freq = list(np.array(self.hipk_freq)[self.hipk])
                            self.feature_sets[feature_tag].val = list(np.array(ant[a][p]['val'][IL])[self.hipk])
                            self.feature_sets[feature_tag].maxhold = list(np.array(ant[a][p]['maxhold'][IL])[self.hipk])
                            self.feature_sets[feature_tag].bw = self.hipk_bw
                            self.nsets += 1
                self.timestamp_first = datetime.datetime.strftime(self.timestamp_first, self.time_format)
                self.timestamp_last = datetime.datetime.strftime(self.timestamp_last, self.time_format)
                fn = self.ridz_out_filename(a, directory)
                self.writer(fn)
        if not keep_data:
            # ... delete processed files
            for x in files_this_pass:
                os.remove(x)
        print("\nDone")

    # ##############################################################################################
    # ############################   Spectrum file processing section   ############################
    # ##############################################################################################
    def spfile_proc(self, ident, directory='.', data=[0, -1], peak_on=None, data_only=False,
                    sets_per_pol=10000, keep_data=False, show_progress=False):
        """
        This is the standard method to process spectrum files in a directory to
        produce ridz files.  The module has an "outer loop" that is meant to handle
        ongoing file-writing along with file reading.  It will quit when it has
        run out of appropriate files.

        Format of the spectrum filename (_peel_filename below):
        identifier.{time-stamp}.feature_component.polarization

        Parameters:
        ------------
        directory:  directory where files reside and ridz files get written
        ident:  idents to do.  If 'all' processes all
        data:  indices of events to be saved as data spectra
        peak_on:  feature_component on which to find peaks, default order if None
        data_only:  flag if only saving data and not peaks
        sets_per_pol:  number of feature_sets per pol per ridz file (i.e. number of timestamps/pol/file)
        keep_data:  if False (default) it will delete the processed files, otherwise it will keep
        show_progress:  if True it will display some progress cues, default is False
        """
        self.show_progress = show_progress

        # Get file_idp FileSet of files to do
        file_idp = {}  # keyed set of classes for file-sets
        for af in sorted(os.listdir(directory)):
            if af[0] == '.':
                continue
            fnd = _peel_filename(af, self.feature_components)
            if not fnd['spfile']:
                continue
            if not len(fnd) or\
                    fnd['polarization'].lower() not in self.all_lowercase_polarization_names or\
                    fnd['feature_component'] not in self.feature_components:
                continue
            use_it = False
            if isinstance(ident, six.string_types):
                if ident == 'all' or fnd['ident'] == ident:
                    use_it = True
            elif isinstance(ident, list) and fnd['ident'] in ident:
                use_it = True
            if use_it:
                idkey = fnd['ident']
                pol = fnd['polarization']
                feco = fnd['feature_component']
                if idkey not in file_idp:
                    file_idp[idkey] = FileSet(fnd['ident'])
                file_idp[idkey].included_feature_components.add(feco)
                if pol not in file_idp[idkey].timestamps:
                    file_idp[idkey].timestamps[pol] = set()
                file_idp[idkey].timestamps[pol].add(fnd['timestamp'])
                if pol not in getattr(file_idp[idkey], feco):
                    getattr(file_idp[idkey], feco)[pol] = {}
                getattr(file_idp[idkey], feco)[pol][fnd['timestamp']] = af
        # Now process that dictionary
        for idkey in file_idp:
            #  This is the start of one id
            file_idp[idkey].chunk_it(sets_per_pol)
            bp = file_idp[idkey].biggest_pol

            for i in range(bp['len']):
                # This is start of one file
                self.fmin = 1E9
                self.fmax = -1E9
                self.timestamp_first, self.timestamp_last = file_idp[idkey].chunked_time_limits[i][0], file_idp[idkey].chunked_time_limits[i][-1]
                self.feature_sets = {}  # Reset the features sets for new file.
                self.nsets = 0
                files_this_pass = set()
                for pol in file_idp[idkey].chunked:
                    try:
                        chunk_ts_list = file_idp[idkey].chunked[pol][i]
                    except IndexError:
                        chunk_ts_list = []
                    # This is the start of one feature_set
                    # ... get data spectrum files
                    if isinstance(data, list):
                        for j in data:
                            try:
                                ts = chunk_ts_list[j]
                            except IndexError:
                                break
                            fnargs = {}
                            for feco in file_idp[idkey].included_feature_components:
                                try:
                                    fnargs[feco] = os.path.join(directory, getattr(file_idp[idkey], feco)[pol][ts])
                                    files_this_pass.add(fnargs[feco])
                                except KeyError:
                                    continue
                            if len(fnargs):
                                feature_tag = 'data:{}:'.format(ts)
                                self.get_feature_set_from_files(feature_tag, pol, **fnargs)
                    if not data_only:
                        # ... get feature_sets
                        for ts in chunk_ts_list:
                            fnargs = {}
                            for feco in file_idp[idkey].included_feature_components:
                                try:
                                    fnargs[feco] = os.path.join(directory, getattr(file_idp[idkey], feco)[pol][ts])
                                    files_this_pass.add(fnargs[feco])
                                except KeyError:
                                    continue
                            if len(fnargs):
                                feature_tag = '{}:'.format(ts)
                                self.get_feature_set_from_files(feature_tag, pol, peak_on=peak_on, **fnargs)
                # Write the ridz file
                fn = self.ridz_out_filename(idkey)
                filename = os.path.join(directory, fn)
                self.writer(filename)
                if not keep_data:
                    # ... delete processed files
                    for x in files_this_pass:
                        os.remove(x)

    def get_feature_set_from_files(self, feature_tag, polarization, peak_on=None, **fnargs):
        """
        Get the feature sets for current iteration from individual spectrum files
        with naming convention as realized in module '_peel_filename' below
        **fnargs are filenames for self.feature_components {"feature_component": <filename>}
        """
        fset_name = feature_tag + polarization
        self.feature_sets[fset_name] = Spectral(polarization=polarization)
        self.feature_sets[fset_name].freq = []
        if len(fnargs) > 1:
            check_timestamps_for_match = _check_timestamps_for_match(**fnargs)
            if len(check_timestamps_for_match):
                self.feature_sets[fset_name].comment += check_timestamps_for_match
        if self.show_progress:
            print("Processing {}".format(fset_name))
        spectra = {}
        for fc, sfn in six.iteritems(fnargs):
            if fc not in self.feature_components or sfn is None:
                continue
            spectra[fc] = Spectral()
            if self.show_progress:
                print("\t{}".format(sfn))
            _spectrum_reader(sfn, spectra[fc], polarization)
            fext = min(spectra[fc].freq)
            if fext < self.fmin:
                self.fmin = fext
            fext = max(spectra[fc].freq)
            if fext > self.fmax:
                self.fmax = fext
            if is_spectrum(feature_tag):
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
        if is_spectrum(feature_tag):
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
        # self.peak_finder(spectra[fc])  # old version not used anymore
        self.peak_det(spectra[fc], delta=self.delta)
        self.find_bw()

        # put values in feature set dictionary
        self.feature_sets[fset_name].freq = list(np.array(self.hipk_freq)[self.hipk])
        if len(spectra[fc].comment):
            self.feature_sets[fset_name].comment += spectra[fc].comment
        for fc, sfn in six.iteritems(fnargs):
            if sfn is None:
                continue
            try:
                setattr(self.feature_sets[fset_name], fc, list(np.array(spectra[fc].val)[self.hipk]))
            except IndexError:
                pass
        self.feature_sets[fset_name].bw = self.hipk_bw


def _spectrum_reader(filename, spec, polarization=None):
    """
    This reads in an ascii spectrum file.
    """
    if filename is None:
        return
    if polarization is not None:
        spec.polarization = polarization
    spec.val = []
    with open(filename, 'r') as f:
        for line in f:
            if line.strip()[0] == ':':
                spec.comment += (line.strip()[1:] + '\n')
                continue
            data = [float(x) for x in line.split()]
            spec.freq.append(data[0])
            spec.val.append(data[1])


def _spectrum_plotter(x, y, fmt=None, is_spectrum=False, figure_name=None):
    if not len(x) or not len(y):
        return None, None
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


def _peel_filename(v, fclist=None):
    if v == 'filename_format_help':
        s = "The filename format convention is:\n"
        s += "\tidentifier.{time-stamp}.feature_component.polarization\n"
        s += "\ni.e. x=filename.split(.) has:\n"
        s += "\tx[0]:  arbitrary identifier\n"
        s += "\tx[...]: time-stamp (may have .'s in it)\n"
        s += "\tx[-2]:  feature_component ('maxhold', 'minhold', 'val')\n"
        s += "\tx[-1]:  polarization\n"
        return s
    if v is None:
        return {}
    s = v.split('/')[-1].split('.')
    fnd = {'filename': '.'.join(s), 'tag': s[-1].lower(), 'spfile': False, 'ridfile': False, 'npzfile': False}
    if len(s) < 2:
        return fnd
    fnd['ident'] = s[0]
    if 'rid' in fnd['tag']:
        fnd['ridfile'] = True
    if 'npz' in fnd['tag']:
        fnd['npzfile'] = True
    if len(s) < 3:
        return fnd
    fnd['timestamp'] = '.'.join(s[1:-2])
    if len(s) < 4:
        return fnd
    fnd['feature_component'] = s[-2].lower()
    fnd['polarization'] = s[-1].upper()
    if fclist is None:
        return fnd
    fclist = [x.lower() for x in fclist]
    for fc in fclist:
        if fnd['feature_component'] in fc:
            fnd['feature_component'] = fc
            fnd['spfile'] = True
            return fnd
    return fnd


def _get_timestr_from_ftr_key(fkey):
    if ':' in fkey:
        return fkey.split(':')[-2]
    if '.' in fkey:
        s = fkey.split('.')
        if is_spectrum(fkey):
            return '.'.join(s[1:-1])
        else:
            return '.'.join(s[0:-1])
    return None


def _check_timestamps_for_match(**fnargs):
    list_of = []
    set_of = set()
    for fc, sfn in six.iteritems(fnargs):
        fnd = _peel_filename(sfn)
        if len(fnd):
            ts = fnd['timestamp']
            list_of.append(ts)
            set_of.add(ts)
    if len(list_of) == len(set_of):
        return "Timestamps don't match: " + ','.join(fnargs.values()) + '\n'
    return ''
