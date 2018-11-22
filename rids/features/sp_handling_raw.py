# _*_ coding: utf-8 _*_
# Copy 2018 the HERA Project
# Licensed under the 2-clause BSD license


"""
SPHandle:  Spectrum_Peak Handling

This has various modules to handle and view ridz files with the SpectrumPeak feature module.

"""
from __future__ import print_function, absolute_import, division
from . import spectrum_peak
from . import sp_utils
from argparse import Namespace
import numpy as np
import matplotlib.pyplot as plt
import six
import datetime


class SPHandling:

    def __init__(self, rid_class):
        self.sp = spectrum_peak.SpectrumPeak()
        self.rid = rid_class
        self.feature_keys_found = False

    def set_feature_keys(self, pol, keys=None):
        """
        Finds the keys corresponding to spectra, or checks/sorts a given list.
        Sets self.feature_keys, the sorted list of keys of _spectrum_ data.
        Sets self.specific_keys to True, if keys provided and fewer than 10.  Used for plotting legend

        Parameters:
        ------------
        keys:  desired keys to check - None or 'all' returns all spectrum keys
        """
        self.specific_keys = False
        if isinstance(keys, six.string_types):
            keys = keys.split(',')
        if keys is None or keys[0] == 'all':
            sorted_ftr_keys = sorted(self.rid.feature_sets.keys())
        else:
            if len(keys) < 10:
                self.specific_keys = True
            sorted_ftr_keys = sorted(keys)
        self.feature_keys = []
        self.polarization = '{}'.format(pol)
        pol = pol.upper()
        for fs in sorted_ftr_keys:
            if pol == self.rid.feature_sets[fs].polarization.upper() and spectrum_peak.is_spectrum(fs):
                self.feature_keys.append(fs)
        self.feature_keys_found = bool(len(self.feature_keys))
        if not self.feature_keys_found:
            print("No keys found for requested values.")

    def set_freq(self, f_range=None):
        """
        Makes f_range a class variable self.f_range, updating if supplied None
        Sets self.full_freq, the "likely" full frequency range (either the
            first one or shared one, if used.)
        Sets self.freq_space, the frequencies used.
        Sets class variable self.chan_size
        Sets class variable self.lo_chan:  index in full_freq of lowest freq
        Sets class variable self.hi_chan:  index in full_freq of highest freq

        Parameters:
        ------------
        f_range:  desired frequency range.  If None, uses entire full_freq array
        """
        if not self.feature_keys_found:
            return None
        self.full_freq = self.rid.feature_sets[self.feature_keys[0]].freq  # chose first one
        if self.full_freq == '@':  # share_freq was set
            self.full_freq = self.rid.freq
        lfrq = len(self.full_freq)
        self.chan_size = (self.full_freq[-1] - self.full_freq[0]) / lfrq
        if f_range is None:
            f_range = [self.full_freq[0], self.full_freq[-1]]
            lo_chan = 0
            hi_chan = -1
        else:
            ch = (self.full_freq[-1] - self.full_freq[0]) / lfrq
            if f_range[0] < self.full_freq[0]:
                lo_chan = 0
            else:
                lo_chan = int((f_range[0] - self.full_freq[0]) / self.chan_size)
            if f_range[1] > self.full_freq[-1]:
                hi_chan = -1
            else:
                hi_chan = int((f_range[1] - self.full_freq[0]) / self.chan_size)
        self.f_range = f_range
        self.lo_chan = lo_chan
        self.hi_chan = hi_chan
        self.freq_space = self.full_freq[lo_chan:hi_chan]

    def set_time_range(self, t_range=None, flip=False):
        """
        Makes t_range a class variable self.t_range, updating if supplied None
            Sets self.time_0 as earliest supplied datetime
            Sets self.time_N as latest supplied datetime
            Sets self.t_range, range to plot as datetime pair

        Parameters:
        ------------
        t_range:  time range, as datetime.datetime list-pair
        flip:  boolean to flip sense of time filter (don't plot within those)
        """
        if not self.feature_keys_found:
            return None
        self.flip = flip
        t0 = self.rid.get_datetime_from_timestamp(spectrum_peak._get_timestr_from_ftr_key(self.feature_keys[0]))
        tn = self.rid.get_datetime_from_timestamp(spectrum_peak._get_timestr_from_ftr_key(self.feature_keys[-1]))
        self.time_0 = t0
        self.time_N = tn
        print("Total data span {} - {}".format(t0, tn))
        if t_range is None:
            self.t_range = [t0, tn]
            return
        if not isinstance(t_range, list):
            raise ValueError("t_range can't be {}".format(type(t_range)))
        if t_range[0] is None:
            t_range[0] = t0
        if t_range[1] is None:
            t_range[1] = tn
        if not isinstance(t_range[0], datetime.datetime) or not isinstance(t_range[1], datetime.datetime):
            raise ValueError("t_range types must be datetimes")
        if t_range[0] < t0:
            print("Requested time before data.  Setting start time to {}".format(t0))
            t_range[0] = t0
        elif t_range[1] > tn:
            print("Requested time after data.  Setting stop time to {}".format(tn))
            t_range[1] = tn
        elif t_range[0] > tn or t_range[1] < t0:
            raise ValueError("Times not spanned by data.")
        self.t_range = t_range

    def time_filter(self, feature_components):
        """
        Gets the final time-filtered keys to use.  Sets:
            self.used_keys: the keys used in the plot
            self.time_space:  datetimes used in plot
            self.delta_t:  timestep in sec
            self.t_elapsed:  elapsed time since beginning of plot in sec
        """
        if not self.feature_keys_found:
            return None
        self.feature_components = feature_components
        self.used_keys = {}
        self.time_space = {}
        self.delta_t = {}
        self.t_elapsed = {}
        for fc in feature_components:
            self.used_keys[fc] = []
            self.time_space[fc] = []
            self.delta_t[fc] = [0.0]
            self.t_elapsed[fc] = [0.0]
            for i, fs in enumerate(self.feature_keys):
                ts = self.rid.get_datetime_from_timestamp(spectrum_peak._get_timestr_from_ftr_key(fs))
                if not self.flip:
                    if ts < self.t_range[0] or ts > self.t_range[1]:
                        continue
                else:
                    if ts > self.t_range[0] and ts < self.t_range[1]:
                        continue
                self.used_keys[fc].append(fs)
                self.time_space[fc].append(ts)
                if len(self.time_space[fc]) > 1:
                    a = self.time_space[fc][-1]
                    b = self.time_space[fc][-2]
                    dab = (a - b).total_seconds()
                    self.delta_t[fc].append((self.time_space[fc][-1] - self.time_space[fc][-2]).total_seconds())
                    self.t_elapsed[fc].append((self.time_space[fc][-1] - self.time_space[fc][0]).total_seconds())
            self.delta_t[fc] = np.array(self.delta_t[fc])

    def process(self, wf_time_fill=None, show_edits=True, total_power_only=False, unit_conversion='none', csv=None):
        """
        Process the rid data to make the plots.
        Sets self.wf:  all of the data in waterfall format
        Sets self.feature_components:  makes global

        Parameters:
        ------------
        feature_components:  feature components to use
        wf_file_fill:  value to use if wish to/need to fill in gaps in the data for waterfall plot.
        show_edits:  If True, shows the plot of what it did to make the waterfall columns align.
        total_power_only:  if True, it doesn't make/save the wf plot (if memory an issue)
        """
        if not self.feature_keys_found:
            return None
        if total_power_only:
            wf_time_fill = None
        else:
            self.wf = {}
        self.total_power = {}
        lfrq = len(self.full_freq)
        self.unit_conversion = unit_conversion
        calval = 10.0
        if unit_conversion[-1].lower() == 'v':
            calval = 20.0
        for fc in self.feature_components:
            if not total_power_only:
                self.wf[fc] = []
            self.total_power[fc] = []
            fadd = []
            ftrunc = []
            if len(self.delta_t[fc]) < 2:
                min_t_step = 0.0
                std_t_step = 0.0
            else:
                min_t_step = self.delta_t[fc][1:].min()
                std_t_step = self.delta_t[fc][1:].std()
            for i, fs in enumerate(self.used_keys[fc]):
                x, y = spectrum_peak._spectrum_plotter(self.rid.feature_sets[fs].freq, getattr(self.rid.feature_sets[fs], fc), None)
                if x is None:
                    continue
                if len(x) < lfrq:
                    fadd.append(lfrq - len(x))
                    yend = y[-1]
                    for j in range(len(x), lfrq):
                        y.append(yend)
                elif len(x) > lfrq:
                    ftrunc.append(len(x) - lfrq)
                    y = y[:lfrq]
                xxx = np.array(y)[self.lo_chan:self.hi_chan]
                if unit_conversion.startswith('dB') or unit_conversion_startswith('no'):
                    tp = calval * np.log10(np.sum(np.power(10.0, xxx / calval)))
                elif unit_conversion.startswith('lin'):
                    tp = np.sum(xxx)
                self.total_power[fc].append(tp)
                if i and wf_time_fill is not None:
                    if self.delta_t[fc][i] > min_t_step + std_t_step:
                        num_missing = int(self.delta_t[fc][i] / min_t_step)
                        for j in range(num_missing):
                            self.wf[fc].append(wf_time_fill * np.ones(len(self.freq_space)))
                if not total_power_only:
                    self.wf[fc].append(y[self.lo_chan:self.hi_chan])
            if not total_power_only:
                self.wf[fc] = np.array(self.wf[fc])
                if unit_conversion.startswith('dB'):
                    zzz = np.where(self.wf[fc] == 0.0)
                    if len(zzz):
                        self.wf[fc][zzz] = np.max(self.wf[fc])
                        self.wf[fc][zzz] = np.min(self.wf[fc]) / 1000.0
                    self.wf[fc] = calval * np.log10(self.wf[fc])
                elif unit_converion.startswith('linear'):
                    self.wf[fc] = np.power(self.wf[fc] / calval, 10.0)
                if csv:
                    csvfile = '{}_{}'.format(fc, csv)
                    np.savetxt(csvfile, self.wf[fc], delimiter=',')
                    print("Writing {}.".format(csvfile))
            if show_edits:
                plt.figure('max')
                plt.plot(fadd)
                if len(fadd):
                    print("{}: had to add to {} spectra (max {})".format(fc, len(fadd), max(fadd)))
                if len(ftrunc):
                    print("{}: had to truncate {} spectra (max {})".format(fc, len(ftrunc), max(ftrunc)))

    def raw_waterfall_plot(self, title=None):
        """
        Plots the full waterfall
        """
        # Get data and parameters
        if not self.feature_keys_found:
            return None

        for fc in self.feature_components:
            t_lo = 0.0
            t_hi = sp_utils.get_duration_in_std_units(self.t_elapsed[fc][-1])
            ts_unit = t_hi[1]
            t_hi = t_hi[0]
            f_lo = self.full_freq[self.lo_chan]
            f_hi = self.full_freq[self.hi_chan]
            lims = [f_lo, f_hi, t_hi, t_lo]
            plt.figure(fc)
            if title is None:
                title = '{} after {}'.format(ts_unit, self.time_space[fc][0])
            title += ': {} pol'.format(self.polarization)
            plt.title(title)
            plt.imshow(self.wf[fc], aspect='auto', extent=lims)
            plt.xlabel('Freq [{}]'.format(self.rid.freq_unit))
            plt.ylabel('{} after {}'.format(ts_unit, self.time_space[fc][0]))
            plt.colorbar()

    def raw_2D_plot(self, plot_type, legend=False, all_same_plot=False, title=None):
        # plot data (other)
        if not self.feature_keys_found:
            return None
        if self.unit_conversion.startswith('no'):
            punit = self.rid.val_unit
        else:
            punit = self.unit_conversion
        for fc in self.feature_components:
            if not all_same_plot:
                plt.figure(fc)
            if title is None:
                title = 'Raw 2D {}'.format(fc)
            title += ': {} pol'.format(self.polarization)
            plt.title(title)
            ts_unit = sp_utils.get_duration_in_std_units(self.t_elapsed[fc][-1])[1]
            if plot_type == 'stream':
                for i, f in enumerate(self.freq_space):
                    freq_label = "{:.3f} {}".format(f, self.rid.freq_unit)
                    if all_same_plot:
                        freq_label = fc + ': ' + freq_label
                    plt.plot(self.used_keys[fc], self.wf[fc][:, i], label=freq_label)
                print("Number of plots: {}".format(len(self.freq_space)))
                plt.xlabel('{} after {}'.format(ts_unit, self.time_space[fc][0]))
                plt.ylabel('Power [{}]'.format(punit))
            elif plot_type == 'stack':
                for i, ts in enumerate(self.used_keys[fc]):
                    if self.specific_keys:
                        time_label = self.used_keys[fc][i]
                    else:
                        time_label = "{} {}".format(ts, ts_unit)
                    if all_same_plot:
                        time_label = fc + ': ' + time_label
                    plt.plot(self.freq_space, self.wf[fc][i, :], label=time_label)
                print("Number of plots: {}".format(len(self.used_keys)))
                plt.xlabel('Freq [{}]'.format(self.rid.freq_unit))
                plt.ylabel('Power [{}]'.format(punit))
            if legend:
                plt.legend()

    def raw_totalpower_plot(self, legend=False, title=None):
        if not self.feature_keys_found:
            return None
        if self.unit_conversion.startswith('no'):
            punit = self.rid.val_unit
        else:
            punit = self.unit_conversion
        plt.figure('Total Power')
        if title is None:
            title = 'Total power'
        title += ': {} pol'.format(title)
        plt.title(title)
        ts_unit = sp_utils.get_duration_in_std_units(self.t_elapsed[fc][-1])[1]
        for fc in self.feature_components:
            plt.plot(self.used_keys[fc], self.total_power[fc])
        plt.xlabel('{} after {}'.format(ts_unit, self.time_space[fc][0]))
        plt.ylabel('Power [{}]'.format(punit))
