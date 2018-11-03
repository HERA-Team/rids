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

    def set_feature_keys(self, keys=None):
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
        for fs in sorted_ftr_keys:
            if spectrum_peak.is_spectrum(fs):
                self.feature_keys.append(fs)

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

    def set_time_range(self, t_range=None):
        """
        Makes t_range a class variable self.t_range, updating if supplied None
            Sets self.time_0 as earliest supplied datetime
            Sets self.time_N as latest supplied datetime
            Sets self.t_range, range to plot as datetime pair

        Parameters:
        ------------
        t_range:  time range, as datetime.datetime
        """
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
        elif not isinstance(t_range[0], datetime.datetime) or not isinstance(t_range[1], datetime.datetime):
            raise ValueError("t_range types must be datetimes")
        elif t_range[0] < t0:
            print("Requested time before data.  Setting start time to {}".format(t0))
            t_range[0] = t0
        elif t_range[1] > tn:
            print("Requested time after data.  Setting stop time to {}".format(tn))
            t_range[1] = tn
        elif t_range[0] > tn or t_range[1] > t0:
            raise ValueError("Times not spanned by data.")
        self.t_range = t_range

    def time_filter(self, feature_components):
        """
        Gets the final time-filtered keys to use.
        Sets self.time_space, the time space used
        Sets self.used_keys, the keys used
        Sets self.nominal_t_step, the time step (nominal)
        """
        self.feature_components = feature_components
        self.time_space = {}
        self.used_keys = {}
        self.nominal_t_step = 1E6
        for fc in feature_components:
            self.time_space[fc] = []
            self.used_keys[fc] = []
            for i, fs in enumerate(self.feature_keys):
                ts = self.rid.get_datetime_from_timestamp(spectrum_peak._get_timestr_from_ftr_key(fs))
                if ts < self.t_range[0] or ts > self.t_range[1]:
                    continue
                lents = len(self.time_space[fc])
                self.time_space[fc].append(ts)
                self.used_keys[fc].append(fs)
                if lents and (ts - self.time_space[fc][i - 1]) < self.nominal_t_step:
                    self.nominal_t_step = (ts - self.time_space[fc][i - 1])

    def process(self, wf_time_fill=None, show_edits=True, total_power_only=False):
        """
        Process the rid data to make the plots.
        Sets self.wf:  all of the data in waterfall format
        Sets self.extrema:  time and freq extrema
        Sets self.feature_components:  makes global

        Parameters:
        ------------
        feature_components:  feature components to use
        wf_file_fill:  value to use if wish to/need to fill in gaps in the data for waterfall plot.
        show_edits:  If True, shows the plot of what it did to make the waterfall columns align.
        total_power_only:  if True, it doesn't make/save the wf plot (if memory an issue)
        """
        if total_power_only:
            wf_time_fill = None
        else:
            self.wf = {}
            self.extrema = {}
        self.total_power = {}
        lfrq = len(self.full_freq)
        for fc in self.feature_components:
            if not total_power_only:
                self.wf[fc] = []
                self.extrema[fc] = {'f': Namespace(), 't': Namespace()}
            self.total_power[fc] = []
            fadd = []
            ftrunc = []
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
                tp = 10.0 * np.log10(np.sum(np.power(10.0, np.array(xxx) / 10.0)))
                self.total_power[fc].append(tp)
                if i and wf_time_fill is not None:
                    delta_t = self.time_space[fc][i] - self.time_space[fc][i - 1]
                    if delta_t > 1.2 * self.nominal_t_step:
                        num_missing = int(delta_t / self.nominal_t_step)
                        for j in range(num_missing):
                            self.wf[fc].append(wf_time_fill * np.ones(len(self.freq_space)))
                if not total_power_only:
                    self.wf[fc].append(y[self.lo_chan:self.hi_chan])
            if not total_power_only:
                self.wf[fc] = np.array(self.wf[fc])
                t_lo = self.rid.get_datetime_from_timestamp(spectrum_peak._get_timestr_from_ftr_key(self.used_keys[fc][0]))
                t_hi = self.rid.get_datetime_from_timestamp(spectrum_peak._get_timestr_from_ftr_key(self.used_keys[fc][-1]))
                self.extrema[fc]['t'].lo = sp_utils.get_duration_in_std_units((t_lo - self.time_0).total_seconds(), use_unit=self.ts_unit)[0]
                self.extrema[fc]['t'].hi = sp_utils.get_duration_in_std_units((t_hi - self.time_0).total_seconds(), use_unit=self.ts_unit)[0]
                self.extrema[fc]['f'].lo = self.full_freq[self.lo_chan]
                self.extrema[fc]['f'].hi = self.full_freq[self.hi_chan]
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

        for fc in self.feature_components:
            lims = [self.extrema[fc]['f'].lo, self.extrema[fc]['f'].hi, self.extrema[fc]['t'].hi, self.extrema[fc]['t'].lo]
            plt.figure(fc)
            if title is not None:
                plt.title(title)
            plt.imshow(self.wf[fc], aspect='auto', extent=lims)
            plt.xlabel('Freq [{}]'.format(self.rid.freq_unit))
            plt.ylabel('{} after {}'.format(self.ts_unit, self.time_0))
            plt.colorbar()

    def raw_2D_plot(self, plot_type, legend=False, all_same_plot=False, title=None):
        # plot data (other)
        for fc in self.feature_components:
            if not all_same_plot:
                plt.figure(fc)
            if title is not None:
                plt.title(title)
            if plot_type == 'stream':
                for i, f in enumerate(self.freq_space):
                    freq_label = "{:.3f} {}".format(f, self.rid.freq_unit)
                    if all_same_plot:
                        freq_label = fc + ': ' + freq_label
                    plt.plot(self.time_space[fc], self.wf[fc][:, i], label=freq_label)
                print("Number of plots: {}".format(len(self.freq_space)))
                plt.xlabel('{} after {}'.format(self.ts_unit, self.time_0))
                plt.ylabel('Power [{}]'.format(self.rid.val_unit))
            elif plot_type == 'stack':
                for i, ts in enumerate(self.time_space[fc]):
                    if self.specific_keys:
                        time_label = self.used_keys[fc][i]
                    else:
                        time_label = "{:.4f} {}".format(ts, self.ts_unit)
                    if all_same_plot:
                        time_label = fc + ': ' + time_label
                    plt.plot(self.freq_space, self.wf[fc][i, :], label=time_label)
                print("Number of plots: {}".format(len(self.time_space)))
                plt.xlabel('Freq [{}]'.format(self.rid.freq_unit))
                plt.ylabel('Power [{}]'.format(self.rid.val_unit))
            if legend:
                plt.legend()

    def raw_totalpower_plot(self, legend=False, title=None):
        plt.figure('Total Power')
        if title is not None:
            plt.title(title)
        for fc in self.feature_components:
            plt.plot(self.time_space[fc], self.total_power[fc])
        plt.xlabel('{} after {}'.format(self.ts_unit, self.time_0))
        plt.ylabel('Power [{}]'.format(self.rid.val_unit))
