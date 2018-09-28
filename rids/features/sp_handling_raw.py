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


class SPHandling:

    def __init__(self):
        self.sp = spectrum_peak.SpectrumPeak()

    def get_feature_set_keys(self, rid, keys=None):
        if isinstance(keys, six.string_types):
            keys = [keys]
        if keys is None or keys[0] == 'all':
            sorted_ftr_keys = sorted(rid.feature_sets.keys())
        else:
            sorted_ftr_keys = sorted(keys)
        feature_keys = []
        for fs in sorted_ftr_keys:
            if spectrum_peak.is_spectrum(fs):
                feature_keys.append(fs)
        return sorted(feature_keys)

    def get_freq_chan(self, rid, feature_keys, f_range=None):
        # Get freqs and channels
        freq = rid.feature_sets[feature_keys[0]].freq  # chose first one
        if freq == '@':  # share_freq was set
            freq = rid.freq
        lfrq = len(freq)
        chan_size = (freq[-1] - freq[0]) / lfrq
        if f_range is None:
            f_range = [freq[0], freq[-1]]
            lo_chan = 0
            hi_chan = -1
        else:
            ch = (freq[-1] - freq[0]) / lfrq
            if f_range[0] < freq[0]:
                lo_chan = 0
            else:
                lo_chan = int((f_range[0] - freq[0]) / chan_size)
            if f_range[1] > freq[-1]:
                hi_chan = -1
            else:
                hi_chan = int((f_range[1] - freq[0]) / chan_size)
        return freq, lo_chan, hi_chan

    def raw_data_plot(self, rid, feature_components, plot_type='waterfall', f_range=None, t_range=None,
                      wf_time_fill=None, legend=False, keys=None, all_same_plot=False):
        """
        Give a date range to make a waterfall with whatever raw data is in the file, or in specific keys
        """
        feature_keys = self.get_feature_set_keys(rid, keys=keys)
        t0 = rid.get_datetime_from_timestamp(spectrum_peak._get_timestr_from_ftr_key(feature_keys[0]))
        tn = rid.get_datetime_from_timestamp(spectrum_peak._get_timestr_from_ftr_key(feature_keys[-1]))
        print("Data span {} - {}".format(t0, tn))
        duration, ts_unit = sp_utils.get_duration_in_std_units((tn - t0).total_seconds())
        if t_range is None:
            t_range = [0, duration]
        freq, lo_chan, hi_chan = self.get_freq_chan(rid, feature_keys, f_range=f_range)
        lfrq = len(freq)
        freq_space = freq[lo_chan:hi_chan]

        # Get time and final key list
        time_space = {}
        used_keys = {}
        nominal_t_step = 1E6
        for fc in feature_components:
            time_space[fc] = []
            used_keys[fc] = []
            for i, fs in enumerate(feature_keys):
                ts = (rid.get_datetime_from_timestamp(spectrum_peak._get_timestr_from_ftr_key(fs)) - t0).total_seconds()
                ts, x = sp_utils.get_duration_in_std_units(ts, use_unit=ts_unit)
                if ts < t_range[0] or ts > t_range[1]:
                    continue
                lents = len(time_space[fc])
                time_space[fc].append(ts)
                used_keys[fc].append(fs)
                if lents and (ts - time_space[fc][i - 1]) < nominal_t_step:
                    nominal_t_step = (ts - time_space[fc][i - 1])

        # Get data and parameters
        wf = {}
        extrema = {}
        for fc in feature_components:
            wf[fc] = []
            extrema[fc] = {'f': Namespace(), 't': Namespace()}
            fadd = []
            ftrunc = []
            for i, fs in enumerate(used_keys[fc]):
                x, y = spectrum_peak._spectrum_plotter(rid.feature_sets[fs].freq, getattr(rid.feature_sets[fs], fc), None)
                if x is None:
                    continue
                if len(x) < lfrq:
                    fadd.append(lfrq - len(x))
                    yend = y[-1]
                    for i in range(len(x), lfrq):
                        y.append(yend)
                elif len(x) > lfrq:
                    ftrunc.append(len(x) - lfrq)
                    y = y[:lfrq]
                if i and wf_time_fill is not None:
                    delta_t = time_space[fc][i] - time_space[fc][i - 1]
                    if delta_t > 1.2 * nominal_t_step:
                        num_missing = int(delta_t / nominal_t_step)
                        for j in range(num_missing):
                            wf[fc].append(wf_time_fill * np.ones(len(freq_space)))
                wf[fc].append(y[lo_chan:hi_chan])
            wf[fc] = np.array(wf[fc])
            t_lo = rid.get_datetime_from_timestamp(spectrum_peak._get_timestr_from_ftr_key(used_keys[fc][0]))
            t_hi = rid.get_datetime_from_timestamp(spectrum_peak._get_timestr_from_ftr_key(used_keys[fc][-1]))
            extrema[fc]['t'].lo = sp_utils.get_duration_in_std_units((t_lo - t0).total_seconds(), use_unit=ts_unit)[0]
            extrema[fc]['t'].hi = sp_utils.get_duration_in_std_units((t_hi - t0).total_seconds(), use_unit=ts_unit)[0]
            extrema[fc]['f'].lo = freq[lo_chan]
            extrema[fc]['f'].hi = freq[hi_chan]
            if len(fadd):
                print("{}: had to add to {} spectra".format(fc, len(fadd)))
                print("{} ... {}".format(fadd[0], fadd[-1]))
            if len(ftrunc):
                print("{}: had to truncate {} spectra".format(fc, len(ftrunc)))
                print("{} ... {}".format(ftrunc[0], ftrunc[-1]))

        # plot data (waterfall) - and return
        if plot_type == 'waterfall':
            for fc in feature_components:
                lims = [extrema[fc]['f'].lo, extrema[fc]['f'].hi, extrema[fc]['t'].hi, extrema[fc]['t'].lo]
                plt.figure(fc)
                plt.imshow(wf[fc], aspect='auto', extent=lims)
                plt.xlabel('Freq [{}]'.format(rid.freq_unit))
                plt.ylabel('{} after {}'.format(ts_unit, t0))
                plt.colorbar()
            return

        # plot data (other)
        for fc in feature_components:
            if not all_same_plot:
                plt.figure(fc)
            if plot_type == 'stream':
                for i, f in enumerate(freq_space):
                    freq_label = "{:.3f} {}".format(f, rid.freq_unit)
                    if all_same_plot:
                        freq_label = fc + ': ' + freq_label
                    plt.plot(time_space[fc], wf[fc][:, i], label=freq_label)
                print("Number of plots: {}".format(len(freq_space)))
                plt.xlabel('{} after {}'.format(ts_unit, t0))
                plt.ylabel('Power [{}]'.format(rid.val_unit))
            elif plot_type == 'stack':
                for i, ts in enumerate(time_space):
                    if keys is None:
                        time_label = "{:.4f} {}".format(ts, ts_unit)
                    else:
                        time_label = used_keys[i]
                    if all_same_plot:
                        time_label = fc + ': ' + time_label
                    plt.plot(freq_space, wf[fc][i, :], label=time_label)
                print("Number of plots: {}".format(len(time_space)))
                plt.xlabel('Freq [{}]'.format(rid.freq_unit))
                plt.ylabel('Power [{}]'.format(rid.val_unit))
            if legend:
                plt.legend()
