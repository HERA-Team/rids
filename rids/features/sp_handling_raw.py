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


class SPHandling:

    def __init__(self):
        self.sp = spectrum_peak.SpectrumPeak()

    def raw_data_plot(self, rid, feature_components, plot_type='waterfall', f_range=None, t_range=None,
                      legend=False, keys=None, all_same_plot=False):
        """
        Give a date range to make a waterfall with whatever raw data is in the file, or specific keys
        """
        # Get times
        if keys is None or keys[0] == 'all':
            sorted_ftr_keys = sorted(rid.feature_sets.keys())
        else:
            sorted_ftr_keys = sorted(keys)
        time_keys = []
        for fs in sorted_ftr_keys:
            if spectrum_peak.is_spectrum(fs):
                time_keys.append(fs)
        time_keys = sorted(time_keys)
        t0 = rid.get_datetime_from_timestamp(spectrum_peak._get_timestr_from_ftr_key(time_keys[0]))
        tn = rid.get_datetime_from_timestamp(spectrum_peak._get_timestr_from_ftr_key(time_keys[-1]))
        print("Data span {} - {}".format(t0, tn))
        duration, ts_unit = sp_utils.get_duration_in_std_units((tn - t0).total_seconds())
        if t_range is None:
            t_range = [0, duration]

        # Get freqs
        freq = rid.feature_sets[time_keys[0]].freq  # chose first one
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

        # Get data and parameters
        wf = {}
        used_keys = {}
        dur = {}
        for fc in feature_components:
            wf[fc] = []
            used_keys[fc] = []
            dur[fc] = {'f': Namespace(), 't': Namespace()}
            fadd = []
            ftrunc = []
            for fs in time_keys:
                ts = (rid.get_datetime_from_timestamp(spectrum_peak._get_timestr_from_ftr_key(fs)) - t0).total_seconds()
                ts, x = sp_utils.get_duration_in_std_units(ts, use_unit=ts_unit)
                if ts < t_range[0] or ts > t_range[1]:
                    continue
                x, y = spectrum_peak._spectrum_plotter(rid.feature_sets[fs].freq, getattr(rid.feature_sets[fs], fc), None)
                if x is None:
                    continue
                used_keys[fc].append(fs)
                if len(x) < lfrq:
                    fadd.append(lfrq - len(x))
                    yend = y[-1]
                    for i in range(len(x), lfrq):
                        y.append(yend)
                elif len(x) > lfrq:
                    ftrunc.append(len(x) - lfrq)
                    y = y[:lfrq]
                wf[fc].append(y[lo_chan:hi_chan])
            wf[fc] = np.array(wf[fc])
            t_lo = rid.get_datetime_from_timestamp(spectrum_peak._get_timestr_from_ftr_key(used_keys[fc][0]))
            t_hi = rid.get_datetime_from_timestamp(spectrum_peak._get_timestr_from_ftr_key(used_keys[fc][-1]))
            dur[fc]['t'].lo = sp_utils.get_duration_in_std_units((t_lo - t0).total_seconds(), use_unit=ts_unit)[0]
            dur[fc]['t'].hi = sp_utils.get_duration_in_std_units((t_hi - t0).total_seconds(), use_unit=ts_unit)[0]
            dur[fc]['f'].lo = freq[lo_chan]
            dur[fc]['f'].hi = freq[hi_chan]
            if len(fadd):
                print("{}: had to add to {} spectra".format(fc, len(fadd)))
                print(fadd)
            if len(ftrunc):
                print("{}: had to truncate {} spectra".format(fc, len(ftrunc)))
                print(ftrunc)

        # plot data (waterfall) - and return
        if plot_type == 'waterfall':
            for fc in feature_components:
                plt.figure(fc)
                plt.imshow(wf[fc], aspect='auto', extent=[dur[fc]['f'].lo, dur[fc]['f'].hi, dur[fc]['t'].hi, dur[fc]['t'].lo])
                plt.xlabel('Freq [{}]'.format(rid.freq_unit))
                plt.ylabel('{} after {}'.format(ts_unit, t0))
                plt.colorbar()
            return

        # plot data (other)
        for fc in feature_components:
            if keys is not None:
                keys = used_keys[fc]
            if not all_same_plot:
                plt.figure(fc)
            time_space = []
            for fs in used_keys[fc]:
                tkey = rid.get_datetime_from_timestamp(spectrum_peak._get_timestr_from_ftr_key(fs))
                time_space.append(sp_utils.get_duration_in_std_units((tkey - t0).total_seconds(), use_unit=ts_unit)[0])
            freq_space = np.linspace(dur[fc]['f'].lo, dur[fc]['f'].hi, len(wf[fc][0]))
            if plot_type == 'stream':
                for i, f in enumerate(freq_space):
                    freq_label = "{:.3f} {}".format(f, rid.freq_unit)
                    if all_same_plot:
                        freq_label = fc + ': ' + freq_label
                    plt.plot(time_space, wf[fc][:, i], label=freq_label)
                print("Number of plots: {}".format(len(freq_space)))
                plt.xlabel('{} after {}'.format(ts_unit, t0))
                plt.ylabel('Power [{}]'.format(rid.val_unit))
            elif plot_type == 'stack':
                for i, ts in enumerate(time_space):
                    if keys is None:
                        time_label = "{:.4f} {}".format(ts, ts_unit)
                    else:
                        time_label = keys[i]
                    if all_same_plot:
                        time_label = fc + ': ' + time_label
                    plt.plot(freq_space, wf[fc][i, :], label=time_label)
                print("Number of plots: {}".format(len(time_space)))
                plt.xlabel('Freq [{}]'.format(rid.freq_unit))
                plt.ylabel('Power [{}]'.format(rid.val_unit))
            if legend:
                plt.legend()
