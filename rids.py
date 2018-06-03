from __future__ import print_function
import json
import os
import numpy as np
import gzip
import rids_utils as utils
import peaks


class Spectral:
    def __init__(self, polarization='', comment=''):
        self.comment = comment
        self.polarization = polarization
        self.freq = []
        self.val = []


class Rids:
    """
    RF Interference Data System (RIDS)
    Reads/writes .rids/[.ridz] files, [zipped] JSON files with fields as described below.
    Any field may be omitted or missing.
      - This first set is header information - typically stored in a .rids file that gets read/rewritten
            instrument:  description of the instrument used
            receiver:  description of receiver used
            channel_width:  RF bandwidth
            channel_width_unit:  unit of bandwidth
            vbw: video bandwidth (typically used for spectrum analyzer)
            vbw_unit: unit of bandwidth
            time_constant: averaging time/maxhold reset time
                           though not ideal, can be a descriptive word or word pair
                           for e.g. ongoing maxhold, etc
            time_constant_unit:  unit of time_constant
            threshold:  value used to threshold peaks
            threshold_unit: unit of threshold
            freq_unit:  unit of frequency used in spectra
            val_unit: unit of value used in spectra
            comment:  general comment; rid_reader appends, doesn't overwrite
      - These are typically set in data-taking session
            time_stamp:  time_stamp for file/baseline data event
            cal:  calibration data by polarization/frequency
            events:  baseline or ave/maxhold spectra
    """
    dattr = ['instrument', 'receiver', 'time_stamp', 'comment', 'freq_unit', 'val_unit']
    uattr = ['channel_width', 'time_constant', 'threshold', 'vbw']
    spectral_fields = ['comment', 'polarization', 'freq', 'val', 'ave', 'maxhold', 'minhold']
    polarizations = ['E', 'N']

    def __init__(self, comment=None):
        self.instrument = None
        self.receiver = None
        self.channel_width = None
        self.channel_width_unit = None
        self.vbw = None
        self.vbw_unit = None
        self.time_constant = None
        self.time_constant_unit = None
        self.threshold = None
        self.threshold_unit = None
        self.freq_unit = None
        self.val_unit = None
        self.comment = comment
        self.time_stamp = None
        self.cal = {}
        for pol in self.polarizations:
            self.cal[pol] = Spectral(polarization=pol)
        self.events = {}
        # --Other variables--
        self.hipk = None

    def rid_reader(self, filename):
        """
        This will read a RID file with a full or subset of structure entities
        """
        file_type = filename.split('.')[-1].lower()
        if file_type == 'ridz':
            r_open = gzip.open
        else:
            r_open = open
        with r_open(filename, 'rb') as f:
            data = json.load(f)
        for d in self.dattr:
            if d in data:
                if d == 'comment':
                    self.append_comment(data[d])
                else:
                    setattr(self, d, data[d])
        for d in self.uattr:
            if d in data:
                self._set_uattr(d, data[d])
        if 'cal' in data:
            for pol in self.polarizations:
                if pol in data['cal']:
                    for v in self.spectral_fields:
                        if v in data['cal'][pol]:
                            setattr(self.cal[pol], v, data['cal'][pol][v])
        if 'events' in data:
            for d in data['events']:
                self.events[d] = Spectral()
                for v in self.spectral_fields:
                    if v in data['events'][d]:
                        setattr(self.events[d], v, data['events'][d][v])

    def _set_uattr(self, d, x):
        v = x.split()
        if len(v) == 1:
            d0 = v[0]
            d1 = None
        elif len(v) > 1:
            try:
                d0 = float(v[0])
            except ValueError:
                d0 = v[0]
            d1 = v[1]
        setattr(self, d, d0)
        setattr(self, d + '_unit', d1)

    def rid_writer(self, filename, fix_list=True):
        """
        This writes a RID file with a full structure
        """
        ds = {}
        for d in self.dattr:
            ds[d] = getattr(self, d)
        for d in self.uattr:
            ds[d] = "{} {}".format(getattr(self, d), getattr(self, d + '_unit'))
        ds['cal'] = {'E': {}, 'N': {}}
        for pol in self.polarizations:
            for v in self.spectral_fields:
                try:
                    ds['cal'][pol][v] = getattr(self.cal[pol], v)
                except AttributeError:
                    continue
        ds['events'] = {}
        for d in self.events:
            ds['events'][d] = {}
            for v in self.spectral_fields:
                try:
                    ds['events'][d][v] = getattr(self.events[d], v)
                except AttributeError:
                    continue
        jsd = json.dumps(ds, sort_keys=True, indent=4, separators=(',', ':'))
        if fix_list:
            jsd = utils.fix_json_list(jsd)
        file_type = filename.split('.')[-1].lower()
        if file_type == 'ridz':
            r_open = gzip.open
        else:
            r_open = open
        with r_open(filename, 'wb') as f:
            f.write(jsd)

    def set(self, **kwargs):
        for k in kwargs:
            if k in self.dattr:
                setattr(self, k, kwargs[k])
            elif k in self.uattr:
                self._set_uattr(k, kwargs[k])

    def append_comment(self, comment):
        if comment is None:
            return
        if self.comment is None:
            self.comment = comment
        else:
            self.comment += ('\n' + comment)

    def get_event(self, event, polarization, ave_fn=None, maxh_fn=None, minh_fn=None):
        if ave_fn is not None:
            ave = Spectral()
            utils.spectrum_reader(ave_fn, ave, polarization)
        if maxh_fn is not None:
            maxhold = Spectral()
            utils.spectrum_reader(maxh_fn, maxhold, polarization)
        if minh_fn is not None:
            minhold = Spectral()
            utils.spectrum_reader(minh_fn, minhold, polarization)
        self.events[event] = Spectral(polarization=polarization)
        if 'baseline' in event.lower():
            if ave_fn:
                self.events[event].freq = ave.freq
                self.events[event].ave = ave.val
            if minh_fn:
                if len(minhold.freq) > len(self.events[event].freq):
                    self.events[event].freq = minhold.freq
                self.events[event].minhold = minhold.val
            if maxh_fn:
                if len(maxhold.freq) > len(self.events[event].freq):
                    self.events[event].freq = maxhold.freq
                self.events[event].maxhold = maxhold.val
        else:
            if maxh_fn:
                self.peak_finder(maxhold)
            elif minh_fn:
                self.peak_finder(minhold)
            elif ave_fn:
                self.peak_finder(ave)
            else:
                return
            self.events[event].freq = list(np.array(self.hipk_freq)[self.hipk])
            if ave_fn:
                try:
                    self.events[event].ave = list(np.array(ave.val)[self.hipk])
                except IndexError:
                    pass
            if maxh_fn:
                try:
                    self.events[event].maxhold = list(np.array(maxhold.val)[self.hipk])
                except IndexError:
                    pass
            if minh_fn:
                try:
                    self.events[event].minhold = list(np.array(minhold.val)[self.hipk])
                except IndexError:
                    pass

    def peak_finder(self, spec, cwt_range=[1, 7], rc_range=[4, 4]):
        self.hipk_freq = spec.freq
        self.hipk_val = spec.val
        self.hipk = peaks.fp(spec.val, self.threshold, cwt_range, rc_range)

    def peak_viewer(self):
        if self.hipk is None:
            return
        import matplotlib.pyplot as plt
        plt.plot(self.hipk_freq, self.hipk_val)
        plt.plot(np.array(self.hipk_freq)[self.hipk], np.array(self.hipk_val)[self.hipk], 'kv')

    def viewer(self):
        clr = ['k', 'b', 'g', 'r', 'm', 'c', 'y', '0.25', '0.5', '0.75']
        c = 0
        for e, v in self.events.iteritems():
            s = clr[c % len(clr)]
            # Plot ave if available
            if 'baseline' in e.lower():
                s = 'k'
            try:
                utils.spectrum_plotter(e, v.freq, v.ave, '_', s)
            except AttributeError:
                pass
            # Plot maxhold if available
            if 'baseline' in e.lower():
                s = 'r'
            try:
                utils.spectrum_plotter(e, v.freq, v.maxhold, 'v', s)
            except AttributeError:
                pass
            # Plot minhold if available
            if 'baseline' in e.lower():
                s = 'b'
            try:
                utils.spectrum_plotter(e, v.freq, v.minhold, '^', s)
            except AttributeError:
                pass

            if 'baseline' not in e.lower():
                c += 1

    def stats(self):
        print("Provide standard set of occupancy etc stats")

    def process_files(self, directory, obs_per_file=100, max_loops=1000):
        loop = True
        max_loop_ctr = 0
        while (loop):
            max_loop_ctr += 1
            if max_loop_ctr > max_loops:
                break
            available_files = sorted(os.listdir(directory))
            f = {'ave': {'E': [], 'N': [], 'cnt': {'E': 0, 'N': 0}},
                 'maxh': {'E': [], 'N': [], 'cnt': {'E': 0, 'N': 0}},
                 'minh': {'E': [], 'N': [], 'cnt': {'E': 0, 'N': 0}}}
            loop = False
            for af in available_files:
                ftype, pol = utils.peel_type_polarization(af)
                if ftype in f:
                    loop = True
                    f[ftype][pol].append(os.path.join(directory, af))
                    f[ftype]['cnt'][pol] += 1
            max_pol_cnt = {'E': 0, 'N': 0}
            for pol in self.polarizations:
                for ft in ['ave', 'maxh', 'minh']:
                    if f[ft]['cnt'][pol] > max_pol_cnt[pol]:
                        max_pol_cnt[pol] = f[ft]['cnt'][pol]
                for ft in ['ave', 'maxh', 'minh']:
                    diff_len = max_pol_cnt[pol] - len(f[ft][pol])
                    if diff_len > 0:
                        f[ft][pol] = f[ft][pol] + [None] * diff_len
            for pol in self.polarizations:
                if not max_pol_cnt[pol]:
                    continue
                axn0 = {'a': f['ave'][pol][0], 'x': f['maxh'][pol][0], 'n': f['minh'][pol][0]}
                for h in axn0:
                    ts = utils.peel_time_stamp(axh0[h])
                    if ts is not None:
                        self.set(time_stamp=ts)
                        break
                self.get_event('baseline_' + pol, pol, axn0['a'], axn0['x'], axn0['n'])
                for a, m, n in zip(f['ave'][pol][:obs_per_file],
                                   f['maxh'][pol][:obs_per_file],
                                   f['minh'][pol][:obs_per_file]):
                    time_stamp = utils.peel_time_stamp(a) + pol
                    self.get_event(time_stamp, pol, a, m, n)
                    os.remove(a)
                    os.remove(m)
                    os.remove(n)
            output_file = os.path.join(directory, str(self.time_stamp) + '.ridz')
            self.rid_writer(output_file)
