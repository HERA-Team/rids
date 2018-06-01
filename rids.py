from __future__ import print_function
import json
import os
import numpy as np
import rids_utils as utils
import peaks


class Spectral:
    def __init__(self, polarization='', comment=''):
        self.comment = comment
        self.polarization = polarization
        self.freq = []
        self.val = []


class Rids:
    dattr = ['instrument', 'detector', 'comment', 'time_stamp', 'freq_unit', 'val_unit']
    uattr = ['channel_width', 'time_constant', 'threshold', 'vbw']
    spectral_fields = ['comment', 'polarization', 'freq', 'val', 'ave', 'maxhold']
    polarizations = ['E', 'N']

    def __init__(self):
        self.instrument = None
        self.detector = None
        self.channel_width = None
        self.channel_width_unit = None
        self.vbw = None
        self.vbw_unit = None
        self.time_constant = None
        self.time_constant_unit = None
        self.time_stamp = None
        self.threshold = None
        self.threshold_unit = None
        self.freq_unit = None
        self.val_unit = None
        self.comment = None
        self.cal = {}
        for pol in self.polarizations:
            self.cal[pol] = Spectral(polarization=pol)
        self.events = {}
        # --Other variables--
        self.hipk = None

    def json_reader(self, filename):
        """
        This will read a json file with a full or subset of structure entities
        """
        with open(filename, 'rb') as f:
            data = json.load(f)
        for d in self.dattr:
            if d in data:
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

    def json_writer(self, filename, fix_list=True):
        """
        This writes a JSON file with a full RIDS structure
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
        with open(filename, 'w') as f:
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

    def get_event(self, event, ave_fn, maxhold_fn, polarization):
        ave = Spectral()
        utils.spectrum_reader(ave_fn, ave)
        maxhold = Spectral()
        utils.spectrum_reader(maxhold_fn, maxhold)
        self.events[event] = Spectral(polarization=polarization)
        if 'baseline' in event.lower():
            self.events[event].freq = ave.freq if len(ave.freq) > len(maxhold.freq) else maxhold.freq
            self.events[event].ave = ave.val
            self.events[event].maxhold = maxhold.val
        else:
            self.peak_finder(maxhold)
            self.events[event].freq = list(np.array(maxhold.freq)[self.hipk])
            try:
                self.events[event].ave = list(np.array(ave.val)[self.hipk])
            except IndexError:
                print(event, len(maxhold.freq), len(ave.freq))
            self.events[event].maxhold = list(np.array(maxhold.val)[self.hipk])

    def peak_finder(self, spec, cwt_range=[1, 7], rc_range=[4, 4]):
        self.hipk_freq = spec.freq
        self.hipk_val = spec.val
        self.hipk = peaks.fp(spec.val, self.threshold, cwt_range, rc_range)

    def peak_viewer(self):
        if self.hipk is None:
            print("No peaks sought.")
            return
        import matplotlib.pyplot as plt
        plt.plot(self.hipk_freq, self.hipk_val)
        plt.plot(np.array(self.hipk_freq)[self.hipk], np.array(self.hipk_val)[self.hipk], 'kv')

    def viewer(self):
        import matplotlib.pyplot as plt
        for e, v in self.events.iteritems():
            if 'baseline' in e:
                clr = plt.plot(v.freq[:len(v.ave)], v.ave)
                plt.plot(v.freq, v.maxhold, color=clr[0].get_color())
            else:
                clr = plt.plot(v.freq[:len(v.ave)], v.ave[:len(v.freq)], 'v')
                plt.plot(v.freq, v.maxhold, 'v', color=clr[0].get_color())

    def process_files(self, directory, obs_per_file=100, keep_completed=True):
        available_files = sorted(os.listdir(directory))
        ave_files = {'E': [], 'N': []}
        maxhold_files = {'E': [], 'N': []}
        for af in available_files:
            if 'ave' not in af and 'max' not in af:
                continue
            pol = utils.peal_polarization(af)
            if 'ave' in af:
                ave_files[pol].append(af)
            elif 'max' in af:
                maxhold_files[pol].append(af)
        for pol in self.polarizations:
            if not len(ave_files[pol]) or not len(maxhold_files[pol]):
                continue
            time_stamp = utils.peal_time_stamp(ave_files[pol][0])
            self.set(time_stamp=time_stamp)
            self.get_event('baseline_' + pol, ave_files[pol][0], maxhold_files[pol][0], pol)
            for a, m in zip(ave_files[pol][:obs_per_file], maxhold_files[pol][:obs_per_file]):
                time_stamp = utils.peal_time_stamp(a) + pol
                self.get_event(time_stamp, a, m, pol)
                if not keep_completed:
                    os.remove(a)
                    os.remove(m)
        output_file = os.path.join(directory, str(self.time_stamp) + '.json')
        self.json_writer(output_file)
