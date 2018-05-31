from __future__ import print_function
import json


class Spectral:
    def __init__(self):
        self.comment = ''
        self.freq = []


class Rids:
    dattr = ['instrument', 'detector', 'polarization', 'comment', 'time_stamp', 'freq_unit', 'val_unit']
    uattr = ['channel_width', 'time_constant', 'threshold']
    sattr = {}

    def __init__(self):
        self.instrument = None
        self.detector = None
        self.channel_width = None
        self.channel_width_unit = None
        self.time_constant = None
        self.time_constant_unit = None
        self.time_stamp = None
        self.threshold = None
        self.threshold_unit = None
        self.polarization = None
        self.freq_unit = None
        self.val_unit = None
        self.comment = None
        self.cal = Spectral()
        self.baseline_spectrum = Spectral()
        self.events = {}
        self.sattr = {'cal': self.cal 'baseline_spectrum': self.baseline_spectrum}

    def file_reader(self, fn):
        with open(fn, 'rb') as f:
            data = json.load(f)
        for d in self.dattr:
            if d in data:
                setattr(self, d, data[d])
        for d in self.uattr:
            if d in data:
                v = data[d].split()
                setattr(self, d, v[0])
                setattr(self, d + '_unit', v[1])
        for d in self.sattr:
            if d in data:
                for v in ['comment', 'freq', 'val']:
                    if v in data[d]:
                        setattr(self.sattr[d], v, data[d][v])
        if 'events' in data:
            for e in data['events']:
                self.events[e] = Spectral()
                for v in ['comment', 'freq', 'ave', 'maxhold']:
                    if v in data['events'][e]:
                        setattr(self.events[e], v, data['events'][e][v])

    def spectrum_reader(self, fn, spec):
        with open(fn, 'r') as f:
            for line in f:
                data = [float(x) for x in line.split()]
                spec.freq.append(data[0])
                if len(data) == 2:
                    spec.val.append(data[1])
                else:
                    spec.ave.append(data[1])
                    spec.maxhold.append(data[2])

    def apply_cal(self):
        print("CAL")

    def display(self):
        print("DISP")
