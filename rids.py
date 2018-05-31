from __future__ import print_function
import json
import utils
import matplotlib.pyplot as plt


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
        self.cal.val = []
        self.baseline_spectrum = Spectral()
        self.baseline_spectrum.val = []
        self.events = {}
        self.sattr = {'cal': self.cal, 'baseline_spectrum': self.baseline_spectrum}
        self.sattr_fields = ['comment', 'freq', 'val']
        self.event_fields = ['comment', 'freq', 'ave', 'maxhold']

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
                v = data[d].split()
                setattr(self, d, v[0])
                setattr(self, d + '_unit', v[1])
        for d in self.sattr:
            if d in data:
                for v in self.sattr_fields:
                    if v in data[d]:
                        setattr(self.sattr[d], v, data[d][v])
        if 'events' in data:
            for d in data['events']:
                self.events[d] = Spectral()
                for v in self.event_fields:
                    if v in data['events'][d]:
                        setattr(self.events[d], v, data['events'][d][v])

    def json_writer(self, filename, fix_list=True):
        """
        This writes a JSON file with a full RIDS structure
        """
        ds = {}
        for d in self.dattr:
            ds[d] = getattr(self, d)
        for d in self.uattr:
            ds[d] = "{} {}".format(getattr(self, d), getattr(self, d + '_unit'))
        for d in self.sattr:
            ds[d] = {}
            for v in self.sattr_fields:
                ds[d][v] = getattr(self.sattr[d], v)
        ds['events'] = {}
        for d in self.events:
            ds['events'][d] = {}
            for v in self.event_fields:
                ds['events'][d][v] = getattr(self.events[d], v)
        jsd = json.dumps(ds, sort_keys=True, indent=4, separators=(',', ':'))
        if fix_list:
            jsd = utils.fix_json_list(jsd)
        with open(filename, 'w') as f:
            f.write(jsd)

    def spectrum_reader(self, filename, spec, time_constant):
        """
        This reads in an ascii spectrum file.
        If two columns stores as freq, val (cal and baseline)
        If three columns stores as freq, ave, maxhold (event)
        """
        with open(filename, 'r') as f:
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
