from __future__ import print_function, absolute_import, division
import json
import os
import gzip
import copy
import numpy as np


class Spectral:
    def __init__(self, polarization='', comment=''):
        self.comment = comment
        self.polarization = polarization
        self.freq = []
        self.val = []


class RidsReadWrite:
    """
    RF Interference Data System (RIDS)
    Reads/writes .rids/[.ridz] files, [zipped] JSON files with fields as described below.
    Any field may be omitted or missing.
      - This first set is header information - typically stored in a .rids file that gets read/rewritten
            ident: description of filename
            instrument:  description of the instrument used
            receiver:  description of receiver used
            channel_width:  RF bandwidth (width in file or FFT)
            channel_width_unit:  unit of bandwidth
            vbw: video bandwidth (typically used for spectrum analyzer)
            vbw_unit: unit of bandwidth
            rbw:  resolution bandwidth (typically used for spectrum analyzer)
            rbw_unit:  unit of bandwidth
            nevents:  number of events included in file
            time_constant: averaging time/maxhold reset time
                           though not ideal, can be a descriptive word or word pair
                           for e.g. ongoing maxhold, etc
            time_constant_unit:  unit of time_constant
            freq_unit:  unit of frequency used in spectra
            val_unit: unit of value used in spectra
            comment:  general comment; reader appends, doesn't overwrite
      - These are typically set in data-taking session
            time_stamp_first:  time_stamp for first file/baseline data event
            time_stamp_last:           "      last          "
            cal:  calibration data by polarization
            events:  baseline or ave/maxhold spectra
    """
    direct_attributes = ['rid_file', 'ident', 'instrument', 'receiver', 'comment',
                         'time_stamp_first', 'time_stamp_last', 'time_format',
                         'freq_unit', 'val_unit', 'nevents', 'filename']
    unit_attributes = ['channel_width', 'time_constant']
    spectral_fields = ['comment', 'polarization', 'freq', 'val', 'maxhold', 'minhold', 'bw']
    polarizations = ['E', 'N', 'I']

    def __init__(self, comment=None, feature_module, **diagnose):
        for d in direct_attributes:
            setattr(self, d, None)
        for d in unit_attributes:
            setattr(self, d, None)
            setattr(self, d + 'unit', None)
        # re-initialize non-None
        self.comment = comment
        self.nevents = 0
        self.cal = {}
        self.events = {}
        self.feature_module = feature_module
        self.feature_settings = feature_module.Feature()
        for d in self.feature_settings.direct_attributes:
            setattr(self, d, None)
        for d in self.feature_settings.unit_attributes:
            setattr(self, d, None)
            setattr(self, d + 'unit', None)
        # --Other variables--
        self.hipk = None
        self.hipk_bw = None
        for a, b in diagnose.iteritems():
            setattr(self, a, b)

    def reset(self):
        self.__init__(None, self.feature_module)

    def reader(self, filename, reset=True):
        """
        This will read a RID file with a full or subset of structure entities.

        Parameters:
        ------------
        filename:  rids/z filename to read
        reset:  If true, resets all elements.
                If false, will overwrite headers etc but add events
                    (i.e. things with a unique key won't get overwritten)
        """
        if reset:
            self.reset()
        self.rid_file = filename
        file_type = filename.split('.')[-1].lower()
        if file_type == 'ridz':
            r_open = gzip.open
        else:
            r_open = open
        with r_open(filename, 'rb') as f:
            data = json.load(f)
        for d, X in data.iteritems():
            if d == 'comment':
                self.append_comment(X)
            elif d in self.direct_attributes:
                setattr(self, d, X)
            elif d in self.peak_settings.direct_attributes:
                setattr(self.peak_settings, d, X)
            elif d in self.unit_attributes:
                set_unit_values(self, d, X)
            elif d in self.peak_settings.unit_attributes:
                set_unit_values(self.peak_settings, d, X)
            elif d == 'cal':
                for pol in X:
                    if pol not in self.polarizations:
                        print("Unexpected pol {} in {}".format(pol, d))
                        continue
                    self.cal[pol] = Spectral(pol)
                    for v, Y in X[pol].iteritems():
                        if v not in self.spectral_fields:
                            print("Unexpected field {} in {}".format(v, d))
                            continue
                        setattr(self.cal[pol], v, Y)
            elif d == 'events':
                for e in X:
                    self.events[e] = Spectral()
                    for v, Y in X[e].iteritems():
                        if v not in self.spectral_fields:
                            print("Unexpected field {} in {}".format(v, d))
                            continue
                        setattr(self.events[e], v, Y)

    def writer(self, filename, fix_list=True):
        """
        This writes a RID file with a full structure
        """
        ds = {}
        for d in self.direct_attributes:
            ds[d] = getattr(self, d)
        for d in self.peak_settings.direct_attributes:
            ds[d] = getattr(self.peak_settings, d)
        for d in self.unit_attributes:
            ds[d] = "{} {}".format(getattr(self, d), getattr(self, d + '_unit'))
        for d in self.peak_settings.unit_attributes:
            ds[d] = "{} {}".format(getattr(self.peak_settings, d),
                                   getattr(self.peak_settings, d + '_unit'))
        caltmp = {}
        for pol in self.polarizations:
            for v in self.spectral_fields:
                try:
                    X = getattr(self.cal[pol], v)
                    if pol not in caltmp:
                        caltmp[pol] = {}
                    caltmp[pol][v] = X
                except AttributeError:
                    continue
                except KeyError:
                    continue
        if len(caltmp):
            ds['cal'] = caltmp
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
            jsd = fix_json_list(jsd)
        file_type = filename.split('.')[-1].lower()
        if file_type == 'ridz':
            r_open = gzip.open
        else:
            r_open = open
        with r_open(filename, 'wb') as f:
            f.write(jsd)

    def set(self, **kwargs):
        for k in kwargs:
            if k in self.direct_attributes:
                setattr(self, k, kwargs[k])
            elif k in self.unit_attributes:
                self._set_unit_attributes(k, kwargs[k])

    def append_comment(self, comment):
        if comment is None:
            return
        if self.comment is None:
            self.comment = comment
        else:
            self.comment += ('\n' + comment)


def peel_filename(v, eclist=None):
    if v is None:
        return {}
    s = v.split('/')[-1].split('.')
    if len(s) not in [4, 5]:  # time_stamp may have one '.' in it
        return {}
    fnd = {'ident': s[0]}
    fnd['time_stamp'] = '.'.join(s[1:-2])
    fnd['event_component'] = s[-2].lower()
    fnd['polarization'] = s[-1].upper()
    if eclist is None:
        return fnd
    eclist = [x.lower() for x in eclist]
    for ec in eclist:
        if fnd['event_component'] in ec:
            fnd['event_component'] = ec
            return fnd
    return {}


def fix_json_list(jsd):
    spaces = ['\n', ' ']
    fixed = ''
    in_list = False
    sb_count = 0
    for c in jsd:
        if c == '[':
            in_list = True
            sb_count += 1
        elif c == ']':
            sb_count -= 1
            if not sb_count:
                in_list = False
        if not in_list:
            fixed += c
        elif in_list and c not in spaces:
            fixed += c
    return fixed
