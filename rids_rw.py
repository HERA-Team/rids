from __future__ import print_function, absolute_import, division
import json
import os
import gzip
import copy
import numpy as np


class Spectral:
    spectral_fields = ['comment', 'polarization', 'freq', 'val', 'maxhold', 'minhold', 'bw']

    def __init__(self, polarization='', comment=''):
        self.comment = comment
        self.polarization = polarization
        self.freq = []
        self.val = []


class Temporal:
    temporal_fields = ['comment', 'polarization', 'time', 'val']

    def __init__(self, polarization='', comment=''):
        self.comment = comment
        self.polarization = polarization
        self.time = []
        self.val = []


class RidsReadWrite:
    """
    RF Interference Data System (RIDS)
    Reads/writes .rids/[.ridz] files, [zipped] JSON files with fields as described below
        and in feature module
    Any field may be omitted or missing.
      - This first set is header information - typically stored in a .rids file that gets read/rewritten
            ident: description of filename
            instrument:  description of the instrument used
            receiver:  description of receiver used
            channel_width:  RF bandwidth (width in file or FFT)
            channel_width_unit:  unit of bandwidth
            nsets:  number of feature_sets included in file
            time_constant: averaging time/maxhold reset time
                           though not ideal, can be a descriptive word or word pair
                           for e.g. ongoing maxhold, etc
            time_constant_unit:  unit of time_constant
            freq_unit:  unit of frequency used in spectra
            val_unit: unit of value used in spectra
            comment:  general comment; reader appends, doesn't overwrite
      - These are typically set in data-taking session
            time_stamp_first:  time_stamp for first feature_set
            time_stamp_last:           "      last          "
            feature_sets:  features etc defined in the feature module
    """
    # Along with the feature attributes, these are the allowed attributes for json r/w
    direct_attributes = ['rid_file', 'ident', 'instrument', 'receiver', 'comment',
                         'time_stamp_first', 'time_stamp_last', 'time_format',
                         'freq_unit', 'val_unit', 'nsets']
    unit_attributes = ['channel_width', 'time_constant']
    polarizations = ['E', 'N', 'I']

    def __init__(self, feature_module=None, comment=None, **diagnose):
        for d in self.direct_attributes:
            setattr(self, d, None)
        for d in self.unit_attributes:
            setattr(self, d, None)
            setattr(self, d + '_unit', None)
        # Re-initialize non-None and add other
        self.comment = comment
        self.nsets = 0
        # Add in features
        self.feature_sets = {}
        self.feature_module = feature_module  # used for reset
        if feature_module is None:
            from argparse import Namespace
            self.features = Namespace()
            self.features.direct_attributes = []
            self.features.unit_attributes = []
        else:
            self.features = feature_module
            for d in self.features.direct_attributes:
                setattr(self, d, None)
            for d in self.features.unit_attributes:
                setattr(self, d, None)
                setattr(self, d + '_unit', None)
        # --Other variables--
        for a, b in diagnose.iteritems():
            setattr(self, a, b)

    def reset(self):
        self.__init__(self.feature_module, None)

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
            elif d in self.features.direct_attributes:
                setattr(self.features, d, X)
            elif d in self.unit_attributes:
                set_unit_values(self, d, X)
            elif d in self.features.unit_attributes:
                set_unit_values(self.features, d, X)
            elif d == 'feature_sets':
                print("This goes into the feature module somehow")
                # for e in X:
                #     self.feature_sets[e] = Spectral()
                #     for v, Y in X[e].iteritems():
                #         if v not in self.spectral_fields:
                #             print("Unexpected field {} in {}".format(v, d))
                #             continue
                #         setattr(self.events[e], v, Y)

    def writer(self, filename, fix_list=True):
        """
        This writes a RID file with a full structure
        """
        ds = {}
        for d in self.direct_attributes:
            ds[d] = getattr(self, d)
        for d in self.features.direct_attributes:
            ds[d] = getattr(self.features, d)
        for d in self.unit_attributes:
            ds[d] = "{} {}".format(getattr(self, d), getattr(self, d + '_unit'))
        for d in self.features.unit_attributes:
            ds[d] = "{} {}".format(getattr(self.features, d),
                                   getattr(self.features, d + '_unit'))
        ds['feature_sets'] = {}
        print("In feature modules also")
        # for d in self.feature_sets:
        #     ds['events'][d] = {}
        #     for v in self.spectral_fields:
        #         try:
        #             ds['events'][d][v] = getattr(self.events[d], v)
        #         except AttributeError:
        #             continue
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
            print("RRW183:  ", comment)
            print("         ", self.comment)
            self.comment += ('\n' + comment)

    def info(self):
        print("RIDS Information")
        for d in self.direct_attributes:
            print("\t{}:  {}".format(d, getattr(self, d)))
        for d in self.unit_attributes:
            print("\t{}:  {} {}".format(d, getattr(self, d), getattr(self, d + '_unit')))
        for d in self.features.direct_attributes:
            print("\tfeatures.{}:  {}".format(d, getattr(self.features, d)))
        for d in self.features.unit_attributes:
            print("\tfeatures.{}:  {} {}".format(d, getattr(self.features, d), getattr(self.features, d + '_unit')))
        print("\t{} feature_sets".format(len(self.feature_sets)))


def set_unit_values(C, d, x):
    v = x.split()
    try:
        d0 = float(v[0])
    except ValueError:
        d0 = v[0]
    d1 = None
    if len(v) > 1:
        d1 = v[1]
    setattr(C, d, d0)
    setattr(C, d + '_unit', d1)


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
