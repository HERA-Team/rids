from __future__ import print_function, absolute_import, division
import json
import os
import gzip
import copy
import numpy as np


class RidsReadWrite:
    """
    RF Interference Data System (RIDS)
    Reads/writes .ridm/.ridz files, JSON files with fields as described below
        and in feature module
    Any field may be omitted or missing.
      - This first set is metadata - typically stored in a .ridm file that gets read/rewritten
            ident: description of filename
            instrument:  description of the instrument used
            receiver:  description of receiver used
            channel_width:  RF bandwidth (width in file or FFT)
            channel_width_unit:  unit of bandwidth
            time_constant: averaging time/maxhold reset time
                           though not ideal, can be a descriptive word or word pair
                           for e.g. ongoing maxhold, etc
            time_constant_unit:  unit of time_constant
            freq_unit:  unit of frequency used in spectra
            val_unit: unit of value used in spectra
            comment:  general comment; reader appends, doesn't overwrite
            time_format:  string indicating the format of timestamp in filename
      - These are typically set in data-taking session
            rid_file:  records what it thinks the ridz filename should be
            nsets:  number of feature_sets included in file
            time_stamp_first:  time_stamp for first feature_set
            time_stamp_last:           "      last          "
            feature_module_name:  name of the feature module used
            feature_sets:  features etc defined in the feature module
    """
    # Along with the feature attributes, these are the allowed attributes for json r/w
    direct_attributes = ['rid_file', 'ident', 'instrument', 'receiver', 'comment',
                         'time_stamp_first', 'time_stamp_last', 'time_format',
                         'freq_unit', 'val_unit', 'nsets', 'feature_module_name']
    unit_attributes = ['channel_width', 'time_constant']

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
        self.feature_module_for_reset = feature_module
        if feature_module is None:
            from argparse import Namespace
            self.features = Namespace()
            self.features.direct_attributes = []
            self.features.unit_attributes = []
            self.feature_module_name = 'None'
        else:
            self.features = feature_module
            self.feature_module_name = self.features.feature_module_name
            for d in self.features.direct_attributes:
                setattr(self, d, None)
            for d in self.features.unit_attributes:
                setattr(self, d, None)
                setattr(self, d + '_unit', None)
        # --Other variables--
        for a, b in diagnose.iteritems():
            setattr(self, a, b)

    def reset(self):
        self.__init__(self.feature_module_for_reset)

    def reader(self, filename, reset=True):
        """
        This will read a RID file with a full or subset of structure entities.

        Parameters:
        ------------
        filename:  rids/m/z filename to read
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
        self.nsets = 0
        for d, val in data.iteritems():
            if d == 'comment':
                self.append_comment(val)
            elif d in self.direct_attributes:
                setattr(self, d, val)
            elif d in self.features.direct_attributes:
                setattr(self.features, d, val)
            elif d in self.unit_attributes:
                set_unit_values(self, d, val)
            elif d in self.features.unit_attributes:
                set_unit_values(self.features, d, val)
            elif d == 'feature_sets' or d == 'events':
                for k, fs in val.iteritems():
                    self.feature_sets[k] = self.features.read_feature_set_dict(fs)
                    self.nsets += 1

    def writer(self, filename, fix_list=True):
        """
        This writes a RID file with a full structure.  If a field is None it ignores.
        """
        ds = {}
        for d in self.direct_attributes:
            val = getattr(self, d)
            if val is not None:
                ds[d] = val
        for d in self.features.direct_attributes:
            val = getattr(self.features, d)
            if val is not None:
                ds[d] = val
        for d in self.unit_attributes:
            val = getattr(self, d)
            if val is not None:
                ds[d] = "{} {}".format(val, getattr(self, d + '_unit'))
        for d in self.features.unit_attributes:
            val = getattr(self.features, d)
            if val is not None:
                ds[d] = "{} {}".format(val, getattr(self.features, d + '_unit'))
        ds['feature_sets'] = {}
        for d in self.feature_sets:
            ds['feature_sets'][d] = {}
            for v in self.features.feature_components:
                try:
                    ds['feature_sets'][d][v] = getattr(self.feature_sets[d], v)
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
                self.set_unit_values(self, k, kwargs[k])

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