# -*- coding: utf-8 -*-
# Copy 2018 the HERA Project
# Licensed under the 2-clause BSD license

"""Base module:  RIDS
This module defines the base format.
"""

from __future__ import print_function, absolute_import, division
import json
import os
import six
import gzip
import copy
import numpy as np


class Rids(object):
    """
    RF Interference Data System (RIDS)
    Reads/writes .ridm/.ridz files, JSON files with fields as described below.  This is the building
    block and should read any rids file.  If feature_module is None it ignores features.
    Timestamps should be sortable to increasing time (can fix this later if desired...).

    Any field may be omitted or missing.
      - This first set is metadata - typically stored in a .ridm file that gets read
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
            rid_file:  records what it thinks the ridz filename should be or where the metadata came from
            nsets:  number of feature_sets included in file
            timestamp_first:  timestamp for first feature_set (currently assumes timestamps sort)
            timestamp_last:           "     last          "                 "
            feature_sets:  features etc defined in the feature module
    """
    # Along with the feature attributes, these are the allowed attributes for json r/w
    direct_attributes = ['rid_file', 'ident', 'instrument', 'receiver', 'comment',
                         'timestamp_first', 'timestamp_last', 'time_format', 'nsets']
    unit_attributes = ['channel_width', 'time_constant']

    def __init__(self, comment=None, **diagnose):
        for d in self.direct_attributes:
            setattr(self, d, None)
        for d in self.unit_attributes:
            setattr(self, d, None)
            setattr(self, d + '_unit', None)
        # Re-initialize non-None and add other
        self.comment = comment
        self.nsets = 0
        self.feature_sets = None
        # --Other variables--
        for a, b in six.iteritems(diagnose):
            setattr(self, a, b)

    def reset(self):
        self.__init__()

    def get_datetime_from_timestamp(self, ts):
        if self.time_format.lower() == 'julian':
            from astropy.time import Time
            if isinstance(ts, (str, unicode)):
                ts = float(ts)
            dt = Time(ts, format='jd', scale='utc')
            return dt.datetime
        elif '%' in self.time_format:
            import datetime
            try:
                return datetime.datetime.strptime(ts, str(self.time_format))
            except ValueError:
                print("{} is invalid format for {}".format(self.time_format, ts))
        return None

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
            self.comment += ('\n' + comment)
        self.comment = self.comment.strip()

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
        self._reader(filename, reset=reset)

    def _reader(self, filename, feature_direct=[], feature_unit=[]):
        self.rid_file = filename
        file_type = filename.split('.')[-1].lower()
        if file_type == 'ridz':
            r_open = gzip.open
        else:
            r_open = open
        with r_open(filename, 'rb') as f:
            data = json.load(f)
        for d, val in six.iteritems(data):
            if d == 'comment':
                self.append_comment(val)
            elif d in self.direct_attributes:
                setattr(self, d, val)
            elif d in feature_direct:
                setattr(self, d, val)
            elif d in self.unit_attributes:
                set_unit_values(self, d, val)
            elif d in feature_unit:
                set_unit_values(self, d, val)
            elif d == 'feature_sets':
                if self.feature_sets is None:
                    continue
                for k, fs in six.iteritems(val):
                    self.feature_sets[k] = self.read_feature_set_dict(fs)

    def writer(self, filename, fix_list=True):
        self._writer(filename, fix_list=fix_list)

    def _writer(self, filename, feature_direct=[], feature_unit=[], fix_list=True):
        """
        This writes a RID file with a full structure.  If a field is None it ignores.
        """
        ds = {}
        for d in self.direct_attributes:
            val = getattr(self, d)
            if val is not None:
                ds[d] = val
        for d in feature_direct:
            val = getattr(self, d)
            if val is not None:
                ds[d] = val
        for d in self.unit_attributes:
            val = getattr(self, d)
            if val is not None:
                ds[d] = "{} {}".format(val, getattr(self, d + '_unit'))
        for d in feature_unit:
            val = getattr(self, d)
            if val is not None:
                ds[d] = "{} {}".format(val, getattr(self, d + '_unit'))
        ds['feature_sets'] = {}
        for d in self.feature_sets:
            ds['feature_sets'][d] = {}
            for v in self.feature_components:
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
            f.write(jsd.encode('utf-8'))

    def info(self):
        self._info()

    def _info(self, feature_direct=[], feature_unit=[], dirlen=50):
        print("RIDS Information")
        for d in self.direct_attributes:
            val = getattr(self, d)
            if len(str(val)) > dirlen:
                val = str(val)[:dirlen] + ' ......'
            print("\t{}:  {}".format(d, val))
        for d in self.unit_attributes:
            print("\t{}:  {} {}".format(d, getattr(self, d), getattr(self, d + '_unit')))
        for d in feature_direct:
            val = getattr(self, d)
            if len(str(val)) > dirlen:
                val = str(val)[:dirlen] + ' ......'
            print("\tfeatures.{}:  {}".format(d, val))
        for d in feature_unit:
            print("\tfeatures.{}:  {} {}".format(d, getattr(self, d), getattr(self, d + '_unit')))
        if self.feature_sets is not None and self.nsets != len(self.feature_sets):
            print("Note that the stated nsets={} does not match the found nsets={}".format(self.nsets, len(self.feature_sets)))


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
