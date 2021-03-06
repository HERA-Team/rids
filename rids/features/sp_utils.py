# _*_ coding: utf-8 _*_
# Copy 2018 the HERA Project
# Licensed under the 2-clause BSD license

"""
Utility functions for SpectrumPeak
"""

from __future__ import print_function, absolute_import, division


def parse_feature_key(ftr_key):
    data = ftr_key.split(':')
    des = data[0].lower()
    ts = data[1]
    pol = data[2].lower()
    return des, ts, pol


def get_duration_in_std_units(duration, use_unit=None):
    """
    Returns a duration in units based on either:
        ad hoc defauls
        specified units

    Parameters:
    ------------
    duration:  duration in seconds
    use_unit:  output unit - if None return default
    """
    if use_unit is None:
        ts_unit = 'Sec'
        if duration > 400000.0:
            duration /= 86400.0
            ts_unit = 'Day'
        elif duration > 10000.0:
            duration /= 3600
            ts_unit = 'Hr'
        elif duration > 300.0:
            duration /= 60.0
            ts_unit = 'Min'
        return duration, ts_unit
    unit_div = {'Sec': 1.0, 'Min': 60.0, 'Hr': 3600.0, 'Day': 86400}
    if use_unit in unit_div.keys():
        return duration / unit_div[use_unit], use_unit
    return None, None
