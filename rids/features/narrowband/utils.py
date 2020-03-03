# -*- mode: python; coding: utf-8 -*-
# Copyright 2018 the HERA Collaboration
# Licensed under the 2-clause BSD license.

# update imports as needed; XXX remove this comment before merging to master
import numpy as np
from astropy import constants, units

import itertools
import uvtools
import pyuvdata
from pyuvdata import UVData

def reduce_data(uvd, use_ants=None, use_bls=None, use_pols='linear',
                use_autos=True, use_cross=False, use_times=None, 
                use_freqs=None, use_lsts=None, **kwargs):
    """Select only a subset of the data in ``uvd``.

    Parameters
    ----------
    uvd : UVData or list of UVData
        UVData object, or path to a file that may be read by a UVData object, 
        on which to perform the data reduction. Optionally pass a list.

    """
    pass
