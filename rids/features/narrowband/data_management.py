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
                use_freqs=None):
    """Select only a subset of the data in ``uvd``.

    XXX refer to numpydoc style (does UVData etc receive ``...``?)
    Parameters
    ----------
    uvd : UVData or list of UVData
        UVData object, or path to a file that may be read by a UVData object, 
        on which to perform the data reduction. A list of UVData objects or 
        strings may also be passed, but the list must be of uniform type. If 
        a list is passed, then *all* UVData objects are loaded in a single 
        UVData object.

    use_ants : array-like of int, optional
        List of antenna numbers whose data should be kept. Default is to 
        keep data for all antennas.

    use_bls : array-like of 2- or 3-tuples, optional
        List of antenna pairs or baseline tuples specifying which baselines 
        (and possibly polarizations) to keep. Default is to keep all baselines.

    use_pols : str or array-like, optional
        If passing a string, then it must be one of the following: 'linear', 
        'cross', or 'all' (these refer to which visibility polarizations to 
        keep). If passing an array-like object, then the entries must either 
        be polarization strings or polarization integers. Polarization strings 
        are automatically converted to polarization integers. Default is to 
        use only the linear polarizations ('xx' and 'yy' or 'ee' and 'nn').

    use_autos : bool, optional
        Whether to keep the autocorrelations. Default is to keep the autos.

    use_cross : bool, optional
        Whether to keep the cross-correlations. Default is to discard the 
        cross-correlations.

    use_times : array-like, optional
        Times to keep. If length-2, then it is interpreted as a range of 
        time values and must be specified in Julian Date. Otherwise, the 
        times must exist in the ``time_array`` attribute of the ``UVData`` 
        object corresponding to ``uvd``. Default is to use all times.

    use_freqs : array-like, optional
        Frequencies or frequency channels to keep. If each entry is an 
        integer, then it is interpreted as frequency channels. Default 
        is to use all frequencies.

    Returns
    -------
    uvd : UVData
        UVData object downselected according to the parameters chosen.
    """
    # handle different types for ``uvd``
    if isinstance(uvd, str):
        uvd = uvd.read(uvd)
    elif isinstance(uvd, UVData):
        pass
    elif isinstance(uvd, (list, tuple)):
        if all([isinstance(uvd_, str) for uvd_ in uvd]):
            uvd = uvd.read(uvd)
        elif all([isinstance(uvd_, UVData) for uvd_ in uvd]):
            _uvd = uvd[0]
            for uvd_ in uvd[1:]:
                _uvd += uvd_
            uvd = _uvd
        else:
            raise ValueError(
                "If you pass a list or tuple for ``uvd``, then every entry "
                "in the list must be of the same type (str or UVData)."
            )
    else:
        raise ValueError(
            "``uvd`` must be either a string, UVData object, or list/tuple "
            "of strings or UVData objects (no mixing of types allowed)."
        )

    # first downselect: polarization
    # XXX can make a helper function for this
    pol_strings = uvd.get_pols()
    pol_array = uvd.polarization_array
    if use_pols == 'linear':
        use_pols = [
            pyuvdata.utils.polstr2num(pol) for pol in pol_strings
            if pol[0] == pol[1]
        ]
    elif use_pols == 'cross':
        use_pols = [
            pyuvdata.utils.polstr2num(pol) for pol in pol_strings
            if pol[0] != pol[1]
        ]
    elif use_pols == 'all':
        use_pols = pol_array
    else:
        try:
            _ = iter(use_pols)
        except TypeError:
            raise ValueError(
                "``use_pols`` must be one of the following:\n"
                "'linear' : use only linear polarizations\n"
                "'cross' : use only cross polarizations\n"
                "'all' : use all polarizations\n"
                "iterable of polarization numbers"
            )
        use_pols = [
            pyuvdata.utils.polstr2num(pol) if isinstance(pol, str)
            else pol for pol in use_pols
        ]
    # actually downselect along polarization
    uvd.select(polarizations=use_pols, keep_all_metadata=False)

    # next downselect: visibility type
    if use_autos and not use_cross:
        ant_str = 'auto'
    elif not use_autos and use_cross:
        ant_str = 'cross'
    else:
        ant_str = 'all'
    uvd.select(ant_str=ant_str, keep_all_metadata=False)

    # next downselect: antennas
    if use_ants is not None:
        uvd.select(antenna_nums=use_ants, keep_all_metadata=False)

    # next downselect: baselines
    if use_bls is not None:
        uvd.select(bls=use_bls, keep_all_metadata=False)

    # next downselect: frequency
    if use_freqs is not None:
        if all([isinstance(freq, int) for freq in use_freqs]):
            uvd.select(freq_chans=use_freqs, keep_all_metadata=False)
        else:
            uvd.select(frequencies=use_freqs, keep_all_metadata=False)

    # next downselect: time
    if use_times is not None:
        if len(use_times) == 2:
            uvd.select(time_range=use_times, keep_all_metadata=False)
        else:
            uvd.select(times=use_times, keep_all_metadata=False)

    # all the downselecting should be done at this point
    return uvd
