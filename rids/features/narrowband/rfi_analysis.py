# -*- mode: python; coding: utf-8 -*-
# Copyright 2018 the HERA Collaboration
# Licensed under the 2-clause BSD license.

import copy
import warnings

import numpy as np
from hera_qm import xrfi

def isolate_rfi(uvd, inplace=True, **filter_kwargs):
    """
    Use xrfi.medminfilt to isolate RFI.

    Parameters
    ----------
    uvd : UVData
        UVData object on which to isolate RFI. Assumes that this has 
        already been downselected to keep only the data on which to 
        perform the RFI analysis.

    inplace : bool, optional
        Whether to perform the RFI isolation on ``uvd`` or a copy 
        thereof. Default is to not make a copy.

    **filter_kwargs
        Keyword arguments to pass to xrfi.medminfilt.

    Returns
    -------
    rfi_uvd : UVData
        UVData object whose data array has been adjusted so that any 
        non-RFI signal has been removed (at least approximately).
    """
    # initialize a new data array
    rfi_data = np.zeros_like(uvd.data_array, dtype=uvd.data_array.dtype)

    # get baselines and polarizations
    baselines = np.unique(uvd.baseline_array)
    pols = uvd.get_pols()

    for baseline in baselines:
        for pol in pols:
            antpairpol = uvd.baseline_to_antnums(baseline) + (pol,)
            blt_inds, conj_blt_inds, pol_inds = uvd._key2inds(antpairpol)

            this_data = uvd.get_data(antpairpol)
            filt_data = xrfi.medminfilt(this_data, **filter_kwargs)
            this_rfi = this_data - filt_data

            rfi_data = [blt_inds, 0, :, pol_inds[0]] = this_rfi
            if conj_blt_inds.size != 0:
                rfi_data[conj_blt_inds, 0, :, pol_inds[1]] = this_rfi.conj()

    if not inplace:
        rfi_uvd = copy.deepcopy(uvd)
        rfi_uvd.data_array = rfi_data
        return rfi_uvd

    uvd.data_array = rfi_data
    return uvd

def isolate_rfi_stations(vis_uvd, rfi_uvd=None, detrend='medfilt', 
                         apply_ws=True, detrend_nsig=100, 
                         ws_nsig=20, **detrend_kwargs):
    """
    Extract visibilities for just the narrowband transmitters.

    Parameters
    ----------
    vis_uvd : UVData
        UVData object containing the original visibility data. Assumes 
        downselection has already been applied.

    rfi_uvd : UVData, optional
        UVData object containing the RFI-only visibility data. If not 
        provided, then it is calculated from ``vis_uvd``.

    detrend : str or callable, optional
        String specifying which xrfi detrending method to use. May also 
        be a callable object that takes an array and kwargs and returns 
        an identically-shaped array whose entries give a modified 
        z-score (modified such that it is in units of standard deviations 
        when the input data is Gaussian). Default is to use 
        xrfi.detrend_medfilt.

    apply_ws : bool, optional
        Whether to apply a watershed algorithm to the detrended data. 
        Default is to apply watershed flagging.

    detrend_nsig : float, optional
        Lower bound for flagging data (i.e. any detrended data points 
        with values in excess of ``detrend_nsig`` will be flagged as 
        RFI). Default is 100 standard deviations (most narrowband 
        transmitters are *extremely* bright).

    ws_nsig : float, optional
        Lower bound for expanding RFI flags with the watershed algorithm. 
        Default is 20 standard deviations.

    **detrend_kwargs
        Keyword arguments passed directly to the detrending method used.

    Returns
    -------
    rfi_station_uvd : UVData
        UVData object whose data and flag arrays have been updated to 
        only contain information for narrowband transmitters, as 
        identified by this algorithm.
    """
    # get the detrending method to be used
    if callable(detrend):
        pass
    elif isinstance(detrend, str):
        try:
            detrend = getattr(xrfi, "detrend_%s" % detrend)
        except AttributeError:
            warnings.warn(
                "Detrending method not found. Defaulting to detrend_medfilt."
            )
            detrend = xrfi.detrend_medfilt
    else:
        raise TypeError("Unsupported type for ``detrend``.")

    # make rfi_uvd object if it doesn't exist
    if rfi_uvd is None:
        rfi_uvd = isolate_rfi(vis_uvd, inplace=False)

    # initialize new data and flag arrays
    new_rfi_data = np.zeros_like(
        rfi_uvd.data_array, dtype=rfi_uvd.data_array.dtype
    )
    new_rfi_flags = copy.deepcopy(rfi_uvd.flag_array)

    # prepare things to loop over
    baselines = np.unique(vis_uvd.baseline_array)
    pols = vis_uvd.get_pols()

    # now do things
    for baseline in baselines:
        for pol in pols:
            # get the (ant1, ant2, pol) key and data array indices
            antpairpol = vis_uvd.baseline_to_antnums(baseline) + (pol,)
            blt_inds, conj_blt_inds, pol_inds = vis_uvd._key2inds(antpairpol)

            # actually pull the data
            vis_data = vis_uvd.get_data(antpairpol)
            rfi_data = rfi_uvd.get_data(antpairpol)

            # figure out which pixels to flag
            vis_detrended = detrend(vis_data, **detrend_kwargs)
            these_flags = np.where(
                vis_detrended > detrend_nsig, True, False
            )

            if apply_ws:
                these_flags = xrfi._ws_flag_waterfall(
                    vis_detrended, these_flags, nsig=ws_nsig
                )

            # only use data in flagged channels
            this_rfi = np.where(these_flags, rfi_data, 0)

            # update the new arrays
            new_rfi_data[blt_inds, 0, :, pol_inds[0]] = this_rfi
            new_rfi_flags[blt_inds, 0, :, pol_inds[0]] = these_flags

            if conj_blt_inds.size != 0:
                new_rfi_data[conj_blt_inds, 0, :, pol_inds[1]] = this_rfi.conj()
                new_rfi_flags[conj_blt_inds, 0, :, pol_inds[1]] = these_flags

    # actually write the data and flags to a new UVData object
    rfi_station_uvd = copy.deepcopy(rfi_uvd)
    rfi_station_uvd.data_array = new_rfi_data
    rfi_station_uvd.flag_array = new_rfi_flags
    return rfi_station_uvd

