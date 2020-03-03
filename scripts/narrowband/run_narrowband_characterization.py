# -*- mode: python; coding: utf-8 -*-
# Copyright 2018 the HERA Collaboration
# Licensed under the 2-clause BSD license.

import copy
import glob
import itertools
import json
import multiprocessing
import os
import re
import warnings

import numpy as np
from astropy import units

import uvtools
from hera_qm import xrfi
from pyuvdata import UVData
from rids.features import narrowband
if hera_sim.version.version.startswith('0'):
    from hera_sim.rfi import _listify
else:
    from hera_sim.utils import _listify


# overview

# figure out IO details
# extract analysis parameters
### easiest solution: use yaml files to pull the above info
### probably good practice: make an argparser to let the user
### use this script if they aren't familiar with making yaml files

# make new UVData objects that contain only desired info
# (e.g. only autos if not using cross-correlations)
### make a module in narrowband to handle this

# run characterization routine, parallelizing over files where appropriate
### again, use a module in narrowband for doing this

# save the desired data products
# write out all the info for the rids file
# save the rids file
### make sure to subclass Rids to handle this, again in narrowband

# exit
