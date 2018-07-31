# RF Interference Data System (RIDS)
Reads/writes .ridm/.ridz files, JSON files with fields as described below.
Timestamps should be sortable to increasing time (can fix this later if desired...).

Any field may be omitted or missing.
* This first set is metadata - typically stored in a .ridm file that gets read
  * ident: description of filename
  * instrument:  description of the instrument used
  * receiver:  description of receiver used
  * channel_width:  RF bandwidth (width in file or FFT)
  * channel_width_unit:  unit of bandwidth
  * time_constant: averaging time/maxhold reset time
                   though not ideal, can be a descriptive word or word pair
                   for e.g. ongoing maxhold, etc
  * time_constant_unit:  unit of time_constant
  * freq_unit:  unit of frequency used in spectra
  * val_unit: unit of value used in spectra
  * comment:  general comment; reader appends, doesn't overwrite
  * time_format:  string indicating the format of timestamp in filename
* These are typically set in data-taking session
  * rid_file:  records where the meta-data came from
  * nsets:  number of feature_sets included in file
  * timestamp_first:  timestamp for first feature_set (currently assumes timestamps sort)
  * timestamp_last:           "     last          "                 "
  * feature_sets:  features etc defined in the feature module


# SpectrumPeak
Adds additional attributes and defines the feature_sets.  Redefines reader/writer/info to
include the additional attributes.  Current scripts are `specpeak.py` and `sphandle.py`

The feature_sets are in a dictionary, with a key_name of `[optional_name.]timestamp.polarization`

If optional_name is included and is (currently) 'data', 'baseline', 'cal' this just saves the data as the
appropriate feature_component (currently val, maxhold, or minhold).  Otherwise, it peak-fits on default
feature_component (unless another is specified) and saves the peaks and bandwidth for a given threshold.

## To run in a session
Primarily use scripts, however in a python session
```
import rids
r = rids.spectrum_peak.spectrum_peak.SpectrumPeak()
r.reader(<fn>)
r.info() shows info about file
r.viewer() will plot per the definition in feature_module
```
