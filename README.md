# RF Interference Data System (RIDS)
Reads/writes .ridm/.ridz files, JSON files with fields as described below.
Timestamps should be sortable to increasing time (can fix this later if desired...).

The thought behind rids was that we would be generating a lot of data that we would:
	* want to store sensibly and
	* maybe not store it all, but store desired “features”.

The user is encouraging to think about/define/implement additional features or file formats or etc

Two main divisions are:

* meta-data/header:  descriptive attributes
* feature_sets:  dictionary containing the data

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

File "types" are shown below (the quotes are because they are all the same json structure, just different intents)
  * .ridm files are ascii json files, so use your favorite editor.  They are meant to hold the header or meta data of the instrument (so small amount of specific content).
  * .rids files are ascii json files, so can use your favorite editor.  They are meant to hold a data set (with header/meta info)
  * .ridz files are gzipped .rids files that are intended to hold a bunch of data.  If you want to view them with an editor, cp or mv them to a .gz filename, then gunzip them and view with editor.
	The script `zipr.py` can convert between zipped and unzipped file.


I'm trying to do proper unit testing, but so far haven't set it up properly.  Currently,
`run_ridstests.py` while in the **tests** subdirectory

# SpectrumPeak
Adds additional attributes and defines the feature_sets.  Redefines reader/writer/info to
include the additional attributes.  Current scripts are:

* `specpeak.py`
generates spectrum_peak ridz files from spectrum data files.  Includes a few general tools.

* `rdhandle.py`
handles ridz files for display, analysis, ... Currently only plots waterfall or stack of raw-data
from named file

## To run in a session
Primarily use scripts, however in a python session
```
import rids
r = rids.spectral.spectrum_peak.SpectrumPeak()
r.reader(<fn>)
r.info() shows info about file
r.viewer() will plot per the definition in feature_module
```

The feature_sets are in a dictionary, with a key_name of `[optional_name.]timestamp.polarization`

If optional_name is included and is (currently) 'data', 'baseline', or 'cal' this just saves the data as the
appropriate feature_component (currently val, maxhold, or minhold).  Otherwise, it peak-fits on default
feature_component (unless another is specified) and saves the peaks and bandwidth for a given threshold.

## Hopefully helpful musings for SpectrumPeak

So far, the only "feature" set included is `SpectrumPeak`, with the feature being peaks in a spectrum (and includes a bandwidth).  It also can save the raw spectra along with it (as many or as few as you specify).  With the limited amount of data so far, I have actually just been archiving all of the spectra.  The primary script to generate the files is `specpeak.py`.  Spectra can be 'minhold', 'maxhold', or 'val' (this goes into the filename per below), and it groups these within a 'feature_set' in the written file (sorted by time).

To use in practice, the instrument will write spectra to files with a specified format and filename (other options may be included, but currently only one) and then you'll run the script to process the files in that directory.  [NOTE/WARNING:  it deletes files as it goes.]  `specpeak.py fnhelp` will display the filename format, reproduced here:

```
The filename format convention is:
	identifier.{time-stamp}.feature_component.polarization

i.e. x=filename.split(.) has:
	x[0]:  arbitrary identifier (no .'s may be in it)
	x[...]: time-stamp (may have .'s in it)
	x[-2]:  feature_component (currently one of 'maxhold', 'minhold', 'val')
	x[-1]:  polarization
```

To archive all data (of any id) in a directory where all data have the same frequencies:

`specpeak.py my_instrument.ridm --archive_data --share_freq`

This will write a .ridz file `id_feature.first_timestamp.n#_featuresets.None.ridz`

You can examine it with the `-i`, `-v` or `-k` specpeak flags.

If you want to use the peak stuff to reduce the amount of data saved, but archive every e.g. 100th spectra:

`specpeak.py my_instrument.ridm -r +100`

This uses the default number of feature_sets per pol (currently 10000) and a threshold contained within my_instrument.ridm.


```
~/rids$ specpeak.py -h
usage: specpeak.py [-h] [--directory DIRECTORY] [-i] [-v] [-k]
                   [--archive_data] [--data_only_override] [-r RAWDATA]
                   [-c [COMMENT]] [--id ID] [-# SETS_PER_POL] [--share_freq]
                   [--peak_on PEAK_ON] [--view_peaks_ongoing] [--data_only]
                   [--ecal ECAL] [--ncal NCAL] [--show_fc SHOW_FC]
                   [--threshold_view THRESHOLD_VIEW]
                   rid_filename

positional arguments:
  rid_filename          rids meta-data filename or filename to be viewed
                        (note: type fnhelp to see format of spectrum
                        filenames)

optional arguments:
  -h, --help            show this help message and exit
  --directory DIRECTORY
                        directory for process files and where parameter/rids
                        file lives
  -i, --info            show the info for provided filename
  -v, --view            show plot for provided filename
  -k, --keys            Show the feature_set keys
  --archive_data        Flag to archive all data (shortcut for data_only=True
                        and rawdata='+1').
  --data_only_override  flag to force data_only without saving all
  -r RAWDATA, --rawdata RAWDATA
                        csv indices for raw data to keep, or +step ('n' to
                        stop if view)
  -c [COMMENT], --comment [COMMENT]
                        append a comment or includes comments in keys printout
                        if set
  --id ID               can be a specific id name or 'all'
  -# SETS_PER_POL, --sets_per_pol SETS_PER_POL
                        number of sets per pol per file
  --share_freq          invoke if you know all spectra have same freq axis
  --peak_on PEAK_ON     Peak on event component (if other than max->min->val)
  --view_peaks_ongoing  view all peaks in process (diagnostic only!)
  --data_only           flag to only store data and not peaks
  --ecal ECAL           E-pol cal filename
  --ncal NCAL           N-pol cal filename
  --show_fc SHOW_FC     csv list of feature components to show (if different)
  --threshold_view THRESHOLD_VIEW
                        new threshold for viewing (if possible)


```
