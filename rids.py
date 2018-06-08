from __future__ import print_function
import json
import os
import copy
import numpy as np
import gzip
import peaks


class Spectral:
    def __init__(self, polarization='', comment=''):
        self.comment = comment
        self.polarization = polarization
        self.freq = []
        self.val = []


class Rids:
    """
    RF Interference Data System (RIDS)
    Reads/writes .rids/[.ridz] files, [zipped] JSON files with fields as described below.
    Any field may be omitted or missing.
      - This first set is header information - typically stored in a .rids file that gets read/rewritten
            ident: description of filename
            instrument:  description of the instrument used
            receiver:  description of receiver used
            channel_width:  RF bandwidth
            channel_width_unit:  unit of bandwidth
            vbw: video bandwidth (typically used for spectrum analyzer)
            vbw_unit: unit of bandwidth
            nevents:  number of events included in file
            time_constant: averaging time/maxhold reset time
                           though not ideal, can be a descriptive word or word pair
                           for e.g. ongoing maxhold, etc
            time_constant_unit:  unit of time_constant
            threshold:  value used to threshold peaks
            threshold_unit: unit of threshold
            freq_unit:  unit of frequency used in spectra
            val_unit: unit of value used in spectra
            comment:  general comment; reader appends, doesn't overwrite
      - These are typically set in data-taking session
            time_stamp_first:  time_stamp for first file/baseline data event
            time_stamp_last:           "      last          "
            cal:  calibration data by polarization
            events:  baseline or ave/maxhold spectra
    """
    dattr = ['ident', 'instrument', 'receiver', 'time_stamp_first', 'time_stamp_last',
             'comment', 'freq_unit', 'val_unit', 'nevents']
    uattr = ['channel_width', 'time_constant', 'threshold', 'vbw']
    spectral_fields = ['comment', 'polarization', 'freq', 'val', 'maxhold', 'minhold']
    polarizations = ['E', 'N', 'I']
    event_components = ['maxhold', 'minhold', 'val']

    def __init__(self, comment=None):
        self.ident = None
        self.instrument = None
        self.receiver = None
        self.channel_width = None
        self.channel_width_unit = None
        self.vbw = None
        self.vbw_unit = None
        self.time_constant = None
        self.time_constant_unit = None
        self.threshold = None
        self.threshold_unit = None
        self.freq_unit = None
        self.val_unit = None
        self.comment = comment
        self.time_stamp_first = None
        self.time_stamp_last = None
        self.nevents = 0
        self.cal = {}
        self.events = {}
        # --Other variables--
        self.rid_file = None
        self.hipk = None

    def reset(self):
        self.__init__(None)

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
            elif d in self.dattr:
                setattr(self, d, X)
            elif d in self.uattr:
                self._set_uattr(d, X)
            elif d == 'cal':
                for pol in X:
                    if pol not in self.polarizations:
                        print("Unexpected pol {} in {}".format(pol, d))
                        continue
                    self.cal[pol] = Spectral(pol)
                    for v, Y in X[pol].iteritems():
                        if v not in self.spectral_fields:
                            print("Unexpected field {} in {}".format(v, d))
                            continue
                        setattr(self.cal[pol], v, Y)
            elif d == 'events':
                for e in X:
                    self.events[e] = Spectral()
                    for v, Y in X[e].iteritems():
                        if v not in self.spectral_fields:
                            print("Unexpected field {} in {}".format(v, d))
                            continue
                        setattr(self.events[e], v, Y)

    def _set_uattr(self, d, x):
        v = x.split()
        try:
            d0 = float(v[0])
        except ValueError:
            d0 = v[0]
        d1 = None
        if len(v) > 1:
            d1 = v[1]
        setattr(self, d, d0)
        setattr(self, d + '_unit', d1)

    def writer(self, filename, fix_list=True):
        """
        This writes a RID file with a full structure
        """
        ds = {}
        for d in self.dattr:
            ds[d] = getattr(self, d)
        for d in self.uattr:
            ds[d] = "{} {}".format(getattr(self, d), getattr(self, d + '_unit'))
        caltmp = {}
        for pol in self.polarizations:
            for v in self.spectral_fields:
                try:
                    X = getattr(self.cal[pol], v)
                    if pol not in caltmp:
                        caltmp[pol] = {}
                    caltmp[pol][v] = X
                except AttributeError:
                    continue
                except KeyError:
                    continue
        if len(caltmp):
            ds['cal'] = caltmp
        ds['events'] = {}
        for d in self.events:
            ds['events'][d] = {}
            for v in self.spectral_fields:
                try:
                    ds['events'][d][v] = getattr(self.events[d], v)
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
            if k in self.dattr:
                setattr(self, k, kwargs[k])
            elif k in self.uattr:
                self._set_uattr(k, kwargs[k])

    def append_comment(self, comment):
        if comment is None:
            return
        if self.comment is None:
            self.comment = comment
        else:
            self.comment += ('\n' + comment)

    def get_event(self, event, polarization, **fnargs):
        """
        **fnargs are filenames for self.event_components {"event_component": <filename>}
        """
        event_name = event + polarization
        self.events[event_name] = Spectral(polarization=polarization)
        self.events[event_name].freq = []
        spectra = {}
        for ec, fn in fnargs.iteritems():
            if ec not in self.event_components or fn is None:
                continue
            spectra[ec] = Spectral()
            spectrum_reader(fn, spectra[ec], polarization)
            if 'baseline' in event.lower():
                if len(spectra[ec].freq) > len(self.events[event_name].freq):
                    self.events[event_name].freq = spectra[ec].freq
                setattr(self.events[event_name], ec, spectra[ec].val)
        self.nevents += 1
        if 'baseline' in event.lower():
            return

        for event_component in self.event_components:
            for ec, fn in fnargs.iteritems():
                if ec != event_component or fn is None:
                    continue
                self.peak_finder(spectra[ec])
                break
            break
        else:
            return
        self.events[event_name].freq = list(np.array(self.hipk_freq)[self.hipk])
        for ec, fn in fnargs.iteritems():
            if fn is None:
                continue
            try:
                setattr(self.events[event_name], ec, list(np.array(spectra[ec].val)[self.hipk]))
            except IndexError:
                pass

    def peak_finder(self, spec, cwt_range=[1, 7], rc_range=[4, 4]):
        self.hipk_freq = spec.freq
        self.hipk_val = spec.val
        self.hipk = peaks.fp(spec.val, self.threshold, cwt_range, rc_range)

    def peak_viewer(self):
        if self.hipk is None:
            return
        import matplotlib.pyplot as plt
        plt.plot(self.hipk_freq, self.hipk_val)
        plt.plot(np.array(self.hipk_freq)[self.hipk], np.array(self.hipk_val)[self.hipk], 'kv')

    def viewer(self, show_components='all', show_baseline=True):
        """
        Parameters:
        ------------
        show_components:  event_components to show (list) or 'all'
        show_baseline:  include baseline spectra (Boolean)
        """
        if isinstance(show_components, (str, unicode)):
            show_components = self.event_components
        sfmt = {'maxhold': 'v', 'minhold': '^', 'val': '_'}
        sclr = {'maxhold': 'r', 'minhold': 'b', 'val': 'k'}
        clrs = ['r', 'b', 'k', 'g', 'm', 'c', 'y', '0.25', '0.5', '0.75']
        lss = ['-', '--', ':']
        c = 0
        bl = 0
        for e, v in self.events.iteritems():
            is_baseline = 'baseline' in e.lower()
            if is_baseline and not show_baseline:
                continue
            if is_baseline:
                show_color = [sclr[x] for x in show_components]
                show_linestyle = [lss[bl % len(lss)]] * len(self.event_components)
                bl += 1
            else:
                show_linestyle = [None] * len(self.event_components)
                show_color = [clrs[c % len(clrs)]] * len(self.event_components)
                c += 1
            for sp, s, ls in zip(show_components, show_color, show_linestyle):
                try:
                    spectrum_plotter(self.rid_file, e, v.freq, getattr(v, sp), sfmt[sp], s, ls)
                except AttributeError:
                    pass

    def info(self):
        print("RIDS Information")
        for d in self.dattr:
            print("\t{}:  {}".format(d, getattr(self, d)))
        for d in self.uattr:
            print("\t{}:  {} {}".format(d, getattr(self, d), getattr(self, d + '_unit')))
        for p in self.polarizations:
            if p in self.cal:
                print("\tcal {} {}".format(pol, len(self.cal['E'].freq) > 0))
        print("\t{} events".format(len(self.events)))

    def stats(self):
        print("Provide standard set of occupancy etc stats")

    def apply_cal(self):
        print("Apply the calibration, if available.")

    def process_files(self, directory='.', ident='all', baseline=[0, -1], events_per_pol=100, max_loops=1000):
        """
        This is the standard method to process spectrum files in a directory to
        produce ridz files.

        Format of the spectrum filename (peel_filename below):
        <path>/identifier.time-stamp.event_component.polarization

        Parameters:
        ------------
        directory:  directory where files reside and ridz files get written
        idents:  idents to do.  If 'all' processes all, picking first for overall ident
        baseline:  indices of events to be saved as baseline spectra
        events_per_pol:  number of events (see event_components) per pol per ridz file
        max_loops:  will stop after this number
        """
        loop = True
        loop_ctr = 0
        self.ident = None
        while (loop):
            loop_ctr += 1
            if loop_ctr > max_loops:
                break
            # Set up the event_component file dictionary
            f = {}
            max_pol_cnt = {}
            for pol in self.polarizations:
                f[pol] = {}
                max_pol_cnt[pol] = 0
                for ec in self.event_components:
                    f[pol][ec] = []
            # Go through and sort the available files into file dictionary
            loop = False
            available_files = sorted(os.listdir(directory))
            file_times = []
            for af in available_files:
                fnd = peel_filename(af, self.event_components)
                if not len(fnd) or\
                        fnd['polarization'] not in self.polarizations or\
                        fnd['event_component'] not in self.event_components:
                    continue
                if ident == 'all' or ident == fnd['ident']:
                    loop = True
                    file_list = f[fnd['polarization']][fnd['event_component']]
                    if len(file_list) > events_per_pol:
                        continue
                    file_times.append(fnd['time_stamp'])
                    file_list.append(os.path.join(directory, af))
                    if self.ident is None:
                        self.ident = fnd['ident']
            if not loop:
                break
            # Go through and count the sorted available files
            num_to_read = {}
            for pol in f:
                for ec in f[pol]:
                    if len(f[pol][ec]) > max_pol_cnt[pol]:
                        max_pol_cnt[pol] = len(f[pol][ec])
                for ec in f[pol]:
                    diff_len = max_pol_cnt[pol] - len(f[pol][ec])
                    if diff_len > 0:
                        f[pol][ec] = f[pol][ec] + [None] * diff_len
                    num_to_read[pol] = len(f[pol][ec])  # Yes, does get reset
            file_times = sorted(file_times)
            self.time_stamp_first = file_times[0]
            self.time_stamp_last = file_times[-1]
            # Process the files
            self.events = {}
            self.nevents = 0
            for pol in self.polarizations:
                if not max_pol_cnt[pol]:
                    continue
                # Get the baseline(s)
                for i in baseline:
                    bld = {}
                    for ec, ecfns in f[pol].iteritems():
                        fnd = peel_filename(ecfns[i], self.event_components)
                        if len(fnd):
                            bld[ec] = ecfns[i]
                    evn = 'baseline{}.'.format(i)
                    self.get_event(evn, pol, **bld)
                # Get the events
                for i in range(num_to_read[pol]):
                    ecd = {}
                    for ec, ecfns in f[pol].iteritems():
                        fnd = peel_filename(ecfns[i], self.event_components)
                        if len(fnd):
                            ecd[ec] = ecfns[i]
                    evn = fnd['time_stamp']
                    self.get_event(evn, pol, **ecd)
                    for x in ecd.values():
                        if x is not None:
                            os.remove(x)
            # Write the ridz file
            th = self.threshold
            if abs(self.threshold) < 1.0:
                th *= 100.0
            fn = "{}.{}.e{}.T{:.0f}.ridz".format(self.ident, self.time_stamp_first, self.nevents, th)
            output_file = os.path.join(directory, fn)
            self.writer(output_file)


def spectrum_reader(filename, spec, polarization=None):
    """
    This reads in an ascii spectrum file.
    """
    if filename is None:
        return
    if polarization is not None:
        spec.polarization = polarization
    with open(filename, 'r') as f:
        for line in f:
            data = [float(x) for x in line.split()]
            spec.freq.append(data[0])
            spec.val.append(data[1])


def spectrum_plotter(figure_name, event_name, x, y, fmt, clr, ls):
    import matplotlib.pyplot as plt
    if not len(x) or not len(y):
        return
    try:
        plt.figure(figure_name)
        _X = x[:len(y)]
        if 'baseline' in event_name.lower():
            plt.plot(_X, y, clr, linestyle=ls)
        else:
            plt.plot(_X, y, fmt, color=clr)
    except ValueError:
        _Y = y[:len(x)]
        if 'baseline' in event_name.lower():
            plt.plot(x, _Y, clr, linestyle=ls)
        else:
            plt.plot(x, _Y, fmt, color=clr)


def peel_filename(v, eclist=None):
    if v is None:
        return {}
    s = v.split('/')[-1].split('.')
    if len(s) not in [4, 5]:  # time_stamp may have one '.' in it
        return {}
    fnd = {'ident': s[0]}
    fnd['time_stamp'] = '.'.join(s[1:-2])
    fnd['event_component'] = s[-2].lower()
    fnd['polarization'] = s[-1].upper()
    if eclist is None:
        return fnd
    eclist = [x.lower() for x in eclist]
    for ec in eclist:
        if fnd['event_component'] in ec:
            fnd['event_component'] = ec
            return fnd
    return {}


def fix_json_list(jsd):
    spaces = ['\n', ' ']
    fixed = ''
    in_list = False
    for c in jsd:
        if c == '[':
            in_list = True
        elif c == ']':
            in_list = False

        if not in_list:
            fixed += c
        elif in_list and c not in spaces:
            fixed += c
    return fixed
