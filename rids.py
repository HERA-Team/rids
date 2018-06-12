from __future__ import print_function, absolute_import, division
import json
import os
import gzip
import copy
import numpy as np
# import peaks  # Scipy option
import peak_det  # Another option...
import bw_finder


class Spectral:
    def __init__(self, polarization='', comment=''):
        self.comment = comment
        self.polarization = polarization
        self.freq = []
        self.val = []


class PeakSettings:
    """
                peaked_on:  event_component on which peaks were found
                threshold:  value used to threshold peaks
                threshold_unit: unit of threshold
    """
    dattr = ['comment', 'peaked_on', 'delta', 'bw_range', 'delta_values']
    uattr = ['threshold']

    def __init__(self, comment=''):
        self.comment = comment
        self.peaked_on = None
        self.threshold = None
        self.threshold_unit = None
        self.delta = None
        self.bw_range = []  # Range to search for bandwidth
        self.delta_values = {'zen': 0.1, 'sa': 1.0}


class Rids:
    """
    RF Interference Data System (RIDS)
    Reads/writes .rids/[.ridz] files, [zipped] JSON files with fields as described below.
    Any field may be omitted or missing.
      - This first set is header information - typically stored in a .rids file that gets read/rewritten
            ident: description of filename
            instrument:  description of the instrument used
            receiver:  description of receiver used
            channel_width:  RF bandwidth (width in file or FFT)
            channel_width_unit:  unit of bandwidth
            vbw: video bandwidth (typically used for spectrum analyzer)
            vbw_unit: unit of bandwidth
            rbw:  resolution bandwidth (typically used for spectrum analyzer)
            rbw_unit:  unit of bandwidth
            nevents:  number of events included in file
            time_constant: averaging time/maxhold reset time
                           though not ideal, can be a descriptive word or word pair
                           for e.g. ongoing maxhold, etc
            time_constant_unit:  unit of time_constant
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
             'time_format', 'comment', 'freq_unit', 'val_unit', 'nevents']
    uattr = ['channel_width', 'time_constant', 'rbw', 'vbw']
    spectral_fields = ['comment', 'polarization', 'freq', 'val', 'maxhold', 'minhold', 'bw']
    polarizations = ['E', 'N', 'I']
    event_components = ['maxhold', 'minhold', 'val']

    def __init__(self, comment=None, **diagnose):
        self.rid_file = None
        self.ident = None
        self.instrument = None
        self.receiver = None
        self.channel_width = None
        self.channel_width_unit = None
        self.rbw = None
        self.rbw_unit = None
        self.vbw = None
        self.vbw_unit = None
        self.time_constant = None
        self.time_constant_unit = None
        self.freq_unit = None
        self.val_unit = None
        self.comment = comment
        self.time_stamp_first = None
        self.time_stamp_last = None
        self.time_format = None
        self.nevents = 0
        self.cal = {}
        self.events = {}
        self.peak_settings = PeakSettings()
        # --Other variables--
        self.hipk = None
        self.hipk_bw = None
        for a, b in diagnose.iteritems():
            setattr(self, a, b)

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
            elif d in self.peak_settings.dattr:
                setattr(self.peak_settings, d, X)
            elif d in self.uattr:
                set_unit_values(self, d, X)
            elif d in self.peak_settings.uattr:
                set_unit_values(self.peak_settings, d, X)
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

    def writer(self, filename, fix_list=True):
        """
        This writes a RID file with a full structure
        """
        ds = {}
        for d in self.dattr:
            ds[d] = getattr(self, d)
        for d in self.peak_settings.dattr:
            ds[d] = getattr(self.peak_settings, d)
        for d in self.uattr:
            ds[d] = "{} {}".format(getattr(self, d), getattr(self, d + '_unit'))
        for d in self.peak_settings.uattr:
            ds[d] = "{} {}".format(getattr(self.peak_settings, d),
                                   getattr(self.peak_settings, d + '_unit'))
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

    def get_event(self, event, polarization, peak_on=None, **fnargs):
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

        if peak_on is not None:
            ec = peak_on
        else:
            for ec in self.event_components:
                if ec in fnargs.keys() and fnargs[ec] is not None:
                    break
            else:
                return
        if self.peak_settings.peaked_on is None:
            self.peak_settings.peaked_on = ec
        elif ec != self.peak_settings.peaked_on:
            spectra[ec].comment += 'Peaked on different component: {} rather than {}'.format(ec, self.peak_settings.peaked_on)
        # self.peak_finder(spectra[ec], view_peaks=self.view_peaks_on_event)
        if self.ident not in self.peak_settings.delta_values:
            delta = 0.1
        else:
            delta = self.peak_settings.delta_values[self.ident]
        self.peak_det(spectra[ec], delta=delta)
        self.events[event_name].freq = list(np.array(self.hipk_freq)[self.hipk])
        self.find_bw()
        for ec, fn in fnargs.iteritems():
            if fn is None:
                continue
            try:
                setattr(self.events[event_name], ec, list(np.array(spectra[ec].val)[self.hipk]))
            except IndexError:
                pass
        setattr(self.events[event_name], 'bw', self.hipk_bw)

    def peak_det(self, spec, delta=0.1):
        self.hipk_freq = spec.freq
        self.hipk_val = spec.val
        self.hipk = peak_det.peakdet(spec.val, delta=delta, threshold=self.peak_settings.threshold)
        self.hipk = list(self.hipk)

    def find_bw(self):
        self.hipk_bw = bw_finder.bw_finder(self.hipk_freq, self.hipk_val, self.hipk, self.peak_settings.bw_range)
        if self.view_peaks_on_event:
            self.peak_viewer()

    def peak_finder(self, spec, cwt_range=[1, 3], rc_range=[4, 4]):
        self.hipk_freq = spec.freq
        self.hipk_val = spec.val
        self.hipk = peaks.fp(spec.val, self.peak_settings.threshold, cwt_range, rc_range)

    def peak_viewer(self):
        if self.hipk is None:
            return
        import matplotlib.pyplot as plt
        plt.plot(self.hipk_freq, self.hipk_val)
        plt.plot(np.array(self.hipk_freq)[self.hipk], np.array(self.hipk_val)[self.hipk], 'kv')
        if self.hipk_bw is not None:
            vv = np.array(self.hipk_val)[self.hipk]
            if 'dB' in self.val_unit:
                vv2 = vv - 6.0
            else:
                vv2 = vv / 4.0
            fl = np.array(self.hipk_freq)[self.hipk] + np.array(self.hipk_bw)[:, 0]
            fr = np.array(self.hipk_freq)[self.hipk] + np.array(self.hipk_bw)[:, 1]
            plt.plot([fl, fl], [vv, vv2], 'm')
            plt.plot([fr, fr], [vv, vv2], 'c')
        plt.show()

    def viewer(self, threshold=None, show_components='all', show_baseline=True):
        """
        Parameters:
        ------------
        show_components:  event_components to show (list) or 'all'
        show_baseline:  include baseline spectra (Boolean)
        """
        import matplotlib.pyplot as plt
        if threshold is not None and threshold < self.peak_settings.threshold:
            print("Below threshold - using {}".format(self.peak_settings.threshold))
            threshold = None
        if threshold is not None:
            print("Threshold view not implemented yet.")
            threshold = None
        if isinstance(show_components, (str, unicode)):
            show_components = self.event_components
        ec_list = {'maxhold': ('r', 'v'), 'minhold': ('b', '^'), 'val': ('k', '_'), 'bw': ('g', '|')}
        color_list = ['r', 'b', 'k', 'g', 'm', 'c', 'y', '0.25', '0.5', '0.75']
        line_list = ['-', '--', ':']
        c = 0
        bl = 0
        for e, v in self.events.iteritems():
            is_baseline = 'baseline' in e.lower()
            if is_baseline and not show_baseline:
                continue
            if is_baseline:
                clr = [ec_list[x][0] for x in self.event_components]
                ls = [line_list[bl % len(line_list)]] * len(self.event_components)
                fmt = [None] * len(self.event_components)
                bl += 1
            else:
                clr = [color_list[c % len(color_list)]] * len(self.event_components)
                ls = [None] * len(self.event_components)
                fmt = [ec_list[x][1] for x in self.event_components]
                c += 1
            for ec in show_components:
                try:
                    i = self.event_components.index(ec)
                    spectrum_plotter(self.rid_file, is_baseline, v.freq, getattr(v, ec), fmt[i], clr[i], ls[i], plt)
                except AttributeError:
                    pass
            if not is_baseline:
                try:
                    vv = np.array(getattr(v, self.peak_settings.peaked_on))
                    fv = np.array(v.freq)
                    bw = np.array(v.bw)
                    if 'dB' in self.val_unit:
                        vv2 = vv - 6.0
                    else:
                        vv2 = vv / 4.0
                    fl = fv + bw[:, 0]
                    fr = fv + bw[:, 1]
                    plt.plot([fl, fl], [vv, vv2], clr[0])
                    plt.plot([fr, fr], [vv, vv2], clr[0])
                except AttributeError:
                    continue

    def info(self):
        print("RIDS Information")
        for d in self.dattr:
            print("\t{}:  {}".format(d, getattr(self, d)))
        for d in self.peak_settings.dattr:
            print("\tpeak.{}:  {}".format(d, getattr(self.peak_settings, d)))
        for d in self.uattr:
            print("\t{}:  {} {}".format(d, getattr(self, d), getattr(self, d + '_unit')))
        for d in self.peak_settings.uattr:
            print("\tpeak.{}:  {} {}".format(d, getattr(self.peak_settings, d), getattr(self.peak_settings, d + '_unit')))
        for p in self.polarizations:
            if p in self.cal:
                print("\tcal {} {}".format(pol, len(self.cal['E'].freq) > 0))
        print("\t{} events".format(len(self.events)))

    def apply_cal(self):
        print("NOT IMPLEMENTED YET: Apply the calibration, if available.")

    def process_files(self, directory='.', ident='all', baseline=[0, -1], peak_on=False, events_per_pol=100, max_loops=1000):
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
                if 'rid' in af.split('.')[-1]:
                    continue
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
                        if abs(i) >= len(ecfns) or ecfns[i] is None:
                            continue
                        fnd = peel_filename(ecfns[i], self.event_components)
                        bld[ec] = ecfns[i]
                    if not len(bld):
                        break
                    evn = 'baseline.{}.{}.'.format(i, fnd['time_stamp'])
                    self.get_event(evn, pol, **bld)
                # Get the events
                for i in range(num_to_read[pol]):
                    ecd = {}
                    for ec, ecfns in f[pol].iteritems():
                        fnd = peel_filename(ecfns[i], self.event_components)
                        if len(fnd):
                            ecd[ec] = ecfns[i]
                    evn = fnd['time_stamp']
                    self.get_event(evn, pol, peak_on=peak_on, **ecd)
                    for x in ecd.values():
                        if x is not None:
                            os.remove(x)
            # Write the ridz file
            th = self.peak_settings.threshold
            if abs(self.peak_settings.threshold) < 1.0:
                th *= 100.0
            pk = self.peak_settings.peaked_on[:3]
            fn = "{}.{}.e{}.{}T{:.0f}.ridz".format(self.ident, self.time_stamp_first, self.nevents, pk, th)
            output_file = os.path.join(directory, fn)
            self.writer(output_file)


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


def spectrum_plotter(figure_name, is_baseline, x, y, fmt, clr, ls, plt):
    if not len(x) or not len(y):
        return
    try:
        plt.figure(figure_name)
        _X = x[:len(y)]
        if is_baseline:
            plt.plot(_X, y, clr, linestyle=ls)
        else:
            plt.plot(_X, y, fmt, color=clr)
    except ValueError:
        _Y = y[:len(x)]
        if is_baseline:
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
