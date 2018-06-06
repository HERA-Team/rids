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
            time_stamp:  time_stamp for file/baseline data event
            cal:  calibration data by polarization
            events:  baseline or ave/maxhold spectra
    """
    dattr = ['ident', 'instrument', 'receiver', 'time_stamp', 'comment', 'freq_unit', 'val_unit', 'nevents']
    uattr = ['channel_width', 'time_constant', 'threshold', 'vbw']
    spectral_fields = ['comment', 'polarization', 'freq', 'val', 'ave', 'maxhold', 'minhold']
    polarizations = ['E', 'N']
    event_components = ['maxhold', 'minhold', 'ave']

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
        self.time_stamp = None
        self.nevents = 0
        self.cal = {}
        for pol in self.polarizations:
            self.cal[pol] = Spectral(polarization=pol)
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
        for d in self.dattr:
            if d in data:
                if d == 'comment':
                    self.append_comment(data[d])
                else:
                    setattr(self, d, data[d])
        for d in self.uattr:
            if d in data:
                self._set_uattr(d, data[d])
        if 'cal' in data:
            for pol in self.polarizations:
                if pol in data['cal']:
                    for v in self.spectral_fields:
                        if v in data['cal'][pol]:
                            setattr(self.cal[pol], v, data['cal'][pol][v])
        if 'events' in data:
            for d in data['events']:
                self.events[d] = Spectral()
                for v in self.spectral_fields:
                    if v in data['events'][d]:
                        setattr(self.events[d], v, data['events'][d][v])

    def _set_uattr(self, d, x):
        v = x.split()
        if len(v) == 1:
            d0 = v[0]
            d1 = None
        elif len(v) > 1:
            try:
                d0 = float(v[0])
            except ValueError:
                d0 = v[0]
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
        ds['cal'] = {'E': {}, 'N': {}}
        for pol in self.polarizations:
            for v in self.spectral_fields:
                try:
                    ds['cal'][pol][v] = getattr(self.cal[pol], v)
                except AttributeError:
                    continue
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
        **fnargs are filenames for self.event_components
        """
        event_name = event + polarization
        self.events[event_name] = Spectral(polarization=polarization)
        self.events[event_name].freq = []
        spectra = {}
        for ec, fn in fnargs.iteritems():
            if ec not in self.event_components:
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

        if 'maxhold' in fnargs.keys():
            self.peak_finder(spectra['maxhold'])
        elif 'minhold' in fnargs.keys():
            self.peak_finder(spectra['minhold'])
        elif 'ave' in fnargs.keys():
            self.peak_finder(spectral['ave'])
        else:
            return
        self.events[event_name].freq = list(np.array(self.hipk_freq)[self.hipk])
        for ftype in fnargs:
            try:
                setattr(self.events[event_name], ftype, list(np.array(spectra[ftype].val)[self.hipk]))
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
        sfmt = {'maxhold': 'v', 'minhold': '^', 'ave': '_'}
        sclr = {'maxhold': 'r', 'minhold': 'b', 'ave': 'k'}
        clrs = ['r', 'b', 'k', 'g', 'm', 'c', 'y', '0.25', '0.5', '0.75']
        c = 0
        for e, v in self.events.iteritems():
            is_baseline = 'baseline' in e.lower()
            if is_baseline and not show_baseline:
                continue
            if is_baseline:
                show_color = [sclr[x] for x in show_components]
            else:
                show_color = [clrs[c % len(clrs)]] * len(self.event_components)
                c += 1
            for sp, s in zip(show_components, show_color):
                try:
                    spectrum_plotter(self.rid_file, e, v.freq, getattr(v, sp), sfmt[sp], s)
                except AttributeError:
                    pass

    def info(self):
        print("RIDS Information")
        for d in self.dattr:
            print("\t{}:  {}".format(d, getattr(self, d)))
        for d in self.uattr:
            print("\t{}:  {} {}".format(d, getattr(self, d), getattr(self, d + '_unit')))
        print("\t{} cal E".format(len(self.cal['E'].freq) > 0))
        print("\t{} cal N".format(len(self.cal['N'].freq) > 0))
        print("\t{} events".format(len(self.events)))

    def stats(self):
        print("Provide standard set of occupancy etc stats")

    def apply_cal(self):
        print("Apply the calibration, if available.")

    def process_files(self, directory, events_per_file=100, max_loops=1000):
        """
        This is the standard method to process spectrum files in a directory to
        produce ridz files.

        Format of the spectrum filename (peel_filename below):
        <path>/identifier:time_stamp.event_component.polarization

        Parameters:
        ------------
        directory:  directory in which the files reside and where the ridz files
                    get written
        events_per_file:  number of events (see event_components) per ridz file
        max_loops:  will stop after this number
        """
        loop = True
        loop_ctr = 0
        while (loop):
            loop_ctr += 1
            if loop_ctr > max_loops:
                break
            available_files = sorted(os.listdir(directory))
            f = {}
            for ec in self.event_components:
                f[ec] = {}
                f[ec]['cnt'] = {}
                for pol in self.polarizations:
                    f[ec][pol] = []
                    f[ec]['cnt'][pol] = 0
            loop = False
            for af in available_files:
                fnd = peel_filename(af, self.event_components)
                if not len(fnd):
                    continue
                if fnd['event_component'] in f and fnd['polarization'] in self.polarizations:
                    loop = True
                    f[fnd['event_component']][fnd['polarization']].append(os.path.join(directory, af))
                    f[fnd['event_component']]['cnt'][fnd['polarization']] += 1
            max_pol_cnt = {}
            for pol in self.polarizations:
                max_pol_cnt[pol] = 0
            for pol in self.polarizations:
                for ec in self.event_components:
                    if f[ec]['cnt'][pol] > max_pol_cnt[pol]:
                        max_pol_cnt[pol] = f[ec]['cnt'][pol]
                for ec in self.event_components:
                    diff_len = max_pol_cnt[pol] - len(f[ec][pol])
                    if diff_len > 0:
                        f[ec][pol] = f[ec][pol] + [None] * diff_len
            # This part now "specializes" to the event_components
            for pol in self.polarizations:
                if not max_pol_cnt[pol]:
                    continue
                axn = {'a': f['ave'][pol][0], 'x': f['maxhold'][pol][0], 'n': f['minhold'][pol][0]}
                for a0 in axn.values():
                    fnd = peel_filename(a0)
                    if len(fnd):
                        break
                self.set(time_stamp=fnd['time_stamp'], ident=fnd['ident'])
                self.events = {}
                self.nevents = 0
                self.get_event('baseline_', pol, ave=axn['a'], maxhold=axn['x'], minhold=axn['n'])
                for a, x, n in zip(f['ave'][pol][:events_per_file],
                                   f['maxhold'][pol][:events_per_file],
                                   f['minhold'][pol][:events_per_file]):
                    for axn in [a, x, n]:
                        fnd = peel_filename(axn)
                        if len(fnd):
                            break
                    self.get_event(fnd['time_stamp'], pol, ave=a, maxhold=x, minhold=n)
                    if a is not None:
                        os.remove(a)
                    if x is not None:
                        os.remove(x)
                    if n is not None:
                        os.remove(n)
            fn = "{}{}.s{}.T{}.ridz".format(self.ident, self.time_stamp, self.nevents, self.threshold)
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


def spectrum_plotter(name, e, x, y, fmt, clr):
    import matplotlib.pyplot as plt
    try:
        plt.figure(name)
        _X = x[:len(y)]
        if 'baseline' in e.lower():
            plt.plot(_X, y, clr)
        else:
            plt.plot(_X, y, fmt, color=clr)
    except ValueError:
        _Y = y[:len(x)]
        if 'baseline' in e.lower():
            plt.plot(x, _Y, clr)
        else:
            plt.plot(x, _Y, fmt, color=clr)


def peel_filename(v, eclist=None):
    if v is None:
        return {}
    s = v.split('/')[-1].split(':')
    if len(s) != 2:
        return {}
    ident = s[0]
    s = s[1].split('.')
    if len(s) != 3:
        return {}
    fnd = {'ident': ident}
    fnd['time_stamp'] = s[0]
    fnd['event_component'] = s[1].lower()
    fnd['polarization'] = s[2].upper()
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
