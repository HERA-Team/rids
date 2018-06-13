from __future__ import print_function, absolute_import, division
import os
import numpy as np
import rids_rw
# import peaks  # Scipy option
import peak_det  # Another option...
import bw_finder


class Peak:
    """
                peaked_on:  event_component on which peaks were found
                threshold:  value used to threshold peaks
                threshold_unit: unit of threshold
    """
    direct_attributes = ['comment', 'peaked_on', 'delta', 'bw_range', 'delta_values']
    unit_attributes = ['threshold', 'rbw', 'vbw']
    feature_components = ['maxhold', 'minhold', 'val']

    def __init__(self, comment=''):
        for d in self.direct_attributes:
            setattr(self, d, None)
        for d in self.unit_attributes:
            setattr(self, d, None)
            setattr(self, d + '_unit', None)
        # Re-initialize attributes
        self.comment = comment
        self.bw_range = []  # Range to search for bandwidth
        self.delta_values = {'zen': 0.1, 'sa': 1.0}
        # Other attributes
        self.rids = rids_rw.RidsReadWrite(self)
        self.hipk = None
        self.hipk_bw = None

    def reset(self):
        self.__init__()

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
            self.filename = os.path.join(directory, fn)
            self.writer(self.filename)


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
