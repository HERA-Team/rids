from __future__ import print_function
import os
import numpy as np

available_files = sorted(os.listdir('.'))
df = np.load(available_files[4])
num_chan = 2048
freq = (250.0 / num_chan) * np.arange(num_chan)
day_to_use = None

for af in available_files:
    ident = af.split('.')[0]
    if ident != 'zen':
        continue
    day = af.split('.')[1]
    time_stamp = '.'.join(af.split('.')[1:3])
    if day_to_use is not None and day != day_to_use:
        continue
    print(ident, time_stamp)
    df = np.load(af)
    c = np.zeros(num_chan)
    right_num_chan = True
    for a, v in df.iteritems():
        if a == 'times':
            continue
        if len(v) != num_chan:
            right_num_chan = False
            break
        c += v
    if right_num_chan:
        c /= len(df['times'])
        nfn = "dataout/{}_{}.ave.E".format(ident, time_stamp)
        with open(nfn, 'w') as fp:
            for fc, cc in zip(freq, c):
                fp.write("{} {}\n".format(fc, cc))
