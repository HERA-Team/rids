#! /usr/bin/env python
# _*_ coding: utf-8 _*_
# Copy 2018 the HERA Project
# Licensed under the 2-clause BSD license
from __future__ import print_function, division, absolute_import

import argparse
import gzip

ap = argparse.ArgumentParser()
ap.add_argument('file', help="file to convert, depending on extension", default=None)
ap.add_argument('-d', '--delete', help="Set flag to delete other version of file.", action='store_true')

args = ap.parse_args()


if __name__ == '__main__':
    file_type = args.file.split('.')[-1]
    if file_type == 'ridz':
        out_filename = '.'.join(args.file.split('.')[:-1]) + '.rids'
        infp = gzip.GzipFile(args.file, 'rb')
        outfp = open(out_filename, 'wb')
    else:
        out_filename = '.'.join(args.file.split('.')[:-1]) + '.ridz'
        infp = open(args.file, 'rb')
        outfp = gzip.GzipFile(out_filename, 'wb')

    s = infp.read()
    infp.close()
    outfp.write(s)
    outfp.close()

    if args.delete:
        import os
        os.remove(args.file)
