from __future__ import print_function


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
