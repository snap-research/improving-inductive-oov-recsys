import sys


def is_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def is_bool(s):
    return (s.lower() in ('true', 'false'))


def convert_bool(s):
    return s.lower() == 'true'


def is_list(s):
    return s.startswith('[') and s.endswith(']')


def convert_list(s):
    s_middle = s[1:-1]
    if not s_middle:
        return list()
    pieces = s_middle.split(',')
    return [type_convert(x) for x in pieces]


def type_convert(s: str):
    if s.isnumeric():
        return int(s)
    elif is_float(s):
        return float(s)
    elif is_bool(s):
        return convert_bool(s)
    elif is_list(s):
        return convert_list(s)
    return s


def custom_parse_args():
    argv = sys.argv
    leftover = []
    results = {}

    for arg in argv:
        if not arg.startswith('--'):
            leftover.append(arg)
            continue

        arg_pieces = arg[2:].split('=', 1)
        if len(arg_pieces) == 1:
            results[arg_pieces[0]] = True
        else:
            left, right = arg_pieces
            results[left] = type_convert(right)

    return results
