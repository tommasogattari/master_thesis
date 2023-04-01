

def str2int(value):

    return int(value)

def str2float(value):

    return float(value)

def str2bool(value):

    _value = value.lower()
    return _value == '1' or _value == 'yes' or _value == 'true' or _value == 'on'

def str2none(value):

    return None