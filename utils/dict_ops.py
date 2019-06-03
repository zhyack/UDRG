def invert_dict(d):
    return dict((v,k) for k,v in d.iteritems())
def invert_dict2(d):
    return dict([(v,k) for k,v in d.iteritems()])
from itertools import izip
def invert_dict3(d):
    return dict(izip(d.itervalues(),d.iterkeys()))
def dictSort(d, bigfirst=False):
    return sorted(d.items(),key=lambda d:d[1], reverse=bigfirst)
