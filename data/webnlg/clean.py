#coding: UTF-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import sys
sys.path.append('../../utils/')
from eutils import *
from data_utils import *

import os
import random
import codecs

def getsublist(l,s):
    ret = []
    for i in s:
        ret.append(l[i])
    return ret

def denoise(lines):
    ret = []
    for l in lines:
        l = l.split()
        t = []
        for i in range(len(l)):
            w = l[i]
            r = random.random()
            if r<0.1:
                pass
            elif r<0.2:
                t.append(w)
                t.append(w)
            elif r<0.3:
                if i+1!=len(l):
                    t.append(l[i+1])
                    l[i+1] = w
            else:
                t.append(w)
        ret.append(' '.join(t)+'\n')
    return ret

def printDictInfo(d):
    wc = 0
    for v in d.values():
        wc += v
    print('%d --- %d/%d --- %.2f%%'%(len(d), d['<UNK>'], wc, (1-float(d['<UNK>'])/wc)*100))

n_p = 500
n_u = 1000
n_d = 1000
n_all = 18102

f1 = codecs.open('train-webnlg-all-delex.triple', 'r', 'UTF-8')
f2 = codecs.open('train-webnlg-all-delex.lex', 'r', 'UTF-8')
l1 = f1.readlines()
l2 = f2.readlines()
f1.close()
f2.close()

din = l1[-n_d:]
dout = l2[-n_d:]
f1 = codecs.open('dev.in', 'w', 'UTF-8')
f2 = codecs.open('dev.out', 'w', 'UTF-8')
f1.writelines(din)
f2.writelines(dout)
f1.close()
f2.close()

lnos = range(n_all-n_d)
random.shuffle(lnos)

pin = getsublist(l1,lnos[:n_p])
pout = getsublist(l2,lnos[:n_p])
f1 = codecs.open('train.in', 'w', 'UTF-8')
f2 = codecs.open('train.out', 'w', 'UTF-8')
f1.writelines(pin)
f2.writelines(pout)
f1.close()
f2.close()

lnos = lnos[n_p:]
random.shuffle(lnos)
uin = getsublist(l1,lnos[:n_u])
random.shuffle(lnos)
uout = getsublist(l2,lnos[:n_u])
f1 = codecs.open('supplement.in', 'w', 'UTF-8')
f2 = codecs.open('supplement.out', 'w', 'UTF-8')
f1.writelines(uin)
f2.writelines(uout)
f1.close()
f2.close()


concatFiles(['train.in','supplement.in'], 'all.in')
concatFiles(['train.out', 'supplement.out'], 'all.out')

allin = codecs.open('all.in', 'r', 'UTF-8').readlines()
allin = denoise(allin)
f3 = codecs.open('dn.all.in', 'w', 'UTF-8')
f3.writelines(allin)
f3.close()
allout = codecs.open('all.out', 'r', 'UTF-8').readlines()
allout = denoise(allout)
f3 = codecs.open('dn.all.out', 'w', 'UTF-8')
f3.writelines(allout)
f3.close()


d = buildDict(['train.in'], save2file='in.dict', threshold=1)
printDictInfo(d)
d = buildDict(['train.out'], save2file='out.dict', threshold=1)
printDictInfo(d)
d = buildDict(['all.in'], save2file='all.in.dict', threshold=1)
printDictInfo(d)
d = buildDict(['all.out'], save2file='all.out.dict', threshold=1)
printDictInfo(d)
