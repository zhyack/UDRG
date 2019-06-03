#coding: UTF-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import readability
import argparse
import codecs
parser = argparse.ArgumentParser(
    description="Calculate BLEU scores for the input hypothesis and reference files")
parser.add_argument(
    "-f",
    nargs=1,
    dest="pf",
    type=str,
    help="The path of the hypothesis file.")
args = parser.parse_args()

f = codecs.open(args.pf[0], 'r', 'UTF-8')

lines = f.readlines()
s = ''.join(lines)
scores = readability.multi_sents(lines)
print('Split:\nARI: \t%.2f\nFRE: \t%.2f\nFKGL: \t%.2f\nGFI: \t%.2f\nSI: \t%.2f\nCLT: \t%.2f\nLIX: \t%.2f\nRIX: \t%.2f\n'%tuple(scores))
scores = readability.singel_sent(s)
print('\n\nWhole:\nARI: \t%.2f\nFRE: \t%.2f\nFKGL: \t%.2f\nGFI: \t%.2f\nSI: \t%.2f\nCLT: \t%.2f\nLIX: \t%.2f\nRIX: \t%.2f\n'%tuple(scores))
