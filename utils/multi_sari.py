from sari import *
import argparse
import os
import codecs
parser = argparse.ArgumentParser(
    description="Calculate SARI scores for the input, hypothesis and reference files")
parser.add_argument(
    "-inp",
    nargs=1,
    dest="pf_input",
    type=str,
    help="The path of the input file.")
parser.add_argument(
    "-hyp",
    nargs=1,
    dest="pf_hypothesis",
    type=str,
    help="The path of the hypothesis file.")
parser.add_argument(
    "-ref",
    nargs='+',
    dest="pf_references",
    type=str,
    help="The path of the references files.")
args = parser.parse_args()
def file_exist(pf):
    if os.path.isfile(pf):
        return True
    return False

if args.pf_hypothesis!=None or args.pf_references!=None:
    if args.pf_input==None:
        raise Exception("Missing input files...")
    if args.pf_hypothesis==None:
        raise Exception("Missing references files...")
    if args.pf_references==None:
        raise Exception("Missing hypothesis files...")

    n = None
    data = []
    for pf in args.pf_input+args.pf_hypothesis+args.pf_references:
        if not file_exist(pf):
            raise Exception("File Not Found: %s"%(pf))
        f = codecs.open(pf, 'r', "utf-8", )
        data.append(f.readlines())
        if n==None:
            n = len(data[-1])
        elif n != len(data[-1]):
            raise Exception("Not parrallel: %s %d-%d"%(pf, n, len(data[-1])))
        f.close()

    inp_data = data[0]
    hyp_data = data[1]
    ref_data = list(map(list, zip(*data[2:])))

    sari_score = corpus_sari(inp_data, hyp_data, ref_data)

    print("SARI = %.2f"%(sari_score*100))
