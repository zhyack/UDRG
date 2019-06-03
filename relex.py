import sys
sys.path.append('utils/')
from data_utils import *


import argparse
parser = argparse.ArgumentParser(
    description="Train a seq2seq model and save in the specified folder.")
parser.add_argument(
    "-m",
    dest="map_file",
    type=str)
parser.add_argument(
    "-f",
    dest="ori_file",
    type=str)

parser.add_argument(
    "-r",
    nargs='+',
    dest="ref_files",
    type=str)

parser.add_argument(
    "-i",
    dest="inp_file",
    type=str)

args = parser.parse_args()

f = codecs.open(args.ori_file, 'r', 'utf-8')
lines = f.readlines()
f.close()
m = json2load(args.map_file)['test']
for i in range(len(lines)):
    for k in m[str(i+1)]:
        lines[i] = lines[i].replace(k, m[str(i+1)][k])
    lines[i] = lines[i].lower()

f = codecs.open(args.inp_file, 'r', 'utf-8')
inp_lines = f.readlines()
f.close()
findc, totalc = 0,0
for i in range(len(lines)):
    s = ' <bos> ' + lines[i] + ' <eos> '
    r = ' <bos> ' + inp_lines[i] + ' <eos> '
    while(s.find(' <unk>')!=-1):
        ns = len(s)
        st = s.find(' <unk>')
        if st == -1:
            break
        en = st+6
        ucnt = 1
        while en+6<=ns and s[en:en+6]==' <unk>':
            en = en+6
            ucnt += 1
        prev = s[s.rfind(' ', 0,st-1):st]
        nxt = s[en:s.find(' ', en+1)]
        rst, ren = 0,0
        bfind = False
        while(rst != -1):
            rst = r.find(prev, rst)
            if rst == -1:
                break
            rst += len(prev)
            ren = rst
            tcnt = ucnt
            while (tcnt>0):
                ren = r.find(' ', ren+1)
                if ren == -1:
                    break
                tcnt -= 1
            if ren != -1 and r[ren: ren+len(nxt)]==nxt:
                s = s[:st]+r[rst:ren]+s[en:]
                bfind = True
                break
        if bfind:
            findc += 1
        else:
            s = s[:st]+s[st:en].upper()+s[en:]
        totalc += 1
    lines[i] = s[7:-7]
print('Rep/Total=%d/%d'%(findc, totalc))



f = codecs.open(args.ori_file, 'w', 'utf-8')
f.writelines(lines)
f.close()

command_s = 'python utils/multi_bleu.py -hyp %s -ref %s'%(args.ori_file, ' '.join(args.ref_files))

print(os.popen(command_s).readlines())
