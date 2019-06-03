import sys
sys.path.append('utils/')
from data_utils import *


import argparse
parser = argparse.ArgumentParser(
    description="Train a seq2seq model and save in the specified folder.")

parser.add_argument(
    "-f",
    dest="ori_file",
    type=str)

parser.add_argument(
    "-r",
    dest="ref_file",
    type=str)

parser.add_argument(
    "-i",
    dest="inp_file",
    type=str)

args = parser.parse_args()

f = codecs.open(args.ori_file, 'r', 'utf-8')
lines = f.readlines()
f.close()

f = codecs.open(args.inp_file, 'r', 'utf-8')
inp_lines = f.readlines()
f.close()
findc, totalc = 0,0
for i in range(len(lines)):
    s = ' <bos> ' + lines[i] + ' <eos> '
    r = ' <bos> ' + inp_lines[i] + ' <eos> '
    while(s.find(' <UNK>')!=-1):
        ns = len(s)
        st = s.find(' <UNK>')
        if st == -1:
            break
        en = st+6
        ucnt = 1
        while en+6<=ns and s[en:en+6]==' <UNK>':
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
            s = s[:st]+s[st:en].replace('<UNK>', '<unk>')+s[en:]
        totalc += 1
    lines[i] = s[7:-7].replace('<UNK>', '').replace('<unk>', '')
    if len(lines[i].strip())==0:
        lines[i]='.\n'
    print(i, lines[i])
print('Rep/Total=%d/%d'%(findc, totalc))

f = codecs.open(args.ori_file, 'w', 'utf-8')
f.writelines(lines)
f.close()

command_s = command_s = './data/lang8/m2scorer/scripts/m2scorer.py %s %s'%(args.ori_file, args.ref_file)

print(os.popen(command_s).readlines())
