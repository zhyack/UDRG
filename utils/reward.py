from data_utils import *
from eutils import *
import os

import math
def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))
def tanh(x):
    return 2.0*sigmoid(2.0*x)-1

def runMayGetValue(command_s, outputParser = None):
    outputs = os.popen(command_s)
    if outputParser==None:
        try:
            return float(outputs[0].strip())
        except TypeError,ValueError:
            return None
    return outputParser(outputs)

def bleuPerlParser(sl):
    ret = None
    for s in sl:
        if s.find('BLEU')!=-1 and s.find('BP')!=-1 and s.find('ratio')!=-1 and s.find('hyp_len')!=-1 and s.find('ref_len')!=-1:
            st = s.find('=')+1
            en = s.find(',', st)
            ret = float(s[st:en])
            break
    return ret
def m2scorerParser(sl):
    ret = []
    print(sl)
    for s in sl:
        if s.find('Precision')!=-1:
            p = s.find(':', s.find('Precision'))
            ret.append(float(s[p+2:-1]))
        if s.find('Recall')!=-1:
            p = s.find(':', s.find('Recall'))
            ret.append(float(s[p+2:-1]))
        if s.find('F_0.5')!=-1:
            p = s.find(':', s.find('F_0.5'))
            ret.append(float(s[p+2:-1]))
    return ret


def precisionInstance(s='tmp/predictions.txt'):
    f1 = open(s, 'r')
    f2 = open('tmp/predictions.txt')
    l1 = f1.readlines()
    l2 = f2.readlines()
    f1.close()
    f2.close()
    p = 0
    for i in range(len(l1)):
        if l1[i].strip()==l2[i].strip():
            p += 1
    return float(p)/float(len(l1))

def bleuNull(s=''):
    return 0.0

def m2scorerInstance(ss='tmp/predictions.txt'):

    f = codecs.open('tmp/predictions.txt', 'r', 'utf-8')
    lines = f.readlines()
    f.close()
    f = codecs.open('data/lang8/test.in', 'r', 'utf-8')
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
    print('Rep/Total=%d/%d'%(findc, totalc))
    f = codecs.open('tmp/predictions.txt', 'w', 'utf-8')
    f.writelines(lines)
    f.close()


    command_s = './data/lang8/m2scorer/scripts/m2scorer.py tmp/predictions.txt %s'%(ss)
    p = m2scorerParser
    return runMayGetValue(command_s, p)

def bleuPerlInstanceSingle(s='tmp/predictions.txt'):
    command_s = 'python utils/multi_bleu.py -hyp tmp/predictions.txt -ref %s'%(s)
    p = bleuPerlParser
    return runMayGetValue(command_s, p)

def bleuPerlInstanceWikiLarge(s='tmp/predictions.txt'):
    s = 'data/wiki/wikilarge/wiki.full.aner.ori.test.dst'
    for i in range(8):
        s += ' data/wiki/wikilarge/refs/test.8turkers.tok.turk.%d'%(i)
    command_s = 'python utils/multi_bleu.py -hyp tmp/predictions.txt -ref %s'%(s)
    p = bleuPerlParser
    return runMayGetValue(command_s, p)

# python utils/multi_bleu.py -ref data/wiki/wikilarge/wiki.full.aner.ori.test.dst  data/wiki/wikilarge/refs/test.8turkers.tok.turk.0  data/wiki/wikilarge/refs/test.8turkers.tok.turk.1  data/wiki/wikilarge/refs/test.8turkers.tok.turk.2  data/wiki/wikilarge/refs/test.8turkers.tok.turk.3  data/wiki/wikilarge/refs/test.8turkers.tok.turk.4  data/wiki/wikilarge/refs/test.8turkers.tok.turk.5  data/wiki/wikilarge/refs/test.8turkers.tok.turk.6  data/wiki/wikilarge/refs/test.8turkers.tok.turk.7  -hyp xxx/predictions

def bleuPerlInstanceWikiSmallRelex(s='tmp/predictions.txt'):
    f = codecs.open('tmp/predictions.txt', 'r', 'utf-8')
    lines = f.readlines()
    f.close()
    m = json2load('data/wiki/wikismall/map.json')['test']
    for i in range(len(lines)):
        for k in m[str(i+1)]:
            lines[i] = lines[i].replace(k, m[str(i+1)][k])

    f = codecs.open('data/wiki/wikismall/PWKP_108016.tag.80.aner.ori.test.src', 'r', 'utf-8')
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
                s = s[:st]+s[st:en].lower()+s[en:]
            totalc += 1
        lines[i] = s[7:-7]
    print('Rep/Total=%d/%d'%(findc, totalc))
    f = codecs.open('tmp/predictions.txt', 'w', 'utf-8')
    f.writelines(lines)
    f.close()

    s = 'data/wiki/wikismall/PWKP_108016.tag.80.aner.ori.test.dst'
    command_s = 'python utils/multi_bleu.py -hyp tmp/predictions.txt -ref %s'%(s)
    p = bleuPerlParser
    return runMayGetValue(command_s, p)

def bleuPerlInstanceWikiLargeRelex(s='tmp/predictions.txt'):
    f = codecs.open('tmp/predictions.txt', 'r', 'utf-8')
    lines = f.readlines()
    f.close()
    m = json2load('data/wiki/wikilarge/map.json')['test']
    for i in range(len(lines)):
        for k in m[str(i+1)]:
            lines[i] = lines[i].replace(k, m[str(i+1)][k])
        lines[i] = lines[i].lower()

    f = codecs.open('data/wiki/wikilarge/wiki.full.aner.ori.test.src', 'r', 'utf-8')
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
    f = codecs.open('tmp/predictions.txt', 'w', 'utf-8')
    f.writelines(lines)
    f.close()

    s = 'data/wiki/wikilarge/wiki.full.aner.ori.test.dst'
    for i in range(8):
        s += ' data/wiki/wikilarge/refs/test.8turkers.tok.turk.%d'%(i)
    command_s = 'python utils/multi_bleu.py -hyp tmp/predictions.txt -ref %s'%(s)
    p = bleuPerlParser
    return runMayGetValue(command_s, p)




global lm_set, dict_set
lm_set = {}
dict_set = {}

def initLM(pm, pd):
    dict_dst, _ = loadDict(pd)
    f = codecs.open(pm, 'r', 'UTF-8')
    lines = f.readlines()
    f.close()
    ret = dict()
    for l in lines:
        l = l.split()
        if len(l)<3 or l[0]=='ngram':
            continue
        try:
            float(l[-1])
        except ValueError:
            l.append('0.0')
        for i in range(1, len(l)-1):
            if l[i] == '<s>':
                l[i] = '<BOS>'
            elif l[i] == '</s>':
                l[i] = '<EOS>'
            elif l[i] not in dict_dst:
                l[i] = '<UNK>'
            l[i] = str(dict_dst[l[i]])
        k = ' '.join(l[1:-1])
        if k not in ret:
            ret[k]=dict()
            ret[k]['p'] = float(l[0])
            ret[k]['bp'] = float(l[-1])
            # ret[k]['p'] = -1000000.0
            # ret[k]['bp'] = -1000000.0
        # ret[k]['p']=min(math.log(math.exp(float(l[0]))+math.exp(ret[k]['p'])), -0.0001)
        # ret[k]['bp']=min(math.log(math.exp(float(l[0]))+math.exp(ret[k]['p'])), -0.0001)

    return ret, dict_dst

def getBP(k,lm):
    if k in lm:
        return lm[k]['bp']
    else:
        return -3.0

def getP(k, lm):
    if k in lm:
        return lm[k]['p']
    else:
        if k.find(' ')==-1:
            return getBP(k,lm)
        else:
            return getBP(k[:k.rfind(' ')],lm)+getP(k[k.find(' ')+1:],lm)


def LMScore(outputs, plm, pdict):
    global lm_set, dict_set
    if plm not in lm_set:
        lm_set[plm], dict_set[pdict] = initLM(plm, pdict)
    lm = lm_set[plm]
    dict_dst = dict_set[pdict]
    max_len = outputs.shape[1]
    batch_size = len(outputs)
    ret = np.zeros([batch_size,max_len,len(dict_dst)], np.float32)
    for i in range(batch_size):
        predictions = outputs[i].argmax(axis=-1)
        for j in range(max_len):
            p = int(predictions[j])
            k = ' '.join([str(w) for w in predictions[max(0,j-2):j+1]])
            if j>1:
                reward=(tanh(getP(k,lm)/5.0)+0.5)/5.0
                ret[i][j][p]+=reward
                ret[i][j-1][int(predictions[j-1])]+=reward
                ret[i][j-2][int(predictions[j-2])]+=reward
            # bk = ' '.join([str(w) for w in predictions[j:min(j+3,batch_size)]])
            # if j>1 and j<batch_size-2:
            #     ret[i][j][p]=((tanh(getP(k)/5.0)+0.5)/5.0 + (tanh(getP(bk)/5.0)+0.5)/5.0) / 2.0
            # elif j>1:
            #     ret[i][j][p]=(tanh(getP(k)/5.0)+0.5)/5.0
            # elif j<batch_size-2:
            #     ret[i][j][p]=(tanh(getP(bk)/5.0)+0.5)/5.0
            # if j>1:
            #     ret[i][j][p]=(tanh(getP(k)/5.0)+0.5)/10.0
        for j in range(max_len):
            ret[i][j][int(predictions[j])]/=float(min(3,j+1))
        # print([ret[i][j][int(predictions[j])] for j in range(max_len)])
    return ret








def contentPenalty(inputs, outputs, SRC_DICT, DST_DICT, targets):
    pass
def againstInputPenalty(inputs, outputs, SRC_DICT, DST_DICT):
    pass

# global dict_src, rev_dict_src, dict_dst, rev_dict_dst
# dict_src, rev_dict_src, dict_dst, rev_dict_dst = None, None, None, None
#
# def contentPenalty(inputs, outputs, SRC_DICT, DST_DICT, targets):
#     # print(inputs.shape)
#     # print(outputs.shape)
#     global dict_src, rev_dict_src, dict_dst, rev_dict_dst
#     if dict_dst==None or dict_dst==None or rev_dict_dst==None or rev_dict_src==None:
#         dict_src, rev_dict_src = loadDict(SRC_DICT)
#         dict_dst, rev_dict_dst = loadDict(DST_DICT)
#     batch_size = len(inputs)
#     assert(batch_size == len(outputs))
#     max_len = outputs.shape[1]
#     ret = []
#     all_keys = [False]*len(dict_dst)
#     for ind_src in rev_dict_src:
#         word = rev_dict_src[ind_src]
#         if word.upper()==word and len(word)>2 and word!='<EOS>' and dict_dst.has_key(word):
#             ind_dst = dict_dst[word]
#             all_keys[ind_dst]=True
#     for i in range(batch_size):
#         # sb = copy.deepcopy(score_board)
#         expect_eos = np.argwhere(targets[i]==dict_dst['<EOS>'])[0]
#         poskey_cnt = 0
#         pos_keys = [0]*len(dict_dst)
#         for ind in inputs[i]:
#             ind_src = int(ind)
#             word = rev_dict_src[ind_src]
#             if word.upper()==word and len(word)>2 and word!='<PAD>' and word!='<EOS>' and dict_dst.has_key(word):
#                 ind_dst = dict_dst[word]
#                 if pos_keys[ind_dst]!=2:
#                     poskey_cnt += 1
#                 pos_keys[ind_dst]=2
#         ret.append([])
#         predictions = outputs[i].argmax(axis=-1)
#         eos = False
#         for j in range(max_len):
#             score_board = [0.0]*len(dict_dst)
#             if eos:
#                 ret[i].append(score_board)
#                 continue
#             p = int(predictions[j])
#             if (all_keys[p] and pos_keys[p]==0):
#                 score_board[p] = 0.2
#             elif (all_keys[p] and pos_keys[p]==2):
#                 score_board[p] = 2.0
#                 pos_keys[p] -= 1
#             elif (all_keys[p] and pos_keys[p]==1):
#                 score_board[p] = 1.0
#                 pos_keys[p] -= 1
#             else:
#                 score_board[p] = 1.0
#
#             if j>0 and predictions[j]==predictions[j-1]:
#                 score_board[p] -= 1.0
#             ret[i].append(score_board)
#         remain_poskey_cnt = 0
#         for k in range(len(dict_dst)):
#             if pos_keys[k] == 2:
#                 remain_poskey_cnt += 1
#         remain_penalty = remain_poskey_cnt*-0.5/poskey_cnt
#         for j in range(max_len):
#             p = int(predictions[j])
#             if ret[i][j][p] != 2.0:
#                 ret[i][j][p] += remain_penalty
#                 ret[i][j][p] = max(0.0,ret[i][j][p])
#     return np.array(ret, dtype=np.float32)
#
# ref_dict = None
# import bleu
#
#
# def againstInputPenalty(inputs, outputs, SRC_DICT, DST_DICT):
#     global dict_src, rev_dict_src, dict_dst, rev_dict_dst
#     if dict_dst==None or dict_dst==None or rev_dict_dst==None or rev_dict_src==None:
#         dict_src, rev_dict_src = loadDict(SRC_DICT)
#         dict_dst, rev_dict_dst = loadDict(DST_DICT)
#     batch_size = len(inputs)
#     max_len = outputs.shape[1]
#     ret = []
#     for i in range(batch_size):
#         # print('batch%d'%(i))
#         ret.append([])
#         predictions = outputs[i].argmax(axis=-1)
#         src = ' '.join([rev_dict_src[k].decode('UTF-8') for k in inputs[i]])
#         if src.find('<EOS>')!=-1:
#             src = src[5:src.find('<EOS>')]
#         else:
#             src = src[5:]
#         hyp = ' '.join([rev_dict_dst[p].decode('UTF-8') for p in predictions])
#         # print('got hyp and refs')
#         bleu_scores = bleu.incremental_sent_bleu(hyp,[src])
#         # print('got bleu')
#         # print(bleu_scores)
#         last_bleu = 0.0
#         for j in range(max_len):
#             bleu_score = bleu_scores[j]
#             p = int(predictions[j])
#             # print(outputs[i][j][p], outputs[i][j][0])
#             score_board = [0.0]*len(dict_dst)
#             score_board[p] = tanh((last_bleu-bleu_score)*100.0+0.5)*1.5
#             # print(15/max_len, score_board[p])
#             ret[i].append(score_board)
#             last_bleu=bleu_score
#     return np.array(ret, dtype=np.float32)
#
# def sigmoid(x):
#     return 1.0 / (1.0 + math.exp(-x))
# def bleuPenalty(inputs, outputs, SRC_DICT, DST_DICT, HYP_FILE_PATH, REF_FILE_PATH_FORMAT):
#     global dict_src, rev_dict_src, dict_dst, rev_dict_dst
#     if dict_dst==None or dict_dst==None or rev_dict_dst==None or rev_dict_src==None:
#         dict_src, rev_dict_src = loadDict(SRC_DICT)
#         dict_dst, rev_dict_dst = loadDict(DST_DICT)
#     global ref_dict
#     if ref_dict == None:
#         print('Loading references...')
#         ref_dict = dict()
#         hyp_list = []
#         linecnt = 0
#         f = open(HYP_FILE_PATH, 'r')
#         for line in f.readlines():
#             line = _2uni(line.strip())
#             line = line.split()
#             if len(line)>48:
#                 line = line[:48]
#                 line = ' '.join(line)
#                 line = line.replace('_',' ').replace(' ','')
#                 line = line[:150]
#             else:
#                 line = ' '.join(line)
#                 line = line.replace('_',' ').replace(' ','')
#                 line = line[:150]
#             ref_dict[line]=[]
#             hyp_list.append(line)
#             linecnt += 1
#         f.close()
#         ref_file_cnt = 0
#         while(True):
#             if os.path.isfile(REF_FILE_PATH_FORMAT%(ref_file_cnt)):
#                 f = open(REF_FILE_PATH_FORMAT%(ref_file_cnt), 'r')
#                 linecnt = 0
#                 for line in f.readlines():
#                     line = _2uni(line.strip())
#                     if len(line)>0:
#                         ref_dict[hyp_list[linecnt]].append(line)
#                     linecnt += 1
#                 ref_file_cnt+=1
#             else:
#                 break
#
#     batch_size = len(inputs)
#     max_len = outputs.shape[1]
#     ret = []
#     for i in range(batch_size):
#         ret.append([])
#         predictions = outputs[i].argmax(axis=-1)
#         last_bleu = 0.0
#         src = ' '.join([_2uni(rev_dict_src[k]) for k in inputs[i]])
#         src = src.replace('_',' ').replace(' ','')
#         if src.find('<EOS>')!=-1:
#             src = src[5:src.find('<EOS>')]
#         else:
#             src = src[5:]
#         if len(src) > 150:
#             src = src[:150]
#         hyp = ' '.join([_2uni(rev_dict_dst[p]) for p in predictions])
#         try:
#             bleu_scores = bleu.incremental_sent_bleu(hyp,ref_dict[src])
#         except KeyError:
#             print(len(src))
#         for j in range(max_len):
#             bleu_score = bleu_scores[j]
#             p = int(predictions[j])
#             score_board = [0.0]*len(dict_dst)
#             score_board[p] = sigmoid(last_bleu-bleu_scores[0])
#             ret[i].append(score_board)
#     return np.array(ret, dtype=np.float32)
