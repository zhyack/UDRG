import sys
sys.path.append('utils/')
from cseq2seq_model import *
from model_utils import *

import argparse
parser = argparse.ArgumentParser(
    description="Predict with a saved model in the specified folder.")

parser.add_argument(
    "-l",
    dest="load_folder",
    type=str,
    help="The specified folder to load saved model. If not specified, the model will be initialized.")
parser.add_argument(
    "-f",
    dest="input_file",
    type=str,
    help="The specified input file to make predictions.")
parser.add_argument(
    "-n",
    dest="model_index",
    type=int,
    default=-1,
    help=".")
parser.add_argument(
    "-b",
    dest="beam_width",
    type=int,
    default=1,
    help=".")
args = parser.parse_args()

if not os.path.isdir(args.load_folder):
    raise Exception("You should use --l to add the saved model's folder path. (or maybe you gave a wrong path)")
if not os.path.isfile(args.input_file):
    raise Exception("You should use --f to add the input file path. (or maybe you gave a wrong path)")

CONFIG = dict()
CONFIG['MAX_TO_KEEP']=20
CONFIG['PRE_ENCODER']=None
may_have_config = loadConfigFromFolder(None, args.load_folder)
if may_have_config:
    preset_config = copy.deepcopy(may_have_config)
    for k in preset_config:
        CONFIG[k] = preset_config[k]
if args.beam_width > 1:
    CONFIG['USE_BS']=True
    CONFIG['BEAM_WIDTH']=args.beam_width
full_dict_src, rev_dict_src = loadDict(CONFIG['SRC_DICT'])
full_dict_dst, rev_dict_dst = loadDict(CONFIG['DST_DICT'])
print(CONFIG['INPUT_VOCAB_SIZE'],CONFIG['OUTPUT_VOCAB_SIZE'])
CONFIG['ID_END']=full_dict_dst['<EOS>']
CONFIG['ID_BOS']=full_dict_dst['<BOS>']
CONFIG['ID_PAD']=full_dict_dst['<PAD>']
CONFIG['ID_UNK']=full_dict_dst['<UNK>']

f_x = codecs.open(args.input_file,'r','UTF-8')
x_test_raw = f_x.readlines()
test_raw = [ [x_test_raw[i].strip(),''] for i in range(len(x_test_raw))]
test_buckets_raw = arrangeBuckets(test_raw, CONFIG['BUCKETS'])
print([len(b) for b in test_buckets_raw])
f_x.close()

g = tf.Graph()
with g.as_default():
    tf.set_random_seed(CONFIG['SEED'])
    random.seed(CONFIG['SEED'])
    with tf.Session() as sess:
        print('Loading model...')
        print('Result in Training: %.6f'%(CONFIG['LOG'][args.model_index]))
        CONFIG['IS_TRAIN'] = False
        CONFIG['INPUT_DROPOUT'] = 1.0
        CONFIG['OUTPUT_DROPOUT'] = 1.0
        Model = instanceOfInitModel(sess, CONFIG)
        loadModelFromFolder(sess, Model.saver, args.load_folder, args.model_index)

        test_results=dict()
        for b in range(len(CONFIG['BUCKETS'])):
            n_b = len(test_buckets_raw[b])
            for k in range((n_b+CONFIG['BATCH_SIZE']-1)/CONFIG['BATCH_SIZE']):
                test_batch = [ test_buckets_raw[b][i%n_b] for i in range(k*CONFIG['BATCH_SIZE'], (k+1)*CONFIG['BATCH_SIZE']) ]
                print('test process: [%d/%d] [%d/%d]'%(b+1, len(CONFIG['BUCKETS']), k*CONFIG['BATCH_SIZE'], n_b))
                test_batch = map(list, zip(*test_batch))
                model_inputs, len_inputs, inputs_mask = dataSeqs2NpSeqs(test_batch[0], full_dict_src, CONFIG['BUCKETS'][b][0])
                model_outputs, len_outputs, outputs_mask = dataSeqs2NpSeqs(test_batch[1], full_dict_dst, CONFIG['BUCKETS'][b][1])
                model_targets, len_targets, targets_mask = dataSeqs2NpSeqs(test_batch[1], full_dict_dst, CONFIG['BUCKETS'][b][1], bias=1)
                predict_outputs = Model.test_on_batch(sess, model_inputs, len_inputs, inputs_mask, model_outputs, len_outputs, outputs_mask, model_targets, len_targets, targets_mask)

                test_batch = map(list, zip(*test_batch))
                for i in range(CONFIG['BATCH_SIZE']):
                    test_results[test_batch[i][0]] = dataLogits2Seq(predict_outputs[i], rev_dict_dst, calc_argmax=False)
        f_x = codecs.open(args.input_file,'r','UTF-8')
        fname = '/predictions_%d_%.2f.txt'%(args.model_index, CONFIG['LOG'][args.model_index])
        f_y = codecs.open(args.load_folder+fname,'w','UTF-8')
        for line in f_x.readlines():
            s = test_results[line.strip()]
            # s = s.replace('<UNK> ', '')
            p = s.find('<EOS>')
            if p==-1:
                p = len(s)
            f_y.write(s[:p]+'\n')
        f_x.close()
        f_y.close()
        print('Prediction completed! Please check the results @ %s'%(args.load_folder+fname))
