import sys
sys.path.append('utils/')
import tensorflow as tf
import tensorflow.contrib.seq2seq as seq2seq
from tensorflow.python.layers import core as layers_core
from model_utils import *
from reward import *
import rlloss

class Seq2SeqModel():
    def __init__(self, config):

        print('The model is built for training:', config['IS_TRAIN'])

        self.rl_enable = config['RL_ENABLE']
        self.bleu_enable = config['BLEU_RL_ENABLE']
        self.learning_rate = tf.Variable(config['LR'], dtype=tf.float32, name='model_learning_rate', trainable=False)
        self.word_embedding_learning_rate = tf.Variable(config['WE_LR'], dtype=tf.float32, name='model_we_learning_rate', trainable=False)
        self.encoder_learning_rate = tf.Variable(config['ENCODER_LR'], dtype=tf.float32, name='model_enc_learning_rate', trainable=False)
        self.decoder_learning_rate = tf.Variable(config['DECODER_LR'], dtype=tf.float32, name='model_dec_learning_rate', trainable=False)
        if config['SPLIT_LR']:
            def tmp_func():
                self.word_embedding_learning_rate.assign( self.word_embedding_learning_rate * config['LR_DECAY'])
                self.encoder_learning_rate.assign( self.encoder_learning_rate * config['LR_DECAY'])
                self.decoder_learning_rate.assign( self.decoder_learning_rate * config['LR_DECAY'])
            self.lr_decay_op = tmp_func()
        else:
            self.lr_decay_op = self.learning_rate.assign(self.learning_rate * config['LR_DECAY'])

        if config['OPTIMIZER']=='Adam':
            self.optimizer = tf.train.AdamOptimizer
        elif config['OPTIMIZER']=='GD':
            self.optimizer = tf.train.GradientDescentOptimizer
        else:
            raise Exception("Wrong optimizer name...")

        self.global_step = tf.Variable(config['GLOBAL_STEP'], dtype=tf.int32, name='model_global_step', trainable=False)
        self.batch_size = config['BATCH_SIZE']
        self.input_size = config['INPUT_VOCAB_SIZE']
        self.output_size = config['OUTPUT_VOCAB_SIZE']
        self.encoder_hidden_size = config['ENCODER_HIDDEN_SIZE']
        self.decoder_hidden_size = config['DECODER_HIDDEN_SIZE']
        self.embedding_size = config['WORD_EMBEDDING_SIZE']


        self.encoder_inputs = tf.placeholder(dtype=tf.int32, shape=(None, self.batch_size), name='encoder_inputs')
        self.encoder_inputs_length = tf.placeholder(dtype=tf.int32, shape=(self.batch_size, ), name='encoder_inputs_length')
        self.encoder_inputs_mask = tf.placeholder(dtype=tf.float32, shape=(self.batch_size, None), name='encoder_inputs_mask')
        self.decoder_inputs = tf.placeholder(dtype=tf.int32, shape=(None, self.batch_size), name='decoder_inputs')
        self.decoder_inputs_length = tf.placeholder(dtype=tf.int32, shape=(self.batch_size, ), name='decoder_inputs_length')
        self.decoder_inputs_mask = tf.placeholder(dtype=tf.float32, shape=(self.batch_size, None), name='decoder_inputs_mask')
        self.decoder_targets = tf.placeholder(dtype=tf.int32, shape=(None, self.batch_size), name='decoder_targets')
        self.decoder_targets_length = tf.placeholder(dtype=tf.int32, shape=(self.batch_size, ), name='decoder_targets_length')
        self.decoder_targets_mask = tf.placeholder(dtype=tf.float32, shape=(self.batch_size, None), name='decoder_targets_mask')
        self.rewards = tf.placeholder(dtype=tf.float32, shape=(self.batch_size, None, self.output_size), name='decoder_targets_mask')


        with tf.variable_scope("InputWordEmbedding") as scope:

            self.input_word_embedding_matrix = modelInitWordEmbedding(self.input_size, self.embedding_size, name='we_input')

            self.encoder_inputs_embedded = modelGetWordEmbedding(self.input_word_embedding_matrix, self.encoder_inputs)

            # print('Embedding Trainable Variables')
            self.input_embedding_variables = scope.trainable_variables()
            # print(self.embedding_variables)

        with tf.variable_scope("OutputWordEmbedding") as scope:

            self.output_word_embedding_matrix = modelInitWordEmbedding(self.output_size, self.embedding_size, name='we_output')
            self.decoder_inputs_embedded = modelGetWordEmbedding(self.output_word_embedding_matrix, self.decoder_inputs)

            # print('Embedding Trainable Variables')
            self.output_embedding_variables = scope.trainable_variables()
            # print(self.embedding_variables)

        with tf.variable_scope("DynamicEncoder") as scope:
            self.encoder_cell = modelInitRNNCells(self.encoder_hidden_size, config['ENCODER_LAYERS'], config['CELL'], config['INPUT_DROPOUT'], config['OUTPUT_DROPOUT'])
            if config['BIDIRECTIONAL_ENCODER']:
                if config['VAE_ENCODER']:
                    self.encoder_outputs, self.encoder_state, self.vae_loss = modelInitVAEBidirectionalEncoder(self.encoder_cell, self.encoder_inputs_embedded, self.encoder_inputs_length, encoder_type='stack')
                else:
                    self.encoder_outputs, self.encoder_state = modelInitBidirectionalEncoder(self.encoder_cell, self.encoder_inputs_embedded, self.encoder_inputs_length, encoder_type='stack')
            else:
                if config['VAE_ENCODER']:
                    self.encoder_outputs, self.encoder_state, self.vae_loss = modelInitVAEUndirectionalEncoder(self.encoder_cell, self.encoder_inputs_embedded, self.encoder_inputs_length)
                else:
                    self.encoder_outputs, self.encoder_state = modelInitUndirectionalEncoder(self.encoder_cell, self.encoder_inputs_embedded, self.encoder_inputs_length)

            self.encoder_inputs_length_att = self.encoder_inputs_length

            if config['SAE_ENCODER']:
                # sae_h_size = 20
                # W_SAE = tf.get_variable("W_SAE", shape=[config['MAX_IN_LEN'], sae_h_size])
                # self.encoder_outputs =  tf.transpose(tf.reshape(tf.matmul(tf.reshape(tf.transpose(self.encoder_outputs, [1,2,0]), [-1,config['MAX_IN_LEN']]), W_SAE), [self.batch_size,self.encoder_hidden_size*2,sae_h_size]), [2,0,1])
                # print(self.encoder_outputs)
                # self.encoder_inputs_length_att = tf.convert_to_tensor([sae_h_size]*self.batch_size, dtype=tf.int32)

                o_1, o_2 = tf.split(self.encoder_outputs, 2, 1)
                euclidean_dis = tf.reduce_mean(tf.square(o_1-o_2),2)
                self.sae_loss = tf.reduce_mean(euclidean_dis)

            if config['USE_BS'] and not config['IS_TRAIN']:
                self.encoder_state = seq2seq.tile_batch(self.encoder_state, config['BEAM_WIDTH'])
                self.encoder_outputs = tf.transpose(seq2seq.tile_batch(tf.transpose(self.encoder_outputs, [1,0,2]), config['BEAM_WIDTH']), [1,0,2])
                self.encoder_inputs_length_att = seq2seq.tile_batch(self.encoder_inputs_length_att, config['BEAM_WIDTH'])


            # print('Encoder Trainable Variables')
            self.encoder_variables = scope.trainable_variables()
            # print(self.encoder_variables)


        with tf.variable_scope("DynamicDecoder") as scope:
            self.decoder_cell = modelInitRNNCells(self.decoder_hidden_size, config['DECODER_LAYERS'], config['CELL'], config['INPUT_DROPOUT'], config['OUTPUT_DROPOUT'])
            if config['ATTENTION_DECODER']:
                self.decoder_cell = modelInitAttentionDecoderCell(self.decoder_cell, self.decoder_hidden_size, self.encoder_outputs, self.encoder_inputs_length_att, att_type=config['ATTENTION_MECHANISE'], wrapper_type='whole')
            else:
                self.decoder_cell = modelInitRNNDecoderCell(self.decoder_cell)

            initial_state = None

            if config['USE_BS'] and not config['IS_TRAIN']:
                initial_state = self.decoder_cell.zero_state(batch_size=self.batch_size*config['BEAM_WIDTH'], dtype=tf.float32)
                if config['ATTENTION_DECODER']:
                    cat_state = tuple([self.encoder_state] + list(initial_state.cell_state)[:-1])
                    initial_state.clone(cell_state=cat_state)
                else:
                    initial_state = tuple([self.encoder_state] + list(initial_state[:-1]))
            else:
                initial_state = self.decoder_cell.zero_state(batch_size=self.batch_size, dtype=tf.float32)

                if config['ATTENTION_DECODER']:
                    cat_state = tuple([self.encoder_state] + list(initial_state.cell_state)[:-1])
                    initial_state.clone(cell_state=cat_state)
                else:
                    initial_state = tuple([self.encoder_state] + list(initial_state[:-1]))

            self.output_projection_layer = layers_core.Dense(self.output_size, use_bias=False)
            if config['IS_TRAIN']:
                self.train_outputs = modelInitDecoderForTrain(self.decoder_cell, self.decoder_inputs_embedded, self.decoder_inputs_length, initial_state, self.output_projection_layer)
            if config['USE_BS'] and not config['IS_TRAIN']:
                self.infer_outputs = modelInitDecoderForBSInfer(self.decoder_cell, self.decoder_inputs[0], self.output_word_embedding_matrix, config['BEAM_WIDTH'], config['ID_END'], config['MAX_OUT_LEN'], initial_state, self.output_projection_layer)
            else:
                self.infer_outputs = modelInitDecoderForGreedyInfer(self.decoder_cell, self.decoder_inputs[0], self.output_word_embedding_matrix, config['ID_END'], config['MAX_OUT_LEN'], initial_state, self.output_projection_layer)


            if config['IS_TRAIN']:
                self.train_loss = seq2seq.sequence_loss(logits=self.train_outputs, targets=tf.transpose(self.decoder_targets, perm=[1,0]), weights=self.decoder_targets_mask)
                if config['VAE_ENCODER'] and config['PRE_ENCODER']==None:
                    self.train_loss += 0.01 * self.vae_loss
                if config['SAE_ENCODER'] and config['PRE_ENCODER']==None:
                    self.train_loss += self.sae_loss
                self.rewards = tf.py_func(contentPenalty, [tf.transpose(self.encoder_inputs, perm=[1,0]), self.train_outputs, tf.constant(config['SRC_DICT'], dtype=tf.string), tf.constant(config['DST_DICT'], dtype=tf.string), tf.transpose(self.decoder_targets, perm=[1,0])], tf.float32)
                self.rewards.set_shape(self.train_outputs.get_shape())
                if config['RL_ENABLE']:
                    self.train_loss_rl = tf.constant(0.0)
                else:
                    self.train_loss_rl = tf.constant(0.0)
                self.rewards_bleu = tf.py_func(againstInputPenalty, [tf.transpose(self.encoder_inputs, perm=[1,0]), self.train_outputs, tf.constant(config['SRC_DICT'], dtype=tf.string), tf.constant(config['DST_DICT'], dtype=tf.string)], tf.float32)
                self.rewards_bleu.set_shape(self.train_outputs.get_shape())
                if config['BLEU_RL_ENABLE']:
                    self.train_loss_rl_bleu = rlloss.sequence_loss_rl(logits=self.train_outputs, rewards=self.rewards_bleu, weights=self.decoder_targets_mask)
                else:
                    self.train_loss_rl_bleu = tf.constant(0.0)
                self.eval_loss = seq2seq.sequence_loss(logits=self.train_outputs, targets=tf.transpose(self.decoder_targets, perm=[1,0]), weights=self.decoder_targets_mask)

                if config['TRAIN_ON_EACH_STEP']:
                    self.final_loss = self.train_loss
                    if config['RL_ENABLE']:
                        self.final_loss = self.final_loss + self.train_loss_rl
                    if config['BLEU_RL_ENABLE']:
                        self.final_loss = self.final_loss + self.train_loss_rl_bleu
                else:
                    self.final_loss = self.eval_loss

            # print('Decoder Trainable Variables')
            self.decoder_variables = scope.trainable_variables()
            # print(self.decoder_variables)



        print('All Trainable Variables:')
        if config['PRE_ENCODER']:
            self.all_trainable_variables = list(set(tf.trainable_variables()).difference(set(self.encoder_variables)).difference(set(self.input_embedding_variables)))
        else:
            self.all_trainable_variables = tf.trainable_variables()
        print(self.all_trainable_variables)
        if config['IS_TRAIN']:
            if config['SPLIT_LR']:
                self.train_op = updateBP(self.final_loss, [self.word_embedding_learning_rate, self.encoder_learning_rate, self.decoder_learning_rate], [self.embedding_variables, self.encoder_variables, self.decoder_variables], self.optimizer, norm=config['CLIP_NORM'])
            else:
                self.train_op = updateBP(self.final_loss, [self.learning_rate], [self.all_trainable_variables], self.optimizer, norm=config['CLIP_NORM'])
        self.saver = initSaver(tf.global_variables(), config['MAX_TO_KEEP'])
        self.encoder_saver = initSaver(self.encoder_variables+self.input_embedding_variables, config['MAX_TO_KEEP'])


    def make_feed(self, encoder_inputs, encoder_inputs_length, encoder_inputs_mask, decoder_inputs, decoder_inputs_length, decoder_inputs_mask, decoder_targets, decoder_targets_length, decoder_targets_mask):
        return {
            self.encoder_inputs: encoder_inputs,
            self.encoder_inputs_length: encoder_inputs_length,
            self.encoder_inputs_mask: encoder_inputs_mask,
            self.decoder_inputs: decoder_inputs,
            self.decoder_inputs_length: decoder_inputs_length,
            self.decoder_inputs_mask: decoder_inputs_mask,
            self.decoder_targets: decoder_targets,
            self.decoder_targets_length: decoder_targets_length,
            self.decoder_targets_mask: decoder_targets_mask,
        }
    def train_on_batch(self, session, encoder_inputs, encoder_inputs_length, encoder_inputs_mask, decoder_inputs, decoder_inputs_length, decoder_inputs_mask, decoder_targets, decoder_targets_length, decoder_targets_mask):
        train_feed = self.make_feed(encoder_inputs, encoder_inputs_length, encoder_inputs_mask, decoder_inputs, decoder_inputs_length, decoder_inputs_mask, decoder_targets, decoder_targets_length, decoder_targets_mask)
        if self.rl_enable:
            if self.bleu_enable:
                [_, ce_loss, rl_loss, bleu_loss] = session.run([self.train_op, self.train_loss, self.train_loss_rl, self.train_loss_rl_bleu], train_feed)
                return [ce_loss, rl_loss, bleu_loss]
            else:
                [_, ce_loss, rl_loss] = session.run([self.train_op, self.train_loss, self.train_loss_rl], train_feed)
                return [ce_loss, rl_loss]
        else:
            [_, loss] = session.run([self.train_op, self.final_loss], train_feed)
            return loss

    def eval_on_batch(self, session, encoder_inputs, encoder_inputs_length, encoder_inputs_mask, decoder_inputs, decoder_inputs_length, decoder_inputs_mask, decoder_targets, decoder_targets_length, decoder_targets_mask):
        infer_feed = self.make_feed(encoder_inputs, encoder_inputs_length, encoder_inputs_mask, decoder_inputs, decoder_inputs_length, decoder_inputs_mask, decoder_targets, decoder_targets_length, decoder_targets_mask)
        [loss, outputs] = session.run([self.eval_loss, self.infer_outputs], infer_feed)
        return loss, outputs
    def test_on_batch(self, session, encoder_inputs, encoder_inputs_length, encoder_inputs_mask, decoder_inputs, decoder_inputs_length, decoder_inputs_mask, decoder_targets, decoder_targets_length, decoder_targets_mask):
        infer_feed = self.make_feed(encoder_inputs, encoder_inputs_length, encoder_inputs_mask, decoder_inputs, decoder_inputs_length, decoder_inputs_mask, decoder_targets, decoder_targets_length, decoder_targets_mask)
        [outputs] = session.run([self.infer_outputs], infer_feed)
        return outputs

def instanceOfInitModel(sess, config):
    ret = Seq2SeqModel(config)
    sess.run(tf.global_variables_initializer())
    print('Model Initialized.')
    return ret
