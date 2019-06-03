#coding: UTF-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tensorflow as tf
import numpy as np
from six.moves import range


from data_utils import *
from reward import *
from contrib_rnn_cell import ExtendedMultiRNNCell
from GNMTCell import GNMTAttentionMultiCell


def initSaver(vars_to_save, max_to_keep=20):
    saver = tf.train.Saver(vars_to_save, max_to_keep=max_to_keep)
    return saver

def loadConfigFromFolder(config, pf):
    if pf and os.path.isfile(pf+'/config.json'):
        config = json2load(pf+'/config.json')
    else:
        return None
    return config
def loadModelFromFolder(sess, saver, pf, n):
    ckpt = tf.train.get_checkpoint_state(pf)
    if ckpt!=None:
        p = ckpt.model_checkpoint_path
        if n != -1:
            p = p[:p.rfind('-')+1]+str(n)
        print('Restoring checkpoint @ %s'%(p))
        saver.restore(sess, p)
    print("Restored model from %s"%pf)

def saveModelToFolder(sess, saver, pf, config, n_iter):
    save2json(config, pf+'/config.json')
    saver.save(sess, pf+'/checkpoint', global_step=n_iter)
    print("Model saved at %s"%(pf+'/checkpoint-'+str(n_iter)))




def create_learning_rate_decay_fn(decay_steps=200, decay_rate=0.01, decay_type='natural_exp_decay', start_decay_at=0, stop_decay_at=200000, min_learning_rate=0.00001, staircase=False):

    def decay_fn(learning_rate, global_step):
        decay_type_fn = getattr(tf.train, decay_type)
        decayed_learning_rate = decay_type_fn(learning_rate=learning_rate, global_step=tf.minimum(global_step, stop_decay_at) - start_decay_at, decay_steps=decay_steps, decay_rate=decay_rate, staircase=staircase, name="decayed_learning_rate")
        final_lr = tf.train.piecewise_constant(x=global_step, boundaries=[start_decay_at], values=[learning_rate, decayed_learning_rate])

        if min_learning_rate:
            final_lr = tf.maximum(final_lr, min_learning_rate)
        return final_lr

    return decay_fn

def modelInitWordEmbedding(dict_size, embedding_size, name='word_embedding_matrix'):
    sqrt3 = math.sqrt(3)
    initializer = tf.random_uniform_initializer(-sqrt3, sqrt3)

    word_embedding_matrix = tf.get_variable(name=name, shape=[dict_size, embedding_size], initializer=initializer, dtype=tf.float32)
    return word_embedding_matrix

def modelGetWordEmbedding(word_embedding_matrix, inputs):
    return tf.nn.embedding_lookup(word_embedding_matrix, inputs)

def modelInitRNNCells(hidden_size, layers, cell_type, input_dropout, output_dropout):
    cells = []
    for _ in range(layers):
        config_cell = None
        if cell_type in ['LSTM', 'lstm']:
            config_cell = tf.contrib.rnn.BasicLSTMCell(hidden_size)
        elif cell_type in ['GRU', 'gru']:
            config_cell = tf.contrib.rnn.GRUCell(hidden_size)
        else:
            config_cell = tf.contrib.rnn.BasicRNNCell(hidden_size)
        config_cell = tf.contrib.rnn.DropoutWrapper(config_cell, input_keep_prob=input_dropout, output_keep_prob=output_dropout)
        cells.append(config_cell)
    return cells

def modelInitBidirectionalEncoder(cells, encoder_inputs, inputs_lengths, encoder_type='dynamic', outputs_type='concat', states_type='last'):
    encoder_fw_cell = copy.deepcopy(cells)
    encoder_bw_cell = copy.deepcopy(cells)
    encoder_outputs, encoder_fw_states, encoder_bw_states = None, None, None
    if encoder_type=='dynamic':
        encoder_fw_cell = ExtendedMultiRNNCell(encoder_fw_cell)
        encoder_bw_cell = ExtendedMultiRNNCell(encoder_bw_cell)
        ((encoder_fw_outputs, encoder_bw_outputs), (encoder_fw_states, encoder_bw_states)) =                 tf.nn.bidirectional_dynamic_rnn(cell_fw=encoder_fw_cell, cell_bw=encoder_bw_cell, inputs=encoder_inputs, sequence_length=inputs_lengths, time_major=True, dtype=tf.float32)
        encoder_outputs = tf.concat((encoder_fw_outputs, encoder_bw_outputs), 2)
    elif encoder_type=='stack':
        (encoder_outputs, encoder_fw_states, encoder_bw_states) = tf.contrib.rnn.stack_bidirectional_dynamic_rnn( cells_fw=encoder_fw_cell, cells_bw=encoder_bw_cell, inputs=tf.transpose(encoder_inputs,[1,0,2]), sequence_length=inputs_lengths, dtype=tf.float32)
        encoder_outputs = tf.transpose(encoder_outputs, [1,0,2])
    else:
        raise Exception('Unknown encoder type.')

    encoder_states = None
    if states_type=='last':
        encoder_state_c = (encoder_fw_states[-1].c+encoder_bw_states[-1].c)/2.0
        encoder_state_h = (encoder_fw_states[-1].h+encoder_bw_states[-1].h)/2.0
        encoder_states = tf.contrib.rnn.LSTMStateTuple(c=encoder_state_c, h=encoder_state_h)
    else:
        raise Exception('Unknown states type.')

    return encoder_outputs, encoder_states

def modelInitVAEBidirectionalEncoder(cells, encoder_inputs, inputs_lengths, encoder_type='dynamic', outputs_type='concat', states_type='last'):
    encoder_fw_cell = copy.deepcopy(cells)
    encoder_bw_cell = copy.deepcopy(cells)
    encoder_outputs, encoder_fw_states, encoder_bw_states = None, None, None
    vae_loss = 0
    if encoder_type=='dynamic':
        encoder_fw_cell = ExtendedMultiRNNCell(encoder_fw_cell)
        encoder_bw_cell = ExtendedMultiRNNCell(encoder_bw_cell)
        ((encoder_fw_outputs, encoder_bw_outputs), (encoder_fw_states, encoder_bw_states)) =                 tf.nn.bidirectional_dynamic_rnn(cell_fw=encoder_fw_cell, cell_bw=encoder_bw_cell, inputs=encoder_inputs, sequence_length=inputs_lengths, time_major=True, dtype=tf.float32)
        fwo_m, fwo_v = tf.split(encoder_fw_outputs, 2, 2)
        bwo_m, bwo_v = tf.split(encoder_bw_outputs, 2, 2)
        eo_m = tf.concat([fwo_m, bwo_m],-1)
        eo_v = tf.concat([fwo_v, bwo_v],-1)+1
        encoder_outputs = tf.random_normal(tf.shape(eo_m))*eo_v+eo_m
        vae_loss = -0.5*tf.reduce_mean(tf.reduce_sum(1.0+tf.log(eo_v*eo_v)-eo_m*eo_m-eo_v*eo_v, -1))
    elif encoder_type=='stack':
        (encoder_outputs, encoder_fw_states, encoder_bw_states) = tf.contrib.rnn.stack_bidirectional_dynamic_rnn( cells_fw=encoder_fw_cell, cells_bw=encoder_bw_cell, inputs=tf.transpose(encoder_inputs,[1,0,2]), sequence_length=inputs_lengths, dtype=tf.float32)
        encoder_outputs = tf.transpose(encoder_outputs, [1,0,2])
        encoder_fw_outputs, encoder_bw_outputs = tf.split(encoder_outputs, 2, 2)
        fwo_m, fwo_v = tf.split(encoder_fw_outputs, 2, 2)
        bwo_m, bwo_v = tf.split(encoder_bw_outputs, 2, 2)
        eo_m = tf.concat([fwo_m, bwo_m],-1)
        eo_v = tf.concat([fwo_v, bwo_v],-1)+1
        encoder_outputs = tf.random_normal(tf.shape(eo_m))*eo_v+eo_m
        vae_loss = -0.5*tf.reduce_mean(tf.reduce_sum(1.0+tf.log(eo_v*eo_v)-eo_m*eo_m-eo_v*eo_v, -1))
    else:
        raise Exception('Unknown encoder type.')

    encoder_states = None
    if states_type=='last':
        encoder_state_c = (encoder_fw_states[-1].c+encoder_bw_states[-1].c)/2.0
        encoder_state_h = (encoder_fw_states[-1].h+encoder_bw_states[-1].h)/2.0
        encoder_states = tf.contrib.rnn.LSTMStateTuple(c=encoder_state_c, h=encoder_state_h)
    else:
        raise Exception('Unknown states type.')



    return encoder_outputs, encoder_states, vae_loss
def modelInitVAEUndirectionalEncoder(cells, encoder_inputs, inputs_lengths, outputs_type='concat', states_type='last'):
    encoder_cell = ExtendedMultiRNNCell(cells)
    encoder_outputs, encoder_state = tf.nn.dynamic_rnn(cell=encoder_cell, inputs=encoder_inputs, sequence_length=inputs_lengths, time_major=True, dtype=tf.float32)
    o_m, o_v = tf.split(encoder_outputs, 2, 2)
    o_v = o_v + 1
    encoder_outputs = tf.random_normal(tf.shape(o_m))*o_v+o_m
    vae_loss = -0.5*tf.reduce_mean(tf.reduce_sum(1.0+tf.log(o_v*o_v)-o_m*o_m-o_v*o_v, -1))
    if states_type=='last':
        encoder_state_c = (encoder_fw_states[-1].c+encoder_bw_states[-1].c)/2.0
        encoder_state_h = (encoder_fw_states[-1].h+encoder_bw_states[-1].h)/2.0
        encoder_states = tf.contrib.rnn.LSTMStateTuple(c=encoder_state_c, h=encoder_state_h)
    else:
        raise Exception('Unknown states type.')

    return encoder_outputs, encoder_states, vae_loss

def modelInitUndirectionalEncoder(cells, encoder_inputs, inputs_lengths, outputs_type='concat', states_type='last'):
    encoder_cell = ExtendedMultiRNNCell(cells)
    encoder_outputs, encoder_state = tf.nn.dynamic_rnn(cell=encoder_cell, inputs=encoder_inputs, sequence_length=inputs_lengths, time_major=True, dtype=tf.float32)
    if states_type=='last':
        encoder_state_c = (encoder_fw_states[-1].c+encoder_bw_states[-1].c)/2.0
        encoder_state_h = (encoder_fw_states[-1].h+encoder_bw_states[-1].h)/2.0
        encoder_states = tf.contrib.rnn.LSTMStateTuple(c=encoder_state_c, h=encoder_state_h)
    else:
        raise Exception('Unknown states type.')

    return encoder_outputs, encoder_states

def modelInitAttentionDecoderCell(cells, hidden_size, encoder_outputs, encoder_outputs_lengths, advance=True, att_type='LUONG', wrapper_type='whole'):
    op_cell = None
    if wrapper_type=='whole':
        op_cell = ExtendedMultiRNNCell(cells)
    elif wrapper_type=='gnmt':
        op_cell = cells.pop(0)
    else:
        raise Exception('Unknown wrapper type.')

    attention_mechanism = None
    if att_type=='LUONG':
        attention_mechanism = tf.contrib.seq2seq.LuongAttention(hidden_size, tf.transpose(encoder_outputs,perm=[1,0,2]), memory_sequence_length=encoder_outputs_lengths, scale=advance)
    elif att_type=='BAHDANAU':
        attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(hidden_size, tf.transpose(encoder_outputs,perm=[1,0,2]), memory_sequence_length=encoder_outputs_lengths, normalize=advance)
    else:
        raise Exception('Unknown attention type.')

    op_cell = tf.contrib.seq2seq.AttentionWrapper(op_cell, attention_mechanism, attention_layer_size=hidden_size, output_attention=True)

    if wrapper_type=='gnmt':
        op_cell = GNMTAttentionMultiCell(op_cell, cells)

    return op_cell

def modelInitRNNDecoderCell(cells):
    return ExtendedMultiRNNCell(cells)
def modelInitDecoderCellStates(cell, batch_size, beam_width=1):
    pass

def modelInitDecoderForTrain(cell, inputs, inputs_lengths, initial_state,  output_projection_layer):
    train_helper = tf.contrib.seq2seq.TrainingHelper(inputs, inputs_lengths, time_major=True)
    decoder=tf.contrib.seq2seq.BasicDecoder(cell, helper=train_helper, initial_state=initial_state, output_layer=output_projection_layer)
    outputs, states, sequence_lengths = tf.contrib.seq2seq.dynamic_decode(decoder, impute_finished=True, maximum_iterations=None)
    return outputs.rnn_output

def modelInitDecoderForBlindTrain(cell, inputs, inputs_lengths,  embeddings, initial_state,  output_projection_layer):
    train_helper = tf.contrib.seq2seq.ScheduledEmbeddingTrainingHelper(inputs, inputs_lengths, embeddings, 1.0, time_major=True)
    decoder=tf.contrib.seq2seq.BasicDecoder(cell, helper=train_helper, initial_state=initial_state, output_layer=output_projection_layer)
    outputs, states, sequence_lengths = tf.contrib.seq2seq.dynamic_decode(decoder, impute_finished=True, maximum_iterations=None)
    return outputs.rnn_output

def modelInitDecoderForGreedyInfer(cell, inputs, word_embedding_matrix, id_end, max_len, initial_state, output_projection_layer):
    infer_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(word_embedding_matrix, inputs, id_end)
    decoder=tf.contrib.seq2seq.BasicDecoder(cell, helper=infer_helper, initial_state=initial_state, output_layer=output_projection_layer)
    outputs, states, sequence_lengths = tf.contrib.seq2seq.dynamic_decode(decoder, impute_finished=False, maximum_iterations=max_len*2)
    return outputs.sample_id

def modelInitDecoderForBSInfer(cell, inputs, word_embedding_matrix, beam_width, id_end, max_len, initial_state, output_projection_layer):
    decoder=tf.contrib.seq2seq.BeamSearchDecoder(cell, word_embedding_matrix, inputs, id_end, initial_state=initial_state, beam_width=beam_width, output_layer=output_projection_layer)
    outputs, states, sequence_lengths = tf.contrib.seq2seq.dynamic_decode(decoder, impute_finished=False, maximum_iterations=max_len*2)
    return tf.transpose(outputs.predicted_ids, [2,0,1])[0]

def updateBP(loss, lr, var_list, optimizer, norm=None):
    gradients = [tf.gradients(loss, var_list[i]) for i in range(len(lr))]
    if norm!=None:
        gradients = [tf.clip_by_global_norm(gradients[i], norm)[0] for i in range(len(lr))]
    optimizers = [optimizer(lr[i]) for i in range(len(lr))]
    return [optimizers[i].apply_gradients(zip(gradients[i], var_list[i])) for i in range(len(lr))]
