# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Seq2seq loss operations for use in sequence models.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

def sequence_loss_rl(logits, rewards, weights):
    """
    logits: outputs of decoder, [batch_size * max_len * dict_size]
    rewards: reward vector for each step, [batch_size * max_len * dict_size]
    weights: mask, [batch_size * max_len]
    return: float, the loss
    """

    ret = 0.0
    logp = tf.clip_by_value(tf.log(tf.nn.softmax(logits)), -20.0, 0.0)
    scores = tf.reduce_sum(tf.multiply(logp, rewards), 2)
    ret = -tf.reduce_mean(tf.multiply(scores, weights))
    return ret
