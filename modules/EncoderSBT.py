#!/usr/bin/env python
#-*- coding:utf-8 -*-
# author: Zhen YANG
# created at:20/12/2020 7:22 PM
# contact: zhyang8-c@my.cityu.edu.hk

import tensorflow as tf
from modules.TransformerUtils import TransformerUtils as utils
from modules.EncoderLayer import EncoderLayer


class EncoderSBT(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
               maximum_position_encoding, rate=0.1):
        super(EncoderSBT, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        self.pos_encoding = utils.positional_encoding(maximum_position_encoding,
                                                self.d_model)

        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mha_mask):

        seq_len = tf.shape(x)[1]

        # 将嵌入和位置编码相加。

        x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        x = x * tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        # only encoding to the seq_len
        x += self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
          x, mha_attn = self.enc_layers[i](x, training, mha_mask)
        return x, mha_attn  # (batch_size, input_seq_len, d_model)
