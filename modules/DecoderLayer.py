#!/usr/bin/env python
#-*- coding:utf-8 -*-
# author: Zhen YANG
# created at:17/11/2020 7:52 PM
# contact: zhyang8-c@my.cityu.edu.hk

import tensorflow as tf
from modules.MultiHeadAttention import MultiHeadAttention
from modules.TransformerUtils import TransformerUtils as utils


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()
        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)

        self.ffn = utils.point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)


    def call(self, mha_x, enc_output, training, look_ahead_mask, enc_padding_mask):
        # enc_output.shape == (batch_size, input_seq_len, d_model)
        # self-attention in the decoder input sentence
        attn1, attn_weights_block1 = self.mha1(mha_x, mha_x, mha_x, look_ahead_mask)  # (batch_size, target_seq_len, d_model)
        attn1 = self.dropout1(attn1, training=training)
        out1_mha = self.layernorm1(attn1 + mha_x)

        # attention to encoder output are applied on the out1
        attn2, attn_weights_block2 = self.mha2(enc_output, enc_output, out1_mha, enc_padding_mask)  # (batch_size, target_seq_len, d_model)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1_mha)  # (batch_size, target_seq_len, d_model)

        ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)

        return out3, attn_weights_block1, attn_weights_block2
