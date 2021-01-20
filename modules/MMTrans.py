#!/usr/bin/env python
#-*- coding:utf-8 -*-
# author: Zhen YANG
# created at:20/12/2020 8:01 PM
# contact: zhyang8-c@my.cityu.edu.hk

import tensorflow as tf
from modules.EncoderSBT import EncoderSBT
from modules.Decoder import Decoder
from modules.EncoderGraph import EncoderGraph

class MMTrans(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, srcs_vocab_size, graphs_vocab_size, asthop,
               comms_vocab_size, pe_srcs, pe_graphs, pe_comms, rate):
        super(MMTrans, self).__init__()
        self.encoder_sbt = EncoderSBT(num_layers, d_model, num_heads, dff,
                               srcs_vocab_size, pe_srcs, rate)
        self.encoder_graph = EncoderGraph(num_layers, d_model, num_heads, dff,
                               graphs_vocab_size, asthop, pe_graphs, rate)
        self.decoder = Decoder(num_layers, d_model, num_heads, dff,
                               comms_vocab_size, pe_comms, rate)
        self.final_layer = tf.keras.layers.Dense(comms_vocab_size)

    def call(self, sbt_inp, node_inp, edge_inp, comm_tar, training, sbt_padding_mask, node_padding_mask, look_ahead_mask):

        sbt_output, sbt_attn = self.encoder_sbt(sbt_inp, training, sbt_padding_mask)  # (batch_size, inp_seq_len, d_model)
        graph_output, graph_attn = self.encoder_graph(node_inp, edge_inp, training, node_padding_mask)
        # dec_output.shape == (batch_size, tar_seq_len, d_model)

        dec_output, attention_weights = self.decoder(comm_tar, sbt_output, graph_output, training,
                                                     sbt_padding_mask, node_padding_mask, look_ahead_mask)

        final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)

        return final_output, attention_weights, sbt_attn, graph_attn
