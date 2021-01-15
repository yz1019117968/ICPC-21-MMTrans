#!/usr/bin/env python
#-*- coding:utf-8 -*-
# author: Zhen YANG
# created at:4/12/2020 8:56 PM
# contact: zhyang8-c@my.cityu.edu.hk

import tensorflow as tf
from tensorflow.keras import activations

class GCNLayer(tf.keras.layers.Layer):
    def __init__(self, units, activation='relu', initializer='glorot_uniform', sparse=False, use_bias=True, **kwargs):
        self.activation = activations.get(activation)
        self.output_dim = units
        self.initializer = initializer
        self.sparse = sparse
        self.use_bias = use_bias

        super(GCNLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                                          shape=(input_shape[0][-1], self.output_dim),
                                          initializer=self.initializer,
                                          trainable=True)
        if self.use_bias:
            self.bias = self.add_weight(name='bias',
                                              shape=(self.output_dim,),
                                              initializer='zeros',
                                              trainable=True)
        else:
            self.bias = None

        super(GCNLayer, self).build(input_shape)

    def call(self, x):
        assert isinstance(x, list)
        # # Get shapes of our inputs and weights
        nodes, edges = x
        nodes = tf.cast(nodes, tf.float32)
        edges = tf.cast(edges, tf.float32)
        edges += tf.eye(tf.shape(edges)[1])
        output = tf.matmul(edges, nodes)
        output = tf.matmul(output, self.kernel)

        if self.use_bias:
            output += self.bias

        return self.activation(output)

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        return (None,input_shape[0][1], self.output_dim)

    def get_config(self):
        config = {
            'units': self.output_dim,
            'activation': activations.serialize(self.activation),
        }

        base_config = super(GCNLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
