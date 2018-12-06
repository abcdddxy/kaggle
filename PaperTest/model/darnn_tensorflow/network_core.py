# -*- coding: utf-8 -*-
"""
    @Time    : 2018/11/29 11:01
    @Author  : ZERO
    @FileName: util.py.py
    @Software: PyCharm
    @Github    ：https://github.com/abcdddxy
"""

import tensorflow as tf
import keras


class Encoder:
    def __init__(self, batch_size, input_size, hidden_size, T, init, logger):
        super(Encoder, self).__init__()
        self.batch_size = batch_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.T = T
        self.init = init

        self.logger = logger

        self.rnn_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=hidden_size)
        # self.rnn_cell = keras.layers.LSTM(units=self.hidden_size, return_sequences=True)

    def forward(self, input_data):
        with tf.variable_scope('Encoder'):
            input_weight = tf.get_variable("input_weight", shape=[self.batch_size, self.T, self.input_size], dtype=tf.float32,
                                           initializer=tf.zeros_initializer)
            input_encode = tf.get_variable("input_encode", shape=[self.batch_size, self.T, self.hidden_size], dtype=tf.float32,
                                           initializer=tf.zeros_initializer)

            state = self.rnn_cell.zero_state(self.batch_size, dtype=tf.float32)
            cell, hidden = state

            # V = tf.get_variable("Ve", shape=[self.T - 1], dtype=tf.float32, initializer=self.init)
            # W = tf.get_variable("We", shape=[self.T - 1, 2 * self.hidden_size], dtype=tf.float32, initializer=self.init)
            # U = tf.get_variable("Ue", shape=[self.T - 1, self.T - 1], dtype=tf.float32, initializer=self.init)
            # for t in range(self.T - 1):
            #     hs = tf.concat([tf.transpose(tf.tile(tf.expand_dims(hidden, axis=0), multiples=[self.input_size, 1, 1]), perm=[1, 0, 2]),
            #                     tf.transpose(tf.tile(tf.expand_dims(cell, axis=0), multiples=[self.input_size, 1, 1]), perm=[1, 0, 2])],
            #                    axis=2)  # batch_size * input_size * 2*hidden_size
            #     input1 = tf.tensordot(hs, tf.transpose(W, perm=[1, 0]), axes=[[2], [0]])  # batch_size * input_size * (T-1)
            #     input2 = tf.tensordot(tf.transpose(input_data, perm=[0, 2, 1]), U, axes=[[2], [0]])  # batch_size * input_size * (T-1)
            #     x = tf.tanh(tf.add(input1, input2))  # batch_size * input_size * (T-1)
            #     x = tf.squeeze(tf.tensordot(x, V, axes=[[2], [0]]))  # batch_size * input_size
            #     alpha = tf.nn.softmax(x)  # batch_size * input_size
            #     alpha_x = tf.multiply(alpha, input_data[:, t, :])  # batch_size * input_size
            #     _, state = self.rnn_cell(alpha_x, state)
            #     cell, hidden = state  # batch_size * hidden_size
            #     tf.assign(input_weight[:, t, :], alpha_x)  # batch_size * (T-1) * input_size
            #     tf.assign(input_encode[:, t, :], hidden)  # batch_size * (T-1) * hidden_size

            for t in range(self.T):
                hs = tf.concat([tf.transpose(tf.tile(tf.expand_dims(hidden, axis=0), multiples=[self.input_size, 1, 1]), perm=[1, 0, 2]),
                                tf.transpose(tf.tile(tf.expand_dims(cell, axis=0), multiples=[self.input_size, 1, 1]), perm=[1, 0, 2])],
                               axis=2)  # batch_size * input_size * 2*hidden_size
                input1 = tf.layers.dense(hs, units=self.T)  # batch_size * input_size * T
                input2 = tf.layers.dense(tf.transpose(input_data, perm=[0, 2, 1]), units=self.T)  # batch_size * input_size * T
                x = tf.tanh(tf.add(input1, input2))  # batch_size * input_size * T
                x = tf.squeeze(tf.layers.dense(x, units=1))  # batch_size * input_size
                alpha = tf.nn.softmax(x)  # batch_size * input_size
                alpha_x = tf.multiply(alpha, input_data[:, t, :])  # batch_size * input_size
                _, state = self.rnn_cell(alpha_x, state)
                cell, hidden = state  # batch_size * hidden_size
                tf.assign(input_weight[:, t, :], alpha_x)  # batch_size * T * input_size
                tf.assign(input_encode[:, t, :], hidden)  # batch_size * T * hidden_size

            # for t in range(self.T - 1):
            #     x = tf.concat([tf.transpose(tf.tile(tf.expand_dims(hidden, axis=0), multiples=[self.input_size, 1, 1]), perm=[1, 0, 2]),
            #                    tf.transpose(tf.tile(tf.expand_dims(cell, axis=0), multiples=[self.input_size, 1, 1]), perm=[1, 0, 2]),
            #                    tf.transpose(input_data, perm=(0, 2, 1))], axis=2)  # batch_size * input_size * (2*hidden_size + (T-1))
            #     # x = tf.layers.dense(tf.reshape(x, shape=[-1, 2 * self.hidden_size + self.T - 1]), units=1)
            #     x = tf.reshape(x, shape=[-1, 2 * self.hidden_size + self.T - 1])
            #     x = tf.layers.dense(x, units=self.T - 1, activation=tf.nn.tanh)  # (batch_size*input_size) * (T-1)
            #     x = tf.layers.dense(x, units=1)  # (batch_size*(T-1)) * 1
            #     alpha = tf.nn.softmax(tf.reshape(x, shape=[-1, self.input_size]))  # batch_size * input_size
            #     alpha_x = tf.multiply(alpha, input_data[:, t, :])  # batch_size * input_size
            #     _, state = self.rnn_cell(alpha_x, state)
            #     cell, hidden = state  # batch_size * hidden_size1
            #     tf.assign(input_weight[:, t, :], alpha_x)  # batch_size *(T-1) * input_size
            #     tf.assign(input_encode[:, t, :], hidden)  # batch_size * (T-1) * hidden_size

        return input_weight, input_encode


class Decoder:
    def __init__(self, batch_size, encoder_hidden_size, decoder_hidden_size, T, init, logger):
        super(Decoder, self).__init__()
        self.batch_size = batch_size
        self.T = T
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.init = init

        self.logger = logger

        self.rnn_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=decoder_hidden_size)

    def forward(self, input_encoded, y_history):
        with tf.variable_scope('Decoder'):
            state = self.rnn_cell.zero_state(self.batch_size, dtype=tf.float32)
            cell, hidden = state

            # V = tf.get_variable("Vd", shape=[self.encoder_hidden_size], dtype=tf.float32, initializer=self.init)
            # W = tf.get_variable("Wd", shape=[self.encoder_hidden_size, 2 * self.decoder_hidden_size], dtype=tf.float32, initializer=self.init)
            # U = tf.get_variable("Ud", shape=[self.encoder_hidden_size, self.encoder_hidden_size], dtype=tf.float32, initializer=self.init)
            # for t in range(self.T - 1):
            #     hs = tf.concat([tf.transpose(tf.tile(tf.expand_dims(hidden, axis=0), multiples=[self.T - 1, 1, 1]), perm=[1, 0, 2]),
            #                     tf.transpose(tf.tile(tf.expand_dims(cell, axis=0), multiples=[self.T - 1, 1, 1]), perm=[1, 0, 2])],
            #                    axis=2)  # batch_size * (T-1) * 2*decoder_hidden_size
            #     input1 = tf.tensordot(hs, tf.transpose(W, perm=[1, 0]), axes=[[2], [0]])  # batch_size * (T-1) * encoder_hidden_size
            #     input2 = tf.tensordot(input_encoded, U, axes=[[2], [0]])  # batch_size * (T-1) * encoder_hidden_size
            #     x = tf.tanh(tf.add(input1, input2))  # batch_size * (T-1) * encoder_hidden_size
            #     x = tf.squeeze(tf.tensordot(x, V, axes=[[2], [0]]))  # batch_size * (T-1)
            #     x = tf.nn.softmax(x)  # batch_size * (T-1)
            #     context = tf.matmul(tf.expand_dims(x, axis=1), input_encoded)[:, 0, :]  # batch_size * encoder_hidden_size
            #     y_tilde = tf.layers.dense(tf.concat([context, tf.expand_dims(y_history[:, t], axis=1)], axis=1), units=1)  # batch_size * 1
            #     _, state = self.rnn_cell(y_tilde, state)
            #     cell, hidden = state

            for t in range(self.T):
                hs = tf.concat([tf.transpose(tf.tile(tf.expand_dims(hidden, axis=0), multiples=[self.T, 1, 1]), perm=[1, 0, 2]),
                                tf.transpose(tf.tile(tf.expand_dims(cell, axis=0), multiples=[self.T, 1, 1]), perm=[1, 0, 2])],
                               axis=2)  # batch_size * T * 2*decoder_hidden_size
                input1 = tf.layers.dense(hs, units=self.encoder_hidden_size)  # batch_size * T * encoder_hidden_size
                input2 = tf.layers.dense(input_encoded, units=self.encoder_hidden_size)  # batch_size * T * encoder_hidden_size
                x = tf.tanh(tf.add(input1, input2))  # batch_size * T * encoder_hidden_size
                x = tf.squeeze(tf.layers.dense(x, units=1))  # batch_size * T
                x = tf.nn.softmax(x)  # batch_size * T
                context = tf.matmul(tf.expand_dims(x, axis=1), input_encoded)[:, 0, :]  # batch_size * encoder_hidden_size
                if t < self.T - 1:
                    y_tilde = tf.layers.dense(tf.concat([context, tf.expand_dims(y_history[:, t], axis=1)], axis=1), units=1)  # batch_size * 1
                    _, state = self.rnn_cell(y_tilde, state)
                    cell, hidden = state

            # for t in range(self.T - 1):
            #     x = tf.concat([tf.transpose(tf.tile(tf.expand_dims(hidden, axis=0), multiples=[self.T - 1, 1, 1]), perm=[1, 0, 2]),
            #                    tf.transpose(tf.tile(tf.expand_dims(cell, axis=0), multiples=[self.T - 1, 1, 1]), perm=[1, 0, 2]),
            #                    input_encoded], axis=2)  # batch_size * (T-1) * (2*decoder_hidden_size + encoder_hidden_size)
            #     x = tf.reshape(x, shape=[-1, 2 * self.decoder_hidden_size + self.encoder_hidden_size])
            #     x = tf.layers.dense(x, units=self.encoder_hidden_size, activation=tf.nn.tanh)  # (batch_size*(T-1)) * encoder_hidden_size
            #     x = tf.layers.dense(x, units=1)  # (batch_size*(T-1)) * 1
            #     x = tf.nn.softmax(tf.reshape(x, shape=[-1, self.T - 1]))  # batch_size * (T-1)
            #     context = tf.matmul(tf.expand_dims(x, axis=1), input_encoded)[:, 0, :]  # batch_size * encoder_hidden_size
            #     y_tilde = tf.layers.dense(tf.concat([context, tf.expand_dims(y_history[:, t], axis=1)], axis=1), units=1)  # batch_size * 1
            #     _, state = self.rnn_cell(y_tilde, state)
            #     cell, hidden = state

            y_pred = tf.layers.dense(tf.concat([hidden, context], axis=1), units=1, name="prediction")

        return y_pred
