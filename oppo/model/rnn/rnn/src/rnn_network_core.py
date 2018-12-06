#!/usr/bin/env python

import tensorflow as tf
import keras
import numpy as np


def cnn_item(model_inputs, model_inputs_dim, score_weight, tag_weight, keep_prob, layer_conf, initializer, layer_tag='0'):
    filter_width = layer_conf["filter_width"]
    output_channel = layer_conf["output_channel"]

    # batch_size*segment_len(10)*128 加入scores
    # score_weight = tf.divide(score_weight, tf.expand_dims(tf.reduce_sum(score_weight, axis=1), axis=-1))
    model_inputs = model_inputs + tf.expand_dims(score_weight, -1)
    model_inputs = tf.expand_dims(model_inputs, 1)

    with tf.variable_scope("cnn_layer"):
        with tf.variable_scope(layer_tag + "_input_layer"):
            filter_shape = [1, filter_width, model_inputs_dim, output_channel]
            filter_weight = tf.get_variable(
                layer_tag + "_filter_weight",
                shape=filter_shape,
                initializer=initializer
            )
            filter_bias = tf.get_variable(
                layer_tag + "_filter_bias",
                shape=[output_channel]
            )

            layer = tf.nn.conv2d(model_inputs, filter_weight, strides=[1, 1, 1, 1],
                                 padding="SAME", name=layer_tag + "_input_layer")
            layer = tf.nn.bias_add(layer, filter_bias)
            layer = tf.nn.relu(layer)

        layer = tf.squeeze(layer, [1])
        pooling_output = tf.reduce_max(layer, axis=-2, name=layer_tag + "_cnn_pooling_layer")

        # batch_size*64 加入tags
        pooling_output = tf.add(pooling_output, tag_weight)

    return pooling_output, output_channel


def gru_item(model_inputs, model_inputs_dim, model_inputs_len, keep_prob, layer_conf, initializer, layer_tag):
    gru_size = layer_conf["gru_size"]
    gru_layers = layer_conf["gru_layers"]
    padding_len = layer_conf["padding_len"]
    attention_size = layer_conf["attention_size"]

    with tf.variable_scope(layer_tag + "_layer"):
        if model_inputs_len is not None:
            if gru_layers == 1:
                gru_cell_forward = tf.nn.rnn_cell.GRUCell(gru_size)
                gru_cell_backward = tf.nn.rnn_cell.GRUCell(gru_size)
            else:
                gru_cell_forward = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.GRUCell(gru_size) for _ in range(gru_layers)])
                gru_cell_backward = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.GRUCell(gru_size) for _ in range(gru_layers)])
            gru_cell_forward = tf.contrib.rnn.DropoutWrapper(gru_cell_forward, output_keep_prob=keep_prob)
            gru_cell_backward = tf.contrib.rnn.DropoutWrapper(gru_cell_backward, output_keep_prob=keep_prob)
            gru_cell_forward = tf.contrib.rnn.AttentionCellWrapper(gru_cell_forward, attn_length=padding_len)
            gru_cell_backward = tf.contrib.rnn.AttentionCellWrapper(gru_cell_backward, attn_length=padding_len)
            gru_outputs, _ = tf.nn.bidirectional_dynamic_rnn(gru_cell_forward, gru_cell_backward, inputs=model_inputs,
                                                             sequence_length=model_inputs_len, dtype=tf.float32)
            gru_outputs = tf.concat(gru_outputs, 2)
        else:
            gru_cell_forward = tf.nn.rnn_cell.GRUCell(gru_size)
            gru_cell_backward = tf.nn.rnn_cell.GRUCell(gru_size, reuse=tf.AUTO_REUSE)
            gru_cell_forward = tf.contrib.rnn.DropoutWrapper(gru_cell_forward, output_keep_prob=keep_prob)
            gru_cell_backward = tf.contrib.rnn.DropoutWrapper(gru_cell_backward, output_keep_prob=keep_prob)
            gru_cell_forward = tf.contrib.rnn.AttentionCellWrapper(gru_cell_forward, attn_length=padding_len)
            gru_cell_backward = tf.contrib.rnn.AttentionCellWrapper(gru_cell_backward, attn_length=padding_len)
            model_inputs_forward = model_inputs
            model_inputs_backward = tf.reverse(model_inputs, [1])
            forward_outputs, forward_final_state = tf.nn.dynamic_rnn(gru_cell_forward, model_inputs_forward, dtype=tf.float32)
            backward_outputs, backward_final_state = tf.nn.dynamic_rnn(gru_cell_backward, model_inputs_backward, dtype=tf.float32)
            backward_outputs = tf.reverse(backward_outputs, [1])
            gru_outputs = tf.concat([forward_outputs, backward_outputs], 2)
            gru_outputs = tf.nn.relu(gru_outputs)

        # 单层attention
        # with tf.variable_scope("attention"):
        #     hidden_size = gru_outputs.shape[2].value
        #
        #     attention_w = tf.get_variable("attention_w", initializer=tf.truncated_normal([hidden_size, attention_size], stddev=0.1))
        #     attention_b = tf.get_variable("attention_b", initializer=tf.constant(0.1, shape=[attention_size]))
        #     attention_u = tf.get_variable("attention_u", initializer=tf.truncated_normal([attention_size], stddev=0.1))
        #     v = tf.tanh(tf.tensordot(gru_outputs, attention_w, axes=1) + attention_b)
        #     vu = tf.tensordot(v, attention_u, axes=1)
        #     alphas = tf.nn.softmax(vu)
        #     output = tf.reduce_sum(gru_outputs * tf.expand_dims(alphas, -1), 1)

            # 多层attention layer
        #         def attention_layer(inputs, layer_name):
        #             with tf.variable_scope(layer_name):
        #                 attention_w = tf.get_variable("attention_w", initializer=tf.truncated_normal([gru_size, attention_size], stddev=0.1))
        #                 attention_b = tf.get_variable("attention_b", initializer=tf.constant(0.1, shape=[attention_size]))
        #                 attention_u = tf.get_variable("attention_u", initializer=tf.truncated_normal([attention_size], stddev=0.1))
        #                 z_list = []
        #                 for t in range(padding_len):
        #                     try:
        #                         v_att = tf.tanh(tf.matmul(gru_outputs[:, t, :], attention_w) + tf.reshape(attention_b, [1, -1]))
        #                         z_att = tf.matmul(v_att, tf.reshape(attention_u, [-1, 1]))
        #                         z_list.append(z_att)
        #                     except:
        #                         break
        #                 attention_z = tf.concat(z_list, axis=1)
        #                 alpha = tf.nn.softmax(attention_z)
        #                 attention_output = tf.reduce_sum(inputs * tf.reshape(alpha, [-1, model_inputs_len, 1]), 1)
        #                 return attention_output

        gru_outputs = tf.reduce_sum(gru_outputs, axis=1, name=layer_tag + "_gru_pooling_layer")

    return gru_outputs, gru_size * 2


def network(inputs, keep_prob, params, initializer):
    gru_layer_conf = params["gru_layers"]
    cnn_layer_conf = params["cnn_layers"]
    outputs = list()
    outputs_dim = list()
    # ["prefix", "title", "texts", "segments", "tag", "scores", "extra]:
    # inputs["segments"]["embedding"].shape = batch * text_num(10) * text_length * embedding_dim

    with tf.variable_scope("basic_info_layer"):
        for tag in ["prefix", "title"]:
            model_inputs = inputs[tag]["embedding"]
            model_inputs_dim = inputs[tag]["embedding_dim"]
            model_inputs_len = tf.squeeze(inputs[tag + "_len"])
            output, output_dim = gru_item(model_inputs, model_inputs_dim, model_inputs_len, keep_prob, gru_layer_conf, initializer, tag)
            outputs.append(output)
            outputs_dim.append(output_dim)

    with tf.variable_scope("segments_letters_layer"):
        for tag in ["segments", "letters"]:
            segments_outputs = list()
            for i in range(10):
                model_inputs = inputs[tag]["embedding"][:, i, :, :]
                model_inputs_dim = inputs[tag]["embedding_dim"]
                model_inputs_len = tf.squeeze(inputs[tag + "_len"][:, i])
                output, output_dim = gru_item(model_inputs, model_inputs_dim, model_inputs_len, keep_prob, gru_layer_conf, initializer,
                                              tag + "_" + str(i + 1))
                outputs.append(output)
                outputs_dim.append(output_dim)
                segments_outputs.append(tf.expand_dims(output, 1))
            segments_output_dim = output_dim

    with tf.variable_scope("sentence_layer"):
        for tag in ["segments_sentence", "letters_sentence"]:
            segments_outputs = tf.concat(segments_outputs, axis=1)
            model_inputs = segments_outputs
            model_inputs_dim = segments_output_dim
            score_weight = inputs["scores"]
            tag_weight = tf.squeeze(inputs["tag"]["embedding"], axis=1)
            output, output_dim = cnn_item(model_inputs, model_inputs_dim, score_weight, tag_weight, keep_prob, cnn_layer_conf, initializer, tag)
            outputs.append(output)
            outputs_dim.append(output_dim)

    with tf.variable_scope("tag_layer"):
        tag = "tag"
        output = tf.squeeze(inputs[tag]["embedding"], axis=1)
        output_dim = inputs[tag]["embedding_dim"]
        outputs.append(output)
        outputs_dim.append(output_dim)

    with tf.variable_scope("scores_layer"):
        tag = "scores"
        output = inputs[tag]
        output_dim = 10
        outputs.append(output)
        outputs_dim.append(output_dim)

    with tf.variable_scope("distance_layer"):
        for tagx in ["prefix", "title"]:
            for tagy in ["segments", "letters"]:
                # average pooling
                x = tf.nn.l2_normalize(tf.reduce_mean(inputs[tagx]["embedding"], axis=1), dim=0)
                y = tf.nn.l2_normalize(tf.reduce_mean(tf.reduce_mean(inputs[tagy]["embedding"], axis=1), axis=1), dim=0)
                dis = tf.divide(tf.reduce_sum(tf.sqrt(tf.multiply(tf.square(x), tf.square(y))), axis=1),
                                tf.reduce_sum(tf.multiply(x, y), axis=1))
                outputs.append(tf.expand_dims(dis, -1))
                dis = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(x, y)), axis=1))
                outputs.append(tf.expand_dims(dis, -1))
                dis = tf.diag_part(tf.matmul(x, y, transpose_b=True))
                outputs.append(tf.expand_dims(dis, -1))
                # max pooling
                x = tf.nn.l2_normalize(tf.reduce_mean(inputs[tagx]["embedding"], axis=1), dim=0)
                y = tf.nn.l2_normalize(tf.reduce_max(tf.reduce_max(inputs[tagy]["embedding"], axis=1), axis=1), dim=0)
                dis = tf.divide(tf.reduce_sum(tf.sqrt(tf.multiply(tf.square(x), tf.square(y))), axis=1),
                                tf.reduce_sum(tf.multiply(x, y), axis=1))
                outputs.append(tf.expand_dims(dis, -1))
                dis = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(x, y)), axis=1))
                outputs.append(tf.expand_dims(dis, -1))
                dis = tf.diag_part(tf.matmul(x, y, transpose_b=True))
                outputs.append(tf.expand_dims(dis, -1))
                # random pooling
                x = tf.nn.l2_normalize(tf.reduce_mean(inputs[tagx]["embedding"], axis=1), dim=0)
                y = tf.nn.l2_normalize(tf.reduce_max(tf.reduce_mean(inputs[tagy]["embedding"], axis=1), axis=1), dim=0)
                dis = tf.divide(tf.reduce_sum(tf.sqrt(tf.multiply(tf.square(x), tf.square(y))), axis=1),
                                tf.reduce_sum(tf.multiply(x, y), axis=1))
                outputs.append(tf.expand_dims(dis, -1))
                dis = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(x, y)), axis=1))
                outputs.append(tf.expand_dims(dis, -1))
                dis = tf.diag_part(tf.matmul(x, y, transpose_b=True))
                outputs.append(tf.expand_dims(dis, -1))

                outputs_dim.append(9)

    # with tf.variable_scope("extra_feature_layer"):
    #     tag = "extra"
    #     output = inputs[tag]
    #     output_dim = 58
    #     outputs.append(output)
    #     outputs_dim.append(output_dim)

    final_output = tf.concat(outputs, axis=-1)
    final_output_dim = sum(outputs_dim)

    return final_output, final_output_dim
