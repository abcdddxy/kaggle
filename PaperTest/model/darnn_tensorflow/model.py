#!/usr/bin/env python
# coding: utf-8

import sys
import tensorflow as tf

import network_core
import loss_core
import util
import optimizer_core
from TSModel import TSModel


class Model(TSModel):
    def __new__(cls, params, logger):
        model = TSModel.__new__(cls, params)
        model.keep_prob = tf.placeholder(tf.float32, name="keep_prob")

        model.encoder = network_core.Encoder(batch_size=params["batch_size"], input_size=params["input_size"],
                                             hidden_size=params["encoder_hidden_size"], T=params["T"], init=model.initializer, logger=logger)
        model.decoder = network_core.Decoder(batch_size=params["batch_size"], encoder_hidden_size=params["encoder_hidden_size"],
                                             decoder_hidden_size=params["decoder_hidden_size"], T=params["T"], init=model.initializer, logger=logger)

        model.input_weighted, model.input_encoded = model.encoder.forward(model.inputs["X"])
        model.pred = model.decoder.forward(model.input_encoded, model.inputs["y_history"])

        model.loss = loss_core.timeseries_loss(target=tf.expand_dims(model.target, axis=1), pred=model.pred)
        model.train_op = optimizer_core.get_train_op(model.loss, model.global_step, model.params)

        # saver of the model
        model.saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)
        return model
