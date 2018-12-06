# -*- coding: utf-8 -*-
"""
    @Time    : 2018/11/28 11:35
    @Author  : ZERO
    @FileName: TSModel.py
    @Software: PyCharm
    @Github    ：https://github.com/abcdddxy
"""

import sys

import numpy as np
import tensorflow as tf


class TSModel(object):
    def __new__(cls, params):
        model = object.__new__(cls)
        model.params = params

        model.T = params["T"]
        model.encoder_hidden_size = params["encoder_hidden_size"]
        model.decoder_hidden_size = params["decoder_hidden_size"]
        model.input_size = params["input_size"]
        model.batch_size = params["batch_size"]

        # T - 1
        model.inputs = {
            "X": tf.placeholder(tf.float32, shape=[None, model.T, model.input_size], name="X"),
            "y_history": tf.placeholder(tf.float32, shape=[None, model.T], name="y_history")
        }
        model.target = tf.placeholder(dtype=tf.int32, shape=[None], name="target")

        model.initializer = tf.contrib.layers.xavier_initializer()
        model.global_step = tf.train.create_global_step()

        model.conf_keep_prob = params["keep_prob"] if "keep_prob" in params else 1.

        model.keep_prob = None
        model.logits = None
        model.loss = None
        model.pred = None
        model.train_op = None

        model.save_path = params["model_path"]
        model.saver = None

        return model

    def create_feed_dict(self, is_train, batch):
        feed_dict = {self.inputs["X"]: batch[0],
                     self.inputs["y_history"]: batch[1]}
        if is_train:
            feed_dict[self.target] = batch[2]
        if self.keep_prob is not None:
            feed_dict[self.target] = batch[2]
            feed_dict[self.keep_prob] = self.conf_keep_prob
        return feed_dict

    def run_step(self, sess, is_train, batch, merge_summary=None, train_writer=None):
        feed_dict = self.create_feed_dict(is_train, batch)
        if is_train:
            fetchers = [self.global_step, self.loss, self.pred, self.train_op]
            if merge_summary is not None:
                fetchers.append(merge_summary)
            ret = sess.run(fetchers, feed_dict)

            global_step, loss, pred = ret[:3]
            if merge_summary is not None:
                train_writer.add_summary(ret[4], global_step)
            return global_step, loss, pred
        else:
            loss, pred = sess.run([self.loss, self.pred], feed_dict)
            return loss, pred

    def train(self, sess, batch_manager, steps, metrics, merge_summary=None, train_writer=None):
        global_step = 0
        total_loss = 0
        n_steps = 0
        infinity = steps < 0
        target = []
        preds = []
        while True:
            try:
                batch, batch_size = batch_manager.batch()
            except EOFError:
                break
            global_step, loss, pred = self.run_step(sess, True, batch, merge_summary=merge_summary, train_writer=train_writer)
            total_loss += loss
            n_steps += 1
            if not infinity:
                steps -= 1
                if steps <= 0:
                    break
            target.append(batch[2])
            preds.append(pred)

        target = np.concatenate(target)
        preds = np.concatenate(preds)
        print("target: ", target[:5])
        print("pred: ", [i[0] for i in preds][:5])
        return global_step, total_loss / n_steps, n_steps, metrics(target, preds, key="train")

    def eval(self, sess, batch_manager, metrics):
        total_loss = 0
        target = []
        preds = []
        while True:
            try:
                batch, batch_size = batch_manager.batch()
            except EOFError:
                break
            target.append(batch[2])
            loss, pred = self.run_step(sess, False, batch)
            total_loss += loss
            preds.append(pred)
        target = np.concatenate(target)
        preds = np.concatenate(preds)
        return total_loss, metrics(target, preds, key="val")

    def predict(self, sess, batch_manager, steps):
        preds = []
        n_steps = 0
        for _ in range(steps):
            try:
                batch, batch_size = batch_manager.batch()
            except EOFError:
                break
            preds.append(self.run_step(sess, False, batch)[1])
            n_steps += 1
        return np.stack(preds), n_steps

    def save(self, sess, save_path=None):
        if save_path is None:
            save_path = self.save_path
        self.saver.save(sess, save_path)
