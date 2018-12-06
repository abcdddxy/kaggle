#!/usr/bin/env python

import tensorflow as tf


def get_train_op(loss, global_step, params):
    optimizer_type = params["optimizer_type"]
    learning_rate = params["learning_rate"]
    lr_decay = params["lr_decay"]
    lr_decay_steps = params["lr_decay_steps"]
    lr = tf.train.exponential_decay(learning_rate, global_step, lr_decay_steps, lr_decay, staircase=True)
    tf.summary.scalar("learning_rate", lr)

    with tf.variable_scope("optimizer"):
        if optimizer_type == "sgd":
            opt = tf.train.GradientDescentOptimizer(lr)
        elif optimizer_type == "adam":
            opt = tf.train.AdamOptimizer(lr)
        elif optimizer_type == "adgrad":
            opt = tf.train.AdagradOptimizer(lr)
        else:
            raise KeyError
        train_op = opt.minimize(loss, global_step=global_step)

        return train_op
