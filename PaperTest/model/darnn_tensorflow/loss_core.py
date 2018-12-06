#!/usr/bin/env python


import tensorflow as tf


def timeseries_loss(target, pred):
    loss = tf.losses.mean_squared_error(target, pred, reduction=tf.losses.Reduction.MEAN)
    tf.summary.scalar("loss", loss)

    return loss
