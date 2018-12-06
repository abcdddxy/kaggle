"""
    @Time    : 2018/11/28 10:23
    @Author  : ZERO
    @FileName: util.py.py
    @Software: PyCharm
    @Github    ：https://github.com/abcdddxy
"""

import os
import sys
import time
from datetime import timedelta
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error
import math

from input import *
import util
from model import Model

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

expname = "timeseriesModel"

params = {
    "data_file": "../../data/nasdaq100/small/nasdaq100_padding.csv",

    "model_path": "./output/model",
    "train_summary_path": "./output/summary",
    "eval_summary_path": "./output/eval",

    "train_epoch": 100,
    "batch_size": 128,
    "shuffle": False,
    "debug": False,

    "encoder_hidden_size": 64,
    "decoder_hidden_size": 64,
    "T": 10,

    "optimizer_type": "adam",
    "learning_rate": 0.00001,
    "lr_decay": 1,
    "lr_decay_steps": 10,

    "train_size": 35100,
    "val_size": 2500,
    "test_size": 2960,

    "steps_per_run": 100
}


def train():
    logger = util.setup_log("ZERO")
    train_set, valid_set = load_data(params)
    params["input_size"] = train_set["data"].shape[1]
    logger.info("Shape of data: %s.", train_set["data"].shape)

    train_bm = BatchManager(train_set, 1, params)
    valid_bm = BatchManager(valid_set, 1, params)

    model = Model(params, logger)
    merge_summary = tf.summary.merge_all()

    def mean_absolute_percentage_error(y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    def metrics(targets, preds, key="train"):
        # if key == "val":
        #     print('mae:', mean_absolute_error(targets, preds))
        #     print('mape:', mean_absolute_percentage_error(targets, preds))
        #     print('rmse:', math.sqrt(mean_squared_error(targets, preds)))
        return mean_squared_error(targets, preds), ""

    def get_time_dif(start_time):
        end_time = time.time()
        time_dif = end_time - start_time
        return timedelta(seconds=int(round(time_dif)))

    summary_path = os.path.join(params["train_summary_path"], expname)
    _ = os.system("rm -rf %s" % summary_path)
    model_path = os.path.join(params["model_path"], expname)
    _ = os.system("rm -rf %s" % model_path)
    eval_summary_path = os.path.join(params["eval_summary_path"], expname)
    _ = os.system("rm -rf %s" % eval_summary_path)
    _ = os.system("mkdir -p %s" % eval_summary_path)

    train_writer = tf.summary.FileWriter(summary_path, tf.get_default_graph())

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        steps_per_run = params["steps_per_run"]
        global_step = 0
        best_mse = sys.maxsize
        start_time = time.time()
        valid_step = 0
        for epoch in range(params["train_epoch"]):
            train_bm.init()
            while True:
                global_step, loss, n_steps, (mse, _) = model.train(sess, train_bm, steps_per_run, metrics, merge_summary=merge_summary,
                                                                   train_writer=train_writer)
                logger.info("TRAIN %d steps[%d]: loss %.4f  mse %.4f" % (global_step, epoch, loss, mse))
                if train_bm.is_finished:
                    break

                valid_step += 1
                if valid_step % 10 == 0:
                    valid_bm.init()
                    loss, (mse, _) = model.eval(sess, valid_bm, metrics)
                    if mse < best_mse:
                        best_mse = mse
                        model.save(sess, save_path=model_path)
                        best_flag = '*'
                    else:
                        best_flag = ''

                    time_dif = get_time_dif(start_time)
                    logger.info("EVALUATION: %d steps: loss %.4f  mse %.4f  cost_time %s  %s" % (global_step, loss, mse, str(time_dif), best_flag))


def predict():
    class Model:
        def __init__(self, params):
            graph = tf.Graph()
            with graph.as_default():
                config = tf.ConfigProto()
                config.gpu_options.allow_growth = True
                self.sess = tf.Session(config=config)
                with self.sess.as_default():
                    save_path = params["model_path"] + "/" + expname
                    saver = tf.train.import_meta_graph("%s.meta" % save_path)
                    saver.restore(self.sess, save_path)

                    # placeholder
                    self.inputs = {
                        "X": graph.get_operation_by_name("X").outputs[0],
                        "y_history": graph.get_operation_by_name("y_history").outputs[0]
                    }

                    self.keep_prob = graph.get_operation_by_name("keep_prob").outputs[0]
                    self.prediction = graph.get_operation_by_name("Decoder/prediction").outputs[0]

        def predict(self, batch):
            feed_dict = {self.inputs["X"]: batch[0],
                         self.inputs["y_history"]: batch[1],
                         self.keep_prob: 1}
            pred = self.sess.run(self.prediction, feed_dict=feed_dict)
            return pred

    predict_set = load_data(params)
    predict_bm = BatchManager(predict_set, 1, params)

    model = Model(params)
    predict_bm.init()
    with open("./output/result.csv", "w") as f:
        while True:
            batch, batch_size = predict_bm.batch()
            preds = model.predict(batch)
            f.write(preds)
            f.write("\n")
            if predict_bm.is_finished:
                break


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ["train", "pred"]:
        raise ValueError("""usage: python run_model.py [train / pred]""")

    if sys.argv[1] == "train":
        params["keyword"] = "train"
        train()
    else:
        params["keyword"] = "predict"
        predict()
