import tensorflow as tf
from tensorflow.python.ops import math_ops
import time
import random
import numpy as np
from utils import get_batch_index
from tqdm import tqdm_notebook


class IAN(object):
    def __init__(self, config, sess):
        self.embedding_dim = config['embedding_dim']
        self.batch_size = config['batch_size']
        self.n_epoch = config['n_epoch']
        self.n_hidden = config['n_hidden']
        self.n_class = config['n_class']
        self.learning_rate = config['learning_rate']
        self.l2_reg = config['l2_reg']
        self.dropout = config['dropout']
        self.max_aspect_len = config['max_aspect_len']
        self.max_context_len = config['max_context_len']
        self.embedding_matrix = config['embedding_matrix']
        self.early_stop = config['early_stop']
        self.sess = sess

    def build_model(self):
        with tf.name_scope('inputs'):
            self.aspects = tf.placeholder(tf.int32, [None, self.max_aspect_len])
            self.contexts = tf.placeholder(tf.int32, [None, self.max_context_len])
            self.labels = tf.placeholder(tf.int32, [None, self.n_class])
            self.aspect_lens = tf.placeholder(tf.int32, None)
            self.context_lens = tf.placeholder(tf.int32, None)
            self.dropout_keep_prob = tf.placeholder(tf.float32)

            aspect_inputs = tf.nn.embedding_lookup(self.embedding_matrix, self.aspects)
            aspect_inputs = tf.cast(aspect_inputs, tf.float32)
            aspect_inputs = tf.nn.dropout(aspect_inputs, keep_prob=self.dropout_keep_prob)

            context_inputs = tf.nn.embedding_lookup(self.embedding_matrix, self.contexts)
            context_inputs = tf.cast(context_inputs, tf.float32)
            context_inputs = tf.nn.dropout(context_inputs, keep_prob=self.dropout_keep_prob)

        with tf.name_scope('weights'):
            weights = {
                'aspect_score': tf.get_variable(
                    name='W_a',
                    shape=[self.n_hidden, self.n_hidden],
                    initializer=tf.random_uniform_initializer(-0.1, 0.1),
                    regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg)
                ),
                'context_score': tf.get_variable(
                    name='W_c',
                    shape=[self.n_hidden, self.n_hidden],
                    initializer=tf.random_uniform_initializer(-0.1, 0.1),
                    regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg)
                ),
                'softmax': tf.get_variable(
                    name='W_l',
                    shape=[self.n_hidden * 2, self.n_class],
                    initializer=tf.random_uniform_initializer(-0.1, 0.1),
                    regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg)
                ),
            }

        with tf.name_scope('biases'):
            biases = {
                'aspect_score': tf.get_variable(
                    name='B_a',
                    shape=[self.max_aspect_len, 1],
                    initializer=tf.zeros_initializer(),
                    regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg)
                ),
                'context_score': tf.get_variable(
                    name='B_c',
                    shape=[self.max_context_len, 1],
                    initializer=tf.zeros_initializer(),
                    regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg)
                ),
                'softmax': tf.get_variable(
                    name='B_l',
                    shape=[self.n_class],
                    initializer=tf.zeros_initializer(),
                    regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg)
                ),
            }

        with tf.name_scope('dynamic_rnn'):
            aspect_outputs, aspect_state = tf.nn.dynamic_rnn(
                tf.contrib.rnn.LSTMCell(self.n_hidden),
                inputs=aspect_inputs,
                sequence_length=self.aspect_lens,
                dtype=tf.float32,
                scope='aspect_lstm'
            )
            batch_size = tf.shape(aspect_outputs)[0]
            aspect_avg = tf.reduce_mean(aspect_outputs, 1)

            context_outputs, context_state = tf.nn.dynamic_rnn(
                tf.contrib.rnn.LSTMCell(self.n_hidden),
                inputs=context_inputs,
                sequence_length=self.context_lens,
                dtype=tf.float32,
                scope='context_lstm'
            )
            context_avg = tf.reduce_mean(context_outputs, 1)

            aspect_outputs_iter = tf.TensorArray(tf.float32, 1, dynamic_size=True, infer_shape=False)
            aspect_outputs_iter = aspect_outputs_iter.unstack(aspect_outputs)
            context_avg_iter = tf.TensorArray(tf.float32, 1, dynamic_size=True, infer_shape=False)
            context_avg_iter = context_avg_iter.unstack(context_avg)
            aspect_lens_iter = tf.TensorArray(tf.int32, 1, dynamic_size=True, infer_shape=False)
            aspect_lens_iter = aspect_lens_iter.unstack(self.aspect_lens)
            aspect_rep = tf.TensorArray(size=batch_size, dtype=tf.float32)
            aspect_att = tf.TensorArray(size=batch_size, dtype=tf.float32)

            def body(i, aspect_rep, aspect_att):
                a = aspect_outputs_iter.read(i)
                b = context_avg_iter.read(i)
                l = math_ops.to_int32(aspect_lens_iter.read(i))
                aspect_score = tf.reshape(
                    tf.nn.tanh(tf.matmul(tf.matmul(a, weights['aspect_score']), tf.reshape(b, [-1, 1])) + biases['aspect_score']), [1, -1])
                aspect_att_temp = tf.concat([tf.nn.softmax(tf.slice(aspect_score, [0, 0], [1, l])), tf.zeros([1, self.max_aspect_len - l])], 1)
                aspect_att = aspect_att.write(i, aspect_att_temp)
                aspect_rep = aspect_rep.write(i, tf.matmul(aspect_att_temp, a))
                return (i + 1, aspect_rep, aspect_att)

            def condition(i, aspect_rep, aspect_att):
                return i < batch_size

            _, aspect_rep_final, aspect_att_final = tf.while_loop(cond=condition, body=body, loop_vars=(0, aspect_rep, aspect_att))
            self.aspect_atts = tf.reshape(aspect_att_final.stack(), [-1, self.max_aspect_len])
            self.aspect_reps = tf.reshape(aspect_rep_final.stack(), [-1, self.n_hidden])

            context_outputs_iter = tf.TensorArray(tf.float32, 1, dynamic_size=True, infer_shape=False)
            context_outputs_iter = context_outputs_iter.unstack(context_outputs)
            aspect_avg_iter = tf.TensorArray(tf.float32, 1, dynamic_size=True, infer_shape=False)
            aspect_avg_iter = aspect_avg_iter.unstack(aspect_avg)
            context_lens_iter = tf.TensorArray(tf.int32, 1, dynamic_size=True, infer_shape=False)
            context_lens_iter = context_lens_iter.unstack(self.context_lens)
            context_rep = tf.TensorArray(size=batch_size, dtype=tf.float32)
            context_att = tf.TensorArray(size=batch_size, dtype=tf.float32)

            def body(i, context_rep, context_att):
                a = context_outputs_iter.read(i)
                b = aspect_avg_iter.read(i)
                l = math_ops.to_int32(context_lens_iter.read(i))
                context_score = tf.reshape(
                    tf.nn.tanh(tf.matmul(tf.matmul(a, weights['context_score']), tf.reshape(b, [-1, 1])) + biases['context_score']), [1, -1])
                context_att_temp = tf.concat([tf.nn.softmax(tf.slice(context_score, [0, 0], [1, l])), tf.zeros([1, self.max_context_len - l])], 1)
                context_att = context_att.write(i, context_att_temp)
                context_rep = context_rep.write(i, tf.matmul(context_att_temp, a))
                return (i + 1, context_rep, context_att)

            def condition(i, context_rep, context_att):
                return i < batch_size

            _, context_rep_final, context_att_final = tf.while_loop(cond=condition, body=body, loop_vars=(0, context_rep, context_att))
            self.context_atts = tf.reshape(context_att_final.stack(), [-1, self.max_context_len])
            self.context_reps = tf.reshape(context_rep_final.stack(), [-1, self.n_hidden])

            self.reps = tf.concat([self.aspect_reps, self.context_reps], 1)
            self.predict = tf.matmul(self.reps, weights['softmax']) + biases['softmax']
            self.predict_sm = tf.nn.softmax(self.predict)

        with tf.name_scope('loss'):
            self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.predict, labels=self.labels))

            # self.class_w = tf.constant([5.0, 3.0, 5.0, 1])
            # self.cost = -tf.reduce_mean(self.class_w * tf.cast(self.labels, dtype=tf.float32) *tf.log(self.predict_sm))

            self.global_step = tf.Variable(0, name="tr_global_step", trainable=False)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost, global_step=self.global_step)

        with tf.name_scope('predict'):
            self.correct_pred = tf.equal(tf.argmax(self.predict, 1), tf.argmax(self.labels, 1))
            self.accuracy = tf.reduce_sum(tf.cast(self.correct_pred, tf.int32))

        summary_loss = tf.summary.scalar('loss', self.cost)
        summary_acc = tf.summary.scalar('acc', self.accuracy)
        self.train_summary_op = tf.summary.merge([summary_loss, summary_acc])
        self.test_summary_op = tf.summary.merge([summary_loss, summary_acc])
        timestamp = str(int(time.time()))
        _dir = 'logs/' + str(timestamp) + '_r' + str(self.learning_rate) + '_b' + str(self.batch_size) + '_l' + str(self.l2_reg)
        self.train_summary_writer = tf.summary.FileWriter(_dir + '/train', self.sess.graph)
        self.test_summary_writer = tf.summary.FileWriter(_dir + '/test', self.sess.graph)

    def cal_f1(self, labels, predicts):
        '''前三列表示情感，第4列表示不是此分类'''
        labels = np.argmax(labels, axis=1)
        predicts = np.argmax(predicts, axis=1)

        # 判断正确
        tp_flag = predicts < 3
        tp = sum(predicts[tp_flag] == labels[tp_flag])

        # 错判
        fp_flag = predicts < 3
        fp = sum(predicts[fp_flag] != labels[tp_flag])

        # 漏判
        fn = sum((predicts == 3) & (labels < 3))
        return tp, fp, fn

    def train(self, data):
        aspects, contexts, labels, aspect_lens, context_lens = data
        cost, cnt = 0., 0

        for sample, num in self.get_batch_data_balance(aspects, contexts, labels, aspect_lens, context_lens, self.batch_size, True, self.dropout):
            _, loss, step, summary = self.sess.run([self.optimizer, self.cost, self.global_step, self.train_summary_op], feed_dict=sample)
            self.train_summary_writer.add_summary(summary, step)
            cost += loss * num
            cnt += num

        _, train_acc = self.test(data)
        return cost / cnt, train_acc

    def test(self, data):
        aspects, contexts, labels, aspect_lens, context_lens = data
        cost, acc, cnt = 0., 0, 0
        tp, fp, fn = 0, 0, 0
        first = False
        for sample, num in self.get_batch_data(aspects, contexts, labels, aspect_lens, context_lens, self.batch_size, False, 1.0):
            predict, labels, loss, accuracy, step, summary = self.sess.run(
                [self.predict_sm, self.labels, self.cost, self.accuracy, self.global_step, self.test_summary_op], feed_dict=sample)
            if first:
                first = False
                print(predict)
            cost += loss * num
            tp1, fp1, fn1 = self.cal_f1(labels, predict)
            tp += tp1
            fp += fp1
            fn += fn1
            #             acc += accuracy
            cnt += num

        self.test_summary_writer.add_summary(summary, step)

        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * p * r / (p + r)
        print('tp={}, fp={}, fn={}; p={}, r={}, f1={}'.format(tp, fp, fn, p, r, f1))
        return cost / cnt, f1

    def predict_data(self, data):
        aspects, contexts, aspect_lens, context_lens = data
        predicts = []
        for sample, num in self.get_batch_data(aspects, contexts, [], aspect_lens, context_lens, self.batch_size, False, 1.0):
            pred = self.sess.run([self.predict_sm], feed_dict=sample)[0].tolist()
            predicts += pred
        return predicts

    def analysis(self, train_data, test_data):
        timestamp = str(int(time.time()))

        aspects, contexts, labels, aspect_lens, context_lens = train_data
        with open('analysis/train_' + str(timestamp) + '.txt', 'w') as f:
            for sample, num in self.get_batch_data(aspects, contexts, labels, aspect_lens, context_lens, len(aspects), False, 1.0):
                aspect_atts, context_atts, correct_pred = self.sess.run([self.aspect_atts, self.context_atts, self.correct_pred], feed_dict=sample)
                for a, b, c in zip(aspect_atts, context_atts, correct_pred):
                    a = str(a).replace('\n', '')
                    b = str(b).replace('\n', '')
                    f.write('%s\n%s\n%s\n' % (a, b, c))
        print('Finishing analyzing training data')

        aspects, contexts, labels, aspect_lens, context_lens = test_data
        with open('analysis/test_' + str(timestamp) + '.txt', 'w') as f:
            for sample, num in self.get_batch_data(aspects, contexts, labels, aspect_lens, context_lens, len(aspects), False, 1.0):
                aspect_atts, context_atts, correct_pred = self.sess.run([self.aspect_atts, self.context_atts, self.correct_pred], feed_dict=sample)
                for a, b, c in zip(aspect_atts, context_atts, correct_pred):
                    a = str(a).replace('\n', '')
                    b = str(b).replace('\n', '')
                    f.write('%s\n%s\n%s\n' % (a, b, c))
        print('Finishing analyzing testing data')

    def run(self, train_data, test_data, is_analyzing=False, is_early_stop=False):
        saver = tf.train.Saver(tf.trainable_variables())
        print('Training ...')
        self.sess.run(tf.global_variables_initializer())
        max_acc, step = 0., -1
        crt_tol = 0
        for i in range(self.n_epoch):
            train_loss, train_acc = self.train(train_data)
            test_loss, test_acc = self.test(test_data)
            print('>>>>>>>>>> epoch %s: train-loss=%.6f; train-f1=%.6f; test-loss=%.6f; test-f1=%.6f;' % (
            str(i), train_loss, train_acc, test_loss, test_acc))
            if test_acc > max_acc:
                max_acc = test_acc
                step = i
                crt_tol = 0
                saver.save(self.sess, 'models/model_iter', global_step=step)
            else:
                if (crt_tol < self.early_stop):
                    crt_tol += 1
                else:
                    break
        saver.save(self.sess, 'models/model_final')
        print('The max accuracy of testing results is %s of step %s' % (max_acc, step))

        if (is_analyzing):
            print('Analyzing ...')
            saver.restore(self.sess, tf.train.latest_checkpoint('models/'))
            self.analysis(train_data, test_data)

    def get_batch_data(self, aspects, contexts, labels, aspect_lens, context_lens, batch_size, is_shuffle, keep_prob):
        for index in tqdm_notebook(get_batch_index(len(aspects), batch_size, is_shuffle), total=int(len(context_lens) / batch_size) + 1):
            if len(labels) <= 0:
                feed_dict = {
                    self.aspects: aspects[index],
                    self.contexts: contexts[index],
                    self.aspect_lens: aspect_lens[index],
                    self.context_lens: context_lens[index],
                    self.dropout_keep_prob: keep_prob,
                }
            else:
                feed_dict = {
                    self.aspects: aspects[index],
                    self.contexts: contexts[index],
                    self.labels: labels[index],
                    self.aspect_lens: aspect_lens[index],
                    self.context_lens: context_lens[index],
                    self.dropout_keep_prob: keep_prob,
                }
            yield feed_dict, len(index)

    def get_batch_data_balance(self, aspects, contexts, labels, aspect_lens, context_lens, batch_size, is_shuffle, keep_prob):
        index = np.arange(len(aspect_lens))
        cate_idxes = []
        for i in range(4):
            cate_idxes.append(index[labels[:, i] == 1].tolist())
            random.shuffle(cate_idxes[-1])

        every_cate_bs_num = int(batch_size / 9)
        no_cate_num = every_cate_bs_num * 6
        for i in tqdm_notebook(range(int(len(cate_idxes[3]) / no_cate_num) + 1)):
            index = cate_idxes[3][i * no_cate_num: min(len(cate_idxes[3]), no_cate_num * (i + 1))]
            for i in range(3):
                index += random.sample(cate_idxes[i], every_cate_bs_num)
            random.shuffle(index)

            if len(labels) <= 0:
                feed_dict = {
                    self.aspects: aspects[index],
                    self.contexts: contexts[index],
                    self.aspect_lens: aspect_lens[index],
                    self.context_lens: context_lens[index],
                    self.dropout_keep_prob: keep_prob,
                }
            else:
                feed_dict = {
                    self.aspects: aspects[index],
                    self.contexts: contexts[index],
                    self.labels: labels[index],
                    self.aspect_lens: aspect_lens[index],
                    self.context_lens: context_lens[index],
                    self.dropout_keep_prob: keep_prob,
                }
            yield feed_dict, len(index)
