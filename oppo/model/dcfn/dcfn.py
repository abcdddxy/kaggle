import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.metrics import f1_score, recall_score, precision_score
import math
import logging
from scipy.sparse.csr import csr_matrix
from scipy.sparse import hstack, vstack
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def MLP(inp, hidden_dims):
    x = tf.layers.Dense(hidden_dims[0], kernel_initializer=tf.keras.initializers.he_normal(), dtype=tf.float32, activation=tf.nn.leaky_relu)(inp)
    x = tf.layers.BatchNormalization(dtype=tf.float32)(x)
    x = tf.nn.leaky_relu(x)
    for i, dim in enumerate(hidden_dims):
        if i > 0:
            x = tf.layers.Dense(dim, kernel_initializer=tf.keras.initializers.he_normal(), dtype=tf.float32, activation=tf.nn.leaky_relu)(x)
            x = tf.layers.BatchNormalization(dtype=tf.float32)(x)
            x = tf.nn.leaky_relu(x)
    return x

class DCFN:
    def __init__(self, learning_rate, embedding_size, dnn_layers, att_layer, cross_layer_num, conti_fea_cnt,
                 cate_embedding_uni_cnt_list, cate_embedding_w_list=None, fm_embedding_w=None, no_nan_w=None,
                 nan_w=None, fm_drop_outs=[1, 1], result_weight=0.5):
        self.lr = learning_rate
        self.conti_fea_cnt = conti_fea_cnt
        self.embedding_size = embedding_size
        self.fm_drop_outs = fm_drop_outs
        self.dnn_layers = dnn_layers
        self.att_layer = att_layer
        self.cross_layer_num = cross_layer_num
        # cate_embedding_uni_cnt_list离散特征计数
        self.cate_embedding_uni_cnt_list = cate_embedding_uni_cnt_list
        self.cate_embedding_w_list = cate_embedding_w_list

        self.fm_embedding_w = fm_embedding_w
        self.no_nan_w = no_nan_w
        self.nan_w = nan_w

        self.result_weight = result_weight
        self.build()

    def build(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)

            self.input_vecs = []

            self.conti_vec = tf.placeholder(tf.float32, shape=[None, self.conti_fea_cnt], name='conti_vec')
            self.cate_indexs = tf.placeholder(tf.int16, shape=[None, sum(self.cate_embedding_uni_cnt_list)],
                                              name='cate_indexs')
            self.label = tf.placeholder(dtype=tf.int8, shape=[None, 1], name='label')

            self.cate_embeddings = []
            self.fm_fea_size = 0

            # 第一层embedding：降维
            cate_offset = 0
            for cate_idx, uni_cnt in enumerate(self.cate_embedding_uni_cnt_list):
                w = self.cate_embedding_w_list[cate_idx] if self.cate_embedding_w_list else tf.keras.initializers.he_normal()
                embedding_k = uni_cnt if int(2 * np.power(uni_cnt, 1 / 4)) > uni_cnt else int(
                    2 * np.power(uni_cnt, 1 / 4))
                self.fm_fea_size += embedding_k
                print('embedding K:{} -> {}'.format(uni_cnt, embedding_k))
                # embedding矩阵
                self.cate_embeddings.append(
                    tf.get_variable('cate_%d_embedding' % cate_idx, shape=[uni_cnt, embedding_k], dtype=tf.float32,
                                    initializer=w))

                crt_vec_index = self.cate_indexs[:, cate_offset:cate_offset + uni_cnt]  # None * uni_cnt
                cate_offset += uni_cnt
                crt_vec_index = tf.Print(crt_vec_index, [crt_vec_index], message='Debug:', summarize=50)

                crt_vec = tf.nn.embedding_lookup(self.cate_embeddings[cate_idx],
                                                 [i for i in range(uni_cnt)])  # uni_cnt * K
                # 等于1的加起来，求平均（embedding相当于多行相加，multi-hot要除1的个数保证一致）
                crt_vec = tf.matmul(tf.cast(crt_vec_index, tf.float32), crt_vec)  # None * K
                one_cnt = tf.cast(tf.reduce_sum(crt_vec_index, axis=1, keep_dims=True), dtype=tf.float32)  # None * 1
                crt_vec = tf.div(crt_vec, one_cnt)  # None * K
                self.input_vecs.append(crt_vec)

            mv_conti_vec = self.conti_vec
            with tf.variable_scope('Missing-Value-Layer'):
                self.no_nan_w = tf.get_variable('no_nan_w', shape=[self.conti_fea_cnt, ],
                                                initializer=self.no_nan_w if self.no_nan_w else tf.ones_initializer())
                self.nan_w = tf.get_variable('nan_w', shape=[self.conti_fea_cnt, ],
                                                         initializer=self.nan_w if self.nan_w else tf.zeros_initializer())
                mv_conti_vec = tf.multiply(self.conti_vec, self.no_nan_w)
                conti_zero_flag = tf.cast(tf.equal(mv_conti_vec, 0), tf.float32)
                mv_conti_vec += tf.multiply(conti_zero_flag, tf.reshape(self.nan_w, [-1, self.nan_w.shape[0]]))

            self.input_vecs.append(mv_conti_vec)
            self.fm_fea_size += self.conti_fea_cnt

            # 准备输入-----------------------------------------------------------------------------------------------------
            fm_fea = tf.concat(self.input_vecs, axis=-1)

            self.feat_index = [i for i in range(self.fm_fea_size)]
            if self.fm_embedding_w is not None:
                self.fea_embedding = tf.Variable(self.fm_embedding_w, name='fea_embedding', dtype=tf.float32)
            else:
                self.fea_embedding = tf.get_variable('fea_embedding', shape=[self.fm_fea_size, self.embedding_size],
                                                     initializer=tf.keras.initializers.he_normal(), dtype=tf.float32)
            # FM一阶部分权重
            self.feature_bias = tf.get_variable('fea_bias', shape=[self.fm_fea_size, 1],
                                                initializer=tf.keras.initializers.he_normal(), dtype=tf.float32)
            # attention部分权重
            self.attention_h = tf.Variable(np.random.normal(loc=0, scale=1, size=[self.att_layer,]), 
                                           dtype=np.float32, name='attention_h')
            self.attention_p = tf.Variable(np.ones([self.embedding_size, ], dtype=np.float32), 
                                           dtype=tf.float32, name='attention_p')
            # cross部分权重
            self.cross_w = [tf.get_variable(name='cross_weight_%d' % i, shape=[self.fm_fea_size, 1],
                                            initializer=tf.keras.initializers.he_normal(), dtype=tf.float32) for i in
                            range(self.cross_layer_num)]
            self.cross_b = [tf.get_variable(name='cross_bias_%d' % i, shape=[self.fm_fea_size, 1],
                                            initializer=tf.keras.initializers.he_normal(), dtype=tf.float32) for i in
                            range(self.cross_layer_num)]

            # 构造输入
            # 第二层embedding：潜在隐变量
            embeddings = tf.nn.embedding_lookup(self.fea_embedding, self.feat_index)  # None * F * K
            feat_value = tf.reshape(fm_fea, shape=[-1, self.fm_fea_size, 1])
            embeddings = tf.multiply(embeddings, feat_value)  # None * F * K
#             print(embeddings)
#             embeddings = tf.Print(embeddings, [embeddings], message='Debug:', summarize=30)

            # 搭建网络-----------------------------------------------------------------------------------------------------
            # FM部分
            with tf.variable_scope('FM-part'):
                # first order term:输入为原始sparse features
                y_first_order = tf.nn.embedding_lookup(self.feature_bias, self.feat_index)  # None * F * 1
                y_first_order = tf.reduce_sum(tf.multiply(y_first_order, feat_value), 2)  # None * F(对1、2维求和都可以)
                y_first_order = tf.nn.dropout(y_first_order, self.fm_drop_outs[0])  # None * F
                # second order term:输入为dense embedding
                summed_features_emb = tf.reduce_sum(embeddings, 1)  # None * K
                summed_features_emb_square = tf.square(summed_features_emb)  # None * K
                squared_features_emb = tf.square(embeddings)
                squared_sum_features_emb = tf.reduce_sum(squared_features_emb, 1)  # None * K
                y_second_order = 0.5 * tf.subtract(summed_features_emb_square, squared_sum_features_emb)  # None * K
                y_second_order = tf.nn.dropout(y_second_order, self.fm_drop_outs[1])  # None * K
#                 # second order term:加入attention
#                 # Pair-wise Interation Layer
#                 element_wise_product = None
#                 for i in range(0, self.fm_fea_size):
#                     if element_wise_product is None:
#                         element_wise_product = tf.multiply(tf.gather(embeddings, [i], axis=1), 
#                                                            embeddings[:, i+1:self.fm_fea_size, :])
#                     else:
#                         element_wise_product = tf.concat([element_wise_product,
#                                                          tf.multiply(tf.gather(embeddings, [i], axis=1), 
#                                                                      embeddings[:, i+1:self.fm_fea_size, :])],
#                                                          axis=1) # None * (F*(F-1))/2 * K
#                 # Attention-based Pooling Layer
#                 attention_mul = tf.layers.Dense(self.att_layer)(element_wise_product)
#                 attention_exp = tf.exp(tf.reduce_sum(tf.multiply(self.attention_h, tf.nn.relu(attention_mul)),
#                                                      2, keep_dims=True))  # None * (H*(H-1)) * 1
#                 attention_sum = tf.reduce_sum(attention_exp, 1, keep_dims=True)  # None * 1 * 1
#                 attention_out = tf.div(attention_exp, attention_sum, name='attention_out')  #  None * (H*(H-1)) * 1
#                 y_second_order = tf.reduce_sum(tf.multiply(attention_out, element_wise_product), 1, name='afm')  # None * K
#                 y_second_order= tf.multiply(y_second_order, self.attention_p)  # None * K
#                 y_second_order = tf.nn.dropout(y_second_order, self.fm_drop_outs[1])  # None * K
    
            # Cross Layer部分
            with tf.variable_scope('Cross-part'):
                x_0 = feat_value
                x_l = x_0
                for l in range(self.cross_layer_num):
                    x_l = tf.tensordot(tf.matmul(x_0, x_l, transpose_b=True), self.cross_w[l], 1) + self.cross_b[
                        l] + x_l
                cross_output = tf.reshape(x_l, shape=[-1, self.fm_fea_size])

            # DNN部分
            with tf.variable_scope('Deep-part'):
                y_deep = tf.reshape(embeddings, shape=[-1, self.fm_fea_size * self.embedding_size])  # None*(F*K)
                y_deep = MLP(y_deep, self.dnn_layers)

                # 合并
            print('y_deep:{},\n cross_output:{},\n y_first_order:{},\n y_second_order:{}'
                  .format(y_deep, cross_output, y_first_order, y_second_order))
#             last_input = tf.concat([y_deep], axis=-1) # DNN
#             last_input = tf.concat([y_first_order, y_second_order], axis=-1) # FM
            last_input = tf.concat([y_deep, y_first_order, y_second_order], axis=-1) # DeepFM
#             last_input = tf.concat([y_deep, cross_output], axis=-1) # DCN
#             last_input = tf.concat([y_deep, y_first_order, y_second_order, cross_output], axis=-1)  # DCFN

            self.y_pre = tf.layers.Dense(1, activation=tf.nn.sigmoid, kernel_initializer=tf.keras.initializers.he_normal())(
                last_input)  # 二分类
    
            # 损失函数(二分类交叉熵等同于logloss)
            self.loss = tf.losses.log_loss(self.label, self.y_pre)  # 二分类
            
            # 优化方法
            self.opt = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
            self.saver = tf.train.Saver()

    def save_model(self, model_path):
        self.saver.save(self.sess, model_path)

    def load_model(self, model_path):
        self.saver.restore(self.sess, model_path)

    def shuffle_csr_and_list(self, my_array, rng_state):
        np.random.set_state(rng_state)
        if type(my_array) == csr_matrix:
            index = np.arange(np.shape(my_array)[0])
            np.random.shuffle(index)
            print('shuffle csr_matrix ' + str(my_array.shape))
            return my_array[index, :]
        else:
            np.random.shuffle(my_array)
            return my_array

    def shuffle(self, cate_feas, conti_feas, labels):
        rng_state = np.random.get_state()
        cate_feas = self.shuffle_csr_and_list(cate_feas, rng_state)
        conti_feas = self.shuffle_csr_and_list(conti_feas, rng_state)
        labels = self.shuffle_csr_and_list(labels, rng_state)
        return cate_feas, conti_feas, labels

    def get_feed_dict(self, cate_feas, conti_feas, labels=None):
        feed_dict = {
            self.conti_vec: conti_feas,
            self.cate_indexs: cate_feas.todense(),
        }
        if labels is not None:
            feed_dict[self.label] = labels
        return feed_dict

    def gene_data(self, cate_feas, conti_feas, labels, bs, shuffle=False):
        if shuffle:
            cate_feas, conti_feas, labels = self.shuffle(cate_feas, conti_feas, labels)
        bm = math.ceil(cate_feas.shape[0] / bs)
        for j in range(bm):
            a = cate_feas[j * bs:(j + 1) * bs]
            b = conti_feas[j * bs:(j + 1) * bs]
            c = labels[j * bs:(j + 1) * bs]
            yield a, b, c

    def gene_balance_data(self, cate_feas, conti_feas, labels, bs, shuffle=False):
        pos_flag = np.array([l[0] == 1 for l in labels])
        pos_indexing, neg_indexing = np.arange(len(labels))[pos_flag], np.arange(len(labels))[~pos_flag]
        np.random.shuffle(neg_indexing)

        bm = math.ceil(sum(~pos_flag) / bs)
        for j in range(bm):
            need_cnt = int(bs / 2)
            crt_indexing = np.random.choice(pos_indexing, need_cnt).tolist() + neg_indexing[
                                                                               j * need_cnt:(j + 1) * need_cnt].tolist()

            a = cate_feas[crt_indexing, :]
            b = np.take(conti_feas, crt_indexing, axis=0)
            c = np.take(labels, crt_indexing, axis=0)
            yield a, b, c

    def fit(self, model_path, batch_size, epoch, cate_feas, conti_feas, labels, v_cate_feas, v_conti_feas, v_labels,
            es=5):
        print('start training ---------------------------------------------------')
        logging.info('start train')
        with self.graph.as_default():
            self.sess.run(tf.global_variables_initializer())
            best_f1 = 0.0
            no_num = 0
            writer = tf.summary.FileWriter('./logs', self.sess.graph)
            for i in range(epoch):
                t1 = time()
                epoch_losses = []
                for cate_feas_batch, conti_feas_batch, labels_batch in self.gene_data(cate_feas, conti_feas,
                                                                                      labels, batch_size,
                                                                                      shuffle=False):
                    feed = self.get_feed_dict(cate_feas_batch, conti_feas_batch, labels_batch)
                    loss, _ = self.sess.run([self.loss, self.opt], feed_dict=feed)
                    epoch_losses.append(loss)

                v_loss, v_f1, recall, precision = self.eval(batch_size, v_cate_feas, v_conti_feas, v_labels)
                t_loss = np.mean(np.array(epoch_losses))
                logging.info('epoch: %s---train loss %.4f---valid loss: %.4f---valid f1: %.4f'
                             % ((i + 1), t_loss, v_loss, v_f1))
                print('epoch: %s---train loss %.4f---valid loss: %.4f---valid f1: %.4f\n recall: %.4f---precision: %.4f [%.1f s]'
                      % ((i + 1), t_loss, v_loss, v_f1, recall, precision, time() - t1))
                if v_f1 > best_f1:
                    no_num = 0
                    self.save_model(model_path)
                    logging.info('---------- f1 from %.4f to %.4f, saving model' % (best_f1, v_f1))
                    print('---------- f1 from %.4f to %.4f, saving model' % (best_f1, v_f1))
                    best_f1 = v_f1
                else:
                    no_num += 1
                    self.lr = self.lr / 5
                    if no_num >= es:
                        break

    def eval(self, batch_size, cate_feas, conti_feas, labels):
        with self.graph.as_default():
            y_pre = []
            for cate_feas_batch, conti_feas_batch, label_batch in self.gene_data(cate_feas, conti_feas, labels,
                                                                                 batch_size, shuffle=False):
                feed = self.get_feed_dict(cate_feas_batch, conti_feas_batch, label_batch)
                y_ = self.sess.run([self.y_pre], feed_dict=feed)[0]
                y_pre += y_.tolist()
            y_pre = np.array(y_pre)
            y_pre = np.reshape(y_pre, (y_pre.shape[0],))
            labels = np.reshape(labels, (labels.shape[0],))
            loss = log_loss(labels, y_pre)
            y_pre = (y_pre > self.result_weight).astype(int)
            f1 = f1_score(labels, y_pre)
            recall = recall_score(labels, y_pre)
            precision = precision_score(labels, y_pre)
            return loss, f1, recall, precision

    def predict(self, cate_feas, conti_feas, batch_size):
        def gd(cate_feas, conti_feas, bs):
            bm = math.ceil(len(conti_feas) / bs)
            for j in range(bm):
                a = cate_feas[j * bs: (j + 1) * bs]
                b = conti_feas[j * bs: (j + 1) * bs]
                yield a, b

        with self.graph.as_default():
            y_pre = []
            for cate_feas_batch, conti_feas_batch in gd(cate_feas, conti_feas, batch_size):
                feed = self.get_feed_dict(cate_feas_batch, conti_feas_batch)
                y_ = self.sess.run([self.y_pre], feed_dict=feed)[0]
                y_pre += y_.tolist()
            y_pre = np.array(y_pre)
            y_pre = np.reshape(y_pre, (y_pre.shape[0],))
            return y_pre

    def embedding_weights(self):
        cate_embeddings, fea_embedding = self.sess.run([self.cate_embeddings, self.fea_embedding])
        return cate_embeddings, fea_embedding

    def miss_value_layer_w(self):
        nan_embeddings, no_nan_embedding = self.sess.run([self.nan_w, self.no_nan_w])
        return nan_embeddings, no_nan_embedding