import os
import ast
import pandas as pd
import numpy as np
from errno import ENOENT
from collections import Counter

import jieba
jieba.load_userdict('../data/cars.txt')

def get_batch_index(length, batch_size, is_shuffle=True):
    index = list(range(length))
    if is_shuffle:
        np.random.shuffle(index)
    for i in range(int(length / batch_size) + (1 if length % batch_size else 0)):
        yield index[i * batch_size:(i + 1) * batch_size]

def get_stop_word_set():
    words_set = set()
    with open('/home/cjy/extract_tag/data/哈工大停用词表扩展.txt') as f_r:
        for line in f_r:
            words_set |= set(line.strip())
#     print('stop words cnt:', len(words_set))
    return words_set

def read_vectors(path, topn):  # read top n word vectors, i.e. top is 10000
    lines_num, dim = 0, 0
    vectors = {}
    iw = []
    wi = {}
    with open(path, encoding='utf-8', errors='ignore') as f:
        first_line = True
        for line in f:
            if first_line:
                first_line = False
                dim = int(line.rstrip().split()[1])
                continue
            lines_num += 1
            tokens = line.rstrip().split(' ')
            vectors[tokens[0]] = np.asarray([float(x) for x in tokens[1:]])
            iw.append(tokens[0])
            if topn != 0 and lines_num >= topn:
                break
    for i, w in enumerate(iw):
        wi[w] = i
    return vectors, iw, wi, dim

def load_word_embeddings(word2id, is_64_dim=True):
    if is_64_dim:
        from gensim.models import KeyedVectors
        com_w2v = KeyedVectors.load_word2vec_format('/home/cjy/extract_tag/model/news_12g_baidubaike_20g_novel_90g_embedding_64.bin', 
                                                    binary=True)
        com_w2v_embedding_dim = 64
    else:
        com_w2v, _, _, com_w2v_embedding_dim = read_vectors('/home/cjy/pretrain_word2vec/data/merge_sgns_bigram_char300.txt', 0)
    
    word2vec = np.random.uniform(-0.01, 0.01, [len(word2id), com_w2v_embedding_dim])
    for w, w_id in word2id.items():
        if w in com_w2v:
            word2vec[word2id[w]] = com_w2v[w]
    word2vec[word2id['<pad>'], :] = 0
    return word2vec, com_w2v_embedding_dim

def get_word2id(data_path, all_subjects, pre_processed, train_fname, test_fname, filter_stop_ws=True):
    '''构造 word id 映射'''
    if filter_stop_ws:
        save_fname = data_path + 'word2id_map_without_stopword.txt'
    else:
        save_fname = data_path + 'word2id_map.txt'

    word2id = {}
    max_len = 0
    if pre_processed:
        with open(save_fname) as f_r:
            for line in f_r:
                tmp = line[:-1].split(' ')
                word2id[tmp[0]] = int(tmp[1])
    else:
        if filter_stop_ws:
            stop_words = get_stop_word_set()
        word2id['<pad>'] = 0
        for s in all_subjects:
            word2id[s] = len(word2id)
        train_fname = data_path + train_fname + '.csv'
        test_fname = data_path + test_fname + '.csv'
        
        data = pd.read_csv(train_fname)
        data = pd.concat([data, pd.read_csv(test_fname)])
        data = data.drop_duplicates('content_id')
        for content_id, content in data[['content_id', 'content']].values:
            content = content.strip()
            crt_len = 0
            for word in jieba.cut(content):
                crt_len += 1
                if word not in word2id and (not filter_stop_ws or word not in stop_words) and len(word.strip())>0:
                    word2id[word] = len(word2id)
            max_len = max(crt_len, max_len)
        
        with open(save_fname, 'w') as fsave:
            for item in sorted(word2id.items(), key=lambda x:x[1]):
                fsave.write(item[0]+' '+str(item[1])+'\n')
    return word2id, max_len