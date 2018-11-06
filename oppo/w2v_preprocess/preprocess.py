import pandas as pd
import numpy as np
from urllib import parse
import gensim
import gc
from tqdm import tqdm_notebook as tqdm
from langconv import *

def Traditional2Simplified(sentence):
    '''
    将sentence中的繁体字转为简体字
    :param sentence: 待转换的句子
    :return: 将句子中繁体字转换为简体字之后的句子
    '''
    sentence = Converter('zh-hans').convert(sentence)
    return sentence

df_train = pd.read_csv('../data/oppo_round1_train_20180929.txt', sep='\t', encoding='utf-8', header=None,names=['prefix', 'query_prediction', 'title', 'tag', 'label'],keep_default_na=False)
df_val = pd.read_csv('../data/oppo_round1_vali_20180929.txt',sep='\t', encoding='utf-8', header=None,names=['prefix', 'query_prediction', 'title', 'tag', 'label'],keep_default_na=False)
df_test = pd.read_csv('../data/oppo_round1_test_A_20180929.txt',sep='\t', encoding='utf-8', header=None,names=['prefix', 'query_prediction', 'title', 'tag', 'label'],keep_default_na=False)

df_train['dataset'] = 0
df_val['dataset'] = 1
df_test['dataset'] = 2
df_test['label'] = -1
df_all = pd.concat([df_train, df_val, df_test]).reset_index(drop=True)
del df_train, df_val, df_test

df_all.prefix = df_all.prefix.map(Traditional2Simplified)
df_all.query_prediction = df_all.query_prediction.map(Traditional2Simplified)
df_all.title = df_all.title.map(Traditional2Simplified)
remove_chars = ['\u200e','\u3000','\ufeff', '\xa0','\x91', '\x98', ' ']
for c in remove_chars:
    df_all.prefix = df_all.prefix.map(lambda x: x.replace(c, ''))
    df_all.query_prediction = df_all.query_prediction.map(lambda x: x.replace(c, ''))
    df_all.title = df_all.title.map(lambda x: x.replace(c, ''))

df_all.prefix = df_all.prefix.map(lambda x:parse.unquote(x))
df_all.title = df_all.title.map(lambda x: parse.unquote(x))
df_all.query_prediction = df_all.query_prediction.map(lambda x:eval(parse.unquote(x)))
df_all.query_prediction = df_all.query_prediction.map(lambda x:{each:float(x[each]) for each in x})
