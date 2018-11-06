#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

import re
import sys
import json
import time
import jieba

jieba.load_userdict("word_tencent")
import logging
import pandas as pd
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

all_segments = set()

train_len = 2000000
val_len = 50000
test_len = 50000
extra_len = 58


def get_segment(text, tag):
    segs = jieba.cut(text)
    return "|".join(segs)


def move_useless_char(s):
    return re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+??！，。？?、~@#￥%……&*（）]+", "", s)


def parse_query_prediction(query_prediction, data_tag):
    if query_prediction == "":
        return 0, "\t".join(["PAD"] * 10), "\t".join(['0.000'] * 10)

    json_data = json.loads(query_prediction)
    result = sorted(json_data.items(), key=lambda d: d[1], reverse=True)
    texts = [get_segment(move_useless_char(item[0]), data_tag) for item in result]
    scores = [item[1] for item in result]

    n = len(texts)
    return n, "\t".join(texts + ["PAD"] * (10 - n)), "\t".join(scores + ['0.000'] * (10 - n))


def load_data(input_file, extra_file, output_file, data_tag):
    extra_file = pd.read_csv(extra_file)
    offset = 0
    if data_tag == "train":
        extra_file = extra_file.iloc[:train_len, :]
    elif data_tag == "valid":
        extra_file = extra_file.iloc[train_len:train_len+val_len, :]
    else:
        extra_file = extra_file.iloc[train_len+val_len:, :]
    with open(output_file, 'w') as fo:
        fo.write("\t".join([
            "prefix", "title", "tag", "label", "flag", "num",
            "\t".join(["text_" + str(i + 1) for i in range(10)]),
            "\t".join(["score_" + str(i + 1) for i in range(10)]),
            "\t".join(["extra_" + str(i + 1) for i in range(extra_len)])
        ]))
        fo.write("\n")

        with open(input_file, 'r', encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if data_tag == "test":
                    prefix, query_prediction, title, tag = line.split("\t")
                    label = "-1"
                else:
                    prefix, query_prediction, title, tag, label = line.split("\t")

                prefix = move_useless_char(prefix)
                title = move_useless_char(title)

                n, prediction_text, prediction_score = parse_query_prediction(query_prediction, data_tag)
                fo.write("\t".join([
                    get_segment(prefix, data_tag), get_segment(title, data_tag), tag, label, data_tag, str(n), prediction_text, prediction_score
                ] + [str(i) for i in extra_file.iloc[offset].tolist()]))
                offset += 1
                fo.write("\n")


def main():
    data_type = "test_A"
    ORI_TRAIN_DATA_FILE = "../../../../data/new/oppo_round1_train_20180929.txt"
    ORI_VALID_DATA_FILE = "../../../../data/new/oppo_round1_vali_20180929.txt"
    ORI_TEST_DATA_FILE = "../../../../data/new/oppo_round1_test_A_20180929.txt"
    ORI_TEST_B_DATA_FILE = "../../../../data/new/oppo_round1_test_A_20180929.txt"
    EXTRA_FEATURE_FILE = "../../../../feature/df_goodfeatures55.csv"
    TRAIN_DATA_FILE = "../../../../data/new/train_preprocess.txt"
    VALID_DATA_FILE = "../../../../data/new/valid_preprocess.txt"
    TEST_DATA_FILE = "../../../../data/new/test_preprocess.txt"

    # TRAIN_DATA = "../../../../feature/rnn_train.txt"
    # PREDICT_DATA = "../../../../feature/rnn_predict.txt"
    TRAIN_DATA = "./data/train(extra).txt"
    PREDICT_DATA = "./data/predict(extra).txt"

    time_point = time.time()
    if data_type == "test_A":
        logging.info("load train data ...")
        load_data(ORI_TRAIN_DATA_FILE, EXTRA_FEATURE_FILE, TRAIN_DATA_FILE, "train")
        logging.info("load train data cost time: " + str(time.time() - time_point))

        logging.info("load valid data ...")
        time_point = time.time()
        load_data(ORI_VALID_DATA_FILE, EXTRA_FEATURE_FILE, VALID_DATA_FILE, "valid")
        logging.info("load valid data cost time: " + str(time.time() - time_point))

        logging.info("load test data ...")
        time_point = time.time()
        load_data(ORI_TEST_DATA_FILE, EXTRA_FEATURE_FILE, TEST_DATA_FILE, "test")
        logging.info("load test data cost time: " + str(time.time() - time_point))

        logging.info("get word vector cost time: " + str(time.time() - time_point))

    elif data_type == "test_B":
        logging.info("load test B data ...")
        time_point = time.time()
        load_data(ORI_TEST_B_DATA_FILE, TEST_B_DATA_FILE, "test")
        logging.info("load test B data cost time: " + str(time.time() - time_point))

    else:
        logging.error("wrong data_type!")

    train_data = pd.read_csv(TRAIN_DATA_FILE, sep="\t", encoding="utf-8", low_memory=False)
    valid_data = pd.read_csv(VALID_DATA_FILE, sep="\t", encoding="utf-8", low_memory=False)
    if data_type == "test_A":
        test_data = pd.read_csv(TEST_DATA_FILE, sep="\t", encoding="utf-8", low_memory=False)
    elif data_type == "test_B":
        test_data = pd.read_csv(TEST_B_DATA_FILE, sep="\t", encoding="utf-8", low_memory=False)
    else:
        logging.error("wrong data_type!")

    # 完整数据
    train_data = pd.concat([train_data, valid_data])
    train_data.to_csv(TRAIN_DATA, sep="\t", index=False, encoding="utf-8")
    pred_data = test_data
    pred_data.to_csv(PREDICT_DATA, sep="\t", index=False, encoding="utf-8")
    # 测试数据
    train_data = pd.concat([train_data, valid_data])
    train_data.iloc[:10000, :].to_csv(TRAIN_DATA, sep="\t", index=False, encoding="utf-8")
    train_data.iloc[10000:15000, :].to_csv(PREDICT_DATA, sep="\t", index=False, encoding="utf-8")

    logging.info("done!")


if __name__ == "__main__":
    main()
