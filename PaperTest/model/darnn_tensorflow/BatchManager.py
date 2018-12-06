# coding: utf-8


import math
import numpy as np
import random


class BatchManager(object):
    def __new__(cls, data, epoch, params):
        manager = object.__new__(cls)
        manager.data = data
        manager.len_data = len(manager.data["data"])
        manager.keyword = params["keyword"]
        manager.batch_size = params["batch_size"]
        manager.T = params["T"]
        manager.input_size = params["input_size"]
        manager.shuffle = params["shuffle"]
        manager.num_batch = manager.len_data // manager.batch_size
        manager.cur_batch = 0
        manager.epoch = epoch
        manager._epoch = epoch
        manager.index = list(range(manager.len_data))
        # no shuffle padding before training
        manager.batch_data_noshuffle = [
            manager.data["data"][ibatch * manager.batch_size:min((ibatch + 1) * manager.batch_size, manager.len_data)]
            for ibatch in range(manager.num_batch)
        ]
        manager.batch_data_shuffle = manager.data["data"]
        manager.target = manager.data["target"]
        return manager

    def _batch_noshuffle(self):
        batch = self.batch_data_noshuffle[self.cur_batch]
        self.cur_batch += 1
        if self.cur_batch >= self.num_batch:
            self.cur_batch = 0
            self.epoch -= 1
        return batch, len(batch)

    def _batch_shuffle(self):
        if self.keyword == "train":
            # T - 1
            perm_idx = np.random.permutation(self.len_data - self.T)
            batch_idx = perm_idx[self.cur_batch * self.batch_size:(self.cur_batch + 1) * self.batch_size]
            X = np.zeros((len(batch_idx), self.T, self.input_size))
            y_history = np.zeros((len(batch_idx), self.T))
            y_target = self.target[batch_idx + self.T]

            for k in range(len(batch_idx)):
                X[k, :, :] = self.batch_data_shuffle[batch_idx[k]: (batch_idx[k] + self.T), :]
                y_history[k, :] = self.target[batch_idx[k]: (batch_idx[k] + self.T)]
            batch = [X, y_history, y_target]
        else:
            batch_idx = np.array(range(self.len_data))[self.cur_batch * self.batch_size:(self.cur_batch + 1) * self.batch_size]
            X = np.zeros((len(batch_idx), self.T - 1, self.input_size))
            y_history = np.zeros((len(batch_idx), self.T - 1))
            for j in range(len(batch_idx)):
                X[j, :, :] = self.batch_data_shuffle[range(batch_idx[j] - self.T, batch_idx[j] - 1), :]
                y_history[j, :] = self.target[range(batch_idx[j] - self.T, batch_idx[j] - 1)]
            batch = [X, y_history]

        self.cur_batch += 1
        if self.cur_batch >= self.num_batch:
            self.cur_batch = 0
            self.epoch -= 1
        return batch, self.batch_size

    def init(self):
        self.epoch = self._epoch
        self.cur_batch = 0
        random.shuffle(self.index)

    @property
    def is_finished(self):
        return self.epoch <= 0

    def batch(self):
        if self.epoch <= 0:
            raise EOFError("epoch exhausted.")
        batch = self._batch_shuffle()
        return batch
