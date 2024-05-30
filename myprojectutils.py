import torch
import torch.nn as nn
import numpy as np

import torch.distributed as dist


class MyUtils(object):
    @staticmethod
    def init_process_group(world_size, rank):
        dist.init_process_group(
            backend="gloo",  # change to 'nccl' for multiple GPUs
            init_method="tcp://127.0.0.1:12345",
            world_size=world_size,
            rank=rank,
        )

    @staticmethod
    def get_feature(batchg):
        return torch.cat(
            (batchg.ndata["instruction_type"],),
            1,
        )

    @staticmethod
    def try_gpu(i=0):
        if i < 0:
            return torch.device("cpu")
        if torch.cuda.device_count() >= i + 1:
            return torch.device(f"cuda:{i}")
        return torch.device("cpu")

    @staticmethod
    def class_weight(train_dataset, device):
        neg, pos = np.bincount(train_dataset.labels.tolist())
        weight = [(1.0 / neg) * (pos + neg) / 2.0, (1.0 / pos) * (pos + neg) / 2.0]
        return torch.FloatTensor(weight).to(device)

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)

    @staticmethod
    def accuracy(y_hat, y):
        if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
            y_hat = argmax(y_hat, axis=1)
        cmp = astype(y_hat, y.dtype) == y
        return float(reduce_sum(astype(cmp, y.dtype)))

    @staticmethod
    def trainEpoch(net, trainDataLoader, loss, weight, updater, device):
        if isinstance(net, torch.nn.Module):
            net.train()

        metric = MyAccumulator(3)

        for X, y in trainDataLoader:
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y, weight=weight)
            updater.zero_grad()
            l.backward()
            updater.step()
            with torch.no_grad():
                metric.add(
                    l * X.batch_size,
                    MyUtils.accuracy(y_hat, y),
                    X.batch_size,
                )
        return metric[0] / metric[2], metric[1] / metric[2]


class MyAccumulator(object):
    """For accumulating sums over `n` variables."""

    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


reduce_sum = lambda x, *args, **kwargs: x.sum(*args, **kwargs)
argmax = lambda x, *args, **kwargs: x.argmax(*args, **kwargs)
astype = lambda x, *args, **kwargs: x.type(*args, **kwargs)
