import pickle

import numpy as np

DISCONNECT = 0
TRAIN_JOIN = 1
TRAIN_START = 2
TRAIN_INFO = 3
TRAIN_STOP = 5


def join_train(model):
    return pickle.dumps({
        'mtype': TRAIN_JOIN,
        'data': {'model': model},
    })


def start_round(W, C, tau):
    return pickle.dumps({
        'mtype': TRAIN_START,
        'data': {'W': W, 'C': C, 'tau': tau},
    })


def train_info(grads, gtime):
    return pickle.dumps({
        'mtype': TRAIN_INFO,
        'data': {'grads': grads, 'gtime': gtime}
    })


def stop_train(server):
    battery = np.sum(server.battery_usage[-server.status.active])
    data = {
        'performance': server.performance[-1],
        'battery_usage': battery,
        'iteration_cost': np.mean(server.iteration_cost)
    }
    return pickle.dumps({
        'mtype': TRAIN_STOP,
        'data': data,
    })


def disconnect():
    return pickle.dumps({
        'mtype': DISCONNECT,
        'data': {},
    })
