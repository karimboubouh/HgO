import argparse
import pickle
import socket
import time
from math import ceil
from rich.console import Console
from inspect import currentframe, getframeinfo

import numpy as np
from PIL import Image
from rich.theme import Theme
from rich import inspect
from rich import pretty

from src.config import WEAK_DEVICE, AVERAGE_DEVICE, POWERFUL_DEVICE, LAYER_DIMS_FEMNIST, LAYER_DIMS_MNIST, \
    MIN_DATA_SAMPLE, LAYER_DIMS_CIFAR10


def log(*args, style="", debug=False, catch=False):
    pretty.install()
    if len(args) == 1:
        text = args[0]
    else:
        text = args

    styles = {"success": 'green', "warning": 'yellow', "error": 'bold red', "info": 'blue', "title": 'cyan bold'}
    console = Console(theme=Theme(styles))
    if catch:
        console.print(f":arrow_forward: {text}", style="error")
        console.print_exception(show_locals=True)
    elif debug:
        inspect(text, methods=True)
    elif style in styles.keys():
        console.print(f":arrow_forward: {text}", style=style)
    else:
        console.print(f"{style}:arrow_forward: {text}")


def elog(*args, style="", debug=False, catch=False):
    frameinfo = getframeinfo(currentframe().f_back)
    filename = frameinfo.filename.split('/')[-1]
    linenumber = frameinfo.lineno
    if len(args) == 1:
        text = args[0]
    else:
        text = args
    info = 'File: %s, line: %d' % (filename, linenumber)
    log(f"{text}\n{info}", style=style, debug=debug, catch=catch)
    exit()


def exp_details(args):
    log('Experimental details:', style="title")
    log(f'Model             : {args.model.upper()}', style="\t")
    log(f'Number of Workers : {args.workers}', style="\t")
    log(f'Active workers    : {args.q * 100}%', style="\t")
    log(f'Byzantine workers : {args.f}', style="\t")
    log(f'Number of Rounds  : {args.rounds}', style="\t")
    log(f'Unites of time    : {args.tau}', style="\t")
    log(f'Dataset           : {args.dataset}', style="\t")
    log(f'Data distribution : {"IID" if args.iid else "Non-IID"}', style="\t")
    log(f'Non-IID degree    : {args.iid_degree * 100}%', style="\t")
    log(f'Batch size        : {args.batch_size}', style="\t")
    log(f'Block strategy    : {args.block_strategy}', style="\t")
    log(f'Learning rate     : {args.lr}', style="\t")
    log(f'Aggregation rule  : {args.gar}', style="\t")
    log(f'Byzantine attack  : {args.attack}', style="\t")


def load_conf():
    # --model = mlp - -dataset = mnist
    parser = argparse.ArgumentParser()

    # federated arguments (Notation for the arguments followed from paper)
    parser.add_argument('--mp', type=int, default="0",
                        help="Use Message Passing (MP) instead of Shared memory (SM) ")
    parser.add_argument('--host', type=str, default="0.0.0.0", help="host IP address")
    parser.add_argument('--port', type=int, default=45000, help="port number")
    parser.add_argument('--rounds', type=int, default=300, help="number of rounds of training")
    parser.add_argument('--workers', type=int, default=100, help="number of workers.")
    parser.add_argument('--q', type=float, default=0.8, help='the fraction of workers')
    parser.add_argument('--v', type=int, default=1, help='block consistency')
    parser.add_argument('--model', type=str, default='LR', help='model name')
    parser.add_argument('--algo', type=str, default='HgO', help="Optimization algorithm: SGD or HgO (default)")
    parser.add_argument('--epochs', type=int, default=10, help="the number of local epochs")
    parser.add_argument('--tau', type=int, default=1, help="server wait time tau")
    parser.add_argument('--batch_size', type=int, default=128, help="batch size")
    parser.add_argument('--dbs', type=str, default="NO", help="Dynamic batch size.")
    parser.add_argument('--block_strategy', type=str, default="Default",
                        help="block selection strategy: Default, CoordinatesFirst, DataFirst, Hybrid")
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--dataset', type=str, default='mnist', help="name of dataset")
    parser.add_argument('--iid', type=int, default=1, help="IID or non-IID data")
    parser.add_argument('--iid_degree', type=float, default=1, help="degree of non-iidness [0, 1]")
    parser.add_argument('--f', type=int, default=0, help="f: number of Byzantine workers.")
    parser.add_argument('--gar', type=str, default='average', help="Aggregation rule: average, median, krum, aksel")
    parser.add_argument('--alpha', type=float, default=0.5, help="alpha: Byzantine ratio α")
    parser.add_argument('--attack', type=str, default="NO",
                        help="Byzantine attacks: FOE: Fall of Empires | LIE: Little Is Enough")
    parser.add_argument('--debug', type=int, default="1", help='debug, default true')
    parser.add_argument('--seed', type=int, default=0, help='random seed')

    return Map(vars(parser.parse_args()))


class Map(dict):
    """
    Example:
    m = Map({'first_name': 'Eduardo'}, last_name='Pool', age=24, sports=['Soccer'])
    """

    def __init__(self, *args, **kwargs):
        super(Map, self).__init__(*args, **kwargs)
        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.items():
                    self[k] = v

        if kwargs:
            for k, v in kwargs.items():
                self[k] = v

    def __getattr__(self, attr):
        return self.get(attr)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __setitem__(self, key, value):
        super(Map, self).__setitem__(key, value)
        self.__dict__.update({key: value})

    def __delattr__(self, item):
        self.__delitem__(item)

    def __delitem__(self, key):
        super(Map, self).__delitem__(key)
        del self.__dict__[key]


def dataset_split(dataset, mask):
    mask = list(mask)
    # shuffle(mask)
    np.random.shuffle(mask)
    data = dataset.data[mask, :]
    targets = dataset.targets[mask]
    if len(targets) < MIN_DATA_SAMPLE:
        log(f"Worker has less than minimum recommended samples {len(targets)} < {MIN_DATA_SAMPLE}", style="warning")
    return Map({'data': data, 'targets': targets})


def get_batch(X, y, batch_size, j):
    begin = j * batch_size
    end = min(begin + batch_size, X.shape[0])
    if end + batch_size > X.shape[0]:
        end = X.shape[0]
    X_ = X[begin:end, :]
    y_ = y[begin:end]
    return X_, y_


def dynamic_batch_size(args):
    capacity = [float(s) for s in args.dbs.split(',')]
    batches = [WEAK_DEVICE] * ceil(args.workers * capacity[0])
    batches += [AVERAGE_DEVICE] * ceil(args.workers * capacity[1])
    batches += [POWERFUL_DEVICE] * ceil(args.workers * capacity[2])

    return batches


def sigmoid(z):
    return 1. / (1 + np.exp(-z))


def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z) * (1 - sigmoid(z))


def loss(y, y_pred):
    return -(1 / len(y)) * np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))


def cross_entropy_grad(y, y_pred, X):
    return np.dot((y_pred - y), X)[0].reshape(-1, 1)


def accuracy(y, predictions):
    predictions = np.around(predictions)
    predictions = predictions.reshape(-1)
    y = y.reshape(-1)
    # return sum(predictions == y) / y.shape[0]
    return np.mean((predictions == y))


def divide_data(dataset, num_workers):
    """
    Sample I.I.D. workers data from dataset
    :param dataset:
    :param num_workers:
    :return: dict of index
    """
    num_items = int(len(dataset) / num_workers)
    dict_workers, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_workers):
        dict_workers[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_workers[i])

    return dict_workers


def estimate_shards(data_size, num_workers):
    shards = num_workers * 2 if num_workers > 10 else 20
    imgs = int(data_size / shards)

    return shards, imgs


def mnist_noniid(dataset, num_workers, degree=1):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param degree:
    :param dataset:
    :param num_workers:
    :return:
    """
    # 60,000 training imgs -->  200 imgs/shard X 300 shards
    # num_shards, num_imgs = 200, 300
    if not 0 <= degree <= 1:
        exit("!! the degree of non-iidness should be between 0 (IID) and 1 (Strict Non-IID).")
    shared = int(len(dataset.data) * (1 - degree))
    log(f"Non-IID distribution: {shared} samples out of {len(dataset.data)} are randomly shared between workers.")
    tosplit = len(dataset.data) - shared
    num_shards, num_imgs = estimate_shards(tosplit, num_workers)
    idx_shard = [i for i in range(num_shards)]
    dict_workers = {i: np.array([]).astype(int) for i in range(num_workers)}
    idxs = np.arange(num_shards * num_imgs)
    labels = np.argmax(dataset.targets, axis=1)
    # sort labels
    idxs_labels = np.vstack((idxs, labels[:len(idxs)]))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    # divide and assign 2 shards/client
    for i in range(num_workers):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_workers[i] = np.concatenate(
                (dict_workers[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)

    shared_idxs = np.array(range(shared))
    np.random.shuffle(shared_idxs)
    shared_idxs = np.array_split(shared_idxs, num_workers)
    for i in range(num_workers):
        if len(shared_idxs[i]):
            dict_workers[i] = set(np.concatenate((dict_workers[i], shared_idxs[i]), axis=0))
        else:
            dict_workers[i] = set(dict_workers[i])

    return dict_workers


def chunks(arr, n):
    if n > 0:
        np.random.shuffle(arr)
        output = [arr[i:i + n] for i in range(0, len(arr), n)]
        s = len(output[-1])
        if s < n:
            output[-1].extend(output[-2][s:n])
        return output
    else:
        return [list(range(len(arr)))]


def nn_chunks(layers, sizes):
    blocks = []
    dims = [layer.T.shape[0] for layer in layers] + [layers[-1].shape[0]]
    if len(sizes) > 0:
        for i, (dim, size) in enumerate(zip(dims, sizes)):
            mask = list(range(dim))
            if i != len(sizes) - 1:
                np.random.shuffle(mask)
                output = [mask[i:i + size] for i in range(0, len(mask), size)]
                s = len(output[-1])
                if s < size:
                    output[-1].extend(output[-2][s:size])
                blocks.append(output)
            else:
                blocks.append([mask])
        return blocks
    else:
        return None


def create_tcp_socket():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    # sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, TCP_SOCKET_BUFFER_SIZE)
    # sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, TCP_SOCKET_BUFFER_SIZE)
    # sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    return sock


def get_block(blocks):
    if blocks is None:
        return None
    output = []
    for block in blocks:
        index = np.random.choice(len(block), replace=False)
        output.append(block[index])

    return output


def class_name(c):
    return c.__class__.__name__


def save(filename, data):
    with open(filename, 'wb') as fp:
        pickle.dump(data, fp)
        print("Writing to file", filename)
    return


def load(filename):
    with open(filename, 'rb') as fp:
        return pickle.load(fp)


def percentage(value, base):
    if value == float('-inf'):
        return "--"
    return round((value * 100) / base, 2)


def nantrim_mean(percent, arr, axis):
    n = len(arr)
    k = int(np.array(n * percent))

    return np.nanmean(np.sort(arr)[k:n - k], axis=axis)


def flatten_grads(grads):
    flattened = [concat([concat(c) for c in grad]) for grad in grads]
    # print([np.isnan(ff).sum() for ff in flattened])
    # exit()
    return flattened


def concat(c):
    return np.concatenate([e.flatten() for e in c])


def unflatten_grad(grad, dims):
    if grad is None:
        return None
    divider = np.sum(dims[1:])
    dim_w = [(i, j) for i, j in zip(dims[1:], dims[:-1])]
    dim_b = [(i, 1) for i in dims[1:]]
    unf_w = [np.reshape(e, dim_w[i]) for i, e in enumerate(split(grad[:-divider], dim_w))]
    unf_b = [np.reshape(e, dim_b[i]) for i, e in enumerate(split(grad[-divider:], dim_b))]
    return [unf_w, unf_b]


def split(arr, dim):
    ind = [0]
    for i, j in dim:
        ind += [(i * j) + ind[-1]]
    return [arr[ind[i]:ind[i + 1]] for i in range(len(ind) - 1)]


def wait_until(predicate, timeout=1, period=0.05, *args_, **kwargs):
    start_time = time.time()
    mustend = start_time + timeout
    while time.time() < mustend:
        if predicate(*args_, **kwargs):
            return True
        time.sleep(period)
    return False


def mnist_noniid_____(dataset, num_workers, degree=0):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param degree:
    :param dataset:
    :param num_workers:
    :return:
    """
    # 60,000 training imgs -->  200 imgs/shard X 300 shards
    # num_shards, num_imgs = 200, 300
    if not 0 <= degree <= 1:
        exit("!! the degree of non-iidness should be between 0 (IID) and 1 (Strict Non-IID).")
    shared = len(dataset.data) * (1 - degree)
    num_shards, num_imgs = estimate_shards(len(dataset.data), num_workers)
    idx_shard = [i for i in range(num_shards)]
    dict_workers = {i: np.array([]).astype(int) for i in range(num_workers)}
    idxs = np.arange(num_shards * num_imgs)
    labels = np.argmax(dataset.targets, axis=1)
    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    print(idxs)
    exit()
    # divide and assign 2 shards/client
    for i in range(num_workers):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_workers[i] = np.concatenate(
                (dict_workers[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)

    for i in range(num_workers):
        dict_workers[i] = set(dict_workers[i])

    return dict_workers


def model_input(data, args):
    if args.dataset == "mnist":
        if args.model == "DNN":
            if args.dataset == "femnist":
                return LAYER_DIMS_FEMNIST
            else:
                return LAYER_DIMS_MNIST
        elif args.model == "MLR":
            return data.data.shape[1], data.targets.shape[1]
        else:
            return data.data.shape[1]
    elif args.dataset == "cifar10":
        if args.model == "DNN":
            return LAYER_DIMS_CIFAR10
        elif args.model == "MLR":
            return data.data.shape[1], data.targets.shape[1]
        else:
            return data.data.shape[1]
    elif args.dataset == "fashion":
        if args.model == "DNN":
            return LAYER_DIMS_MNIST
        elif args.model == "MLR":
            return data.data.shape[1], data.targets.shape[1]
        else:
            return data.data.shape[1]


def gen_image(arr):
    two_d = (np.reshape(arr, (28, 28)) * 255).astype(np.uint8)
    img = Image.fromarray(two_d, 'L')
    return img


def number_coordinates(W):
    if isinstance(W, np.ndarray):
        dims = list(W.shape)
        d = np.prod(W.shape)
    else:
        dims = [layer.T.shape[0] for layer in W] + [W[-1].shape[0]]
        d = np.sum([np.prod(w.shape) for w in W])

    return dims, d


def clmn(lists, i):
    return [row[i].item() for row in lists if len(row) > i]


def w_slice(w, bv, i):
    rl = np.arange(bv[i + 1]) if w.shape[0] > bv[i + 1] else np.arange(w.shape[0])
    cl = np.arange(bv[i]) if w.shape[1] > bv[i] else np.arange(w.shape[1])
    sliced = w[np.ix_(rl, cl)]
    filled = np.pad(sliced, ((0, bv[i + 1] - len(rl)), (0, bv[i] - len(cl))), 'constant', constant_values=np.nan)
    # print(f"w={w.shape} filled with (0, {bv[i + 1] - len(rl)}), (0, {bv[i] - len(cl)}) --> filled={filled.shape}")
    return filled


def b_slice(b, bv, i):
    rl = np.arange(bv[i]) if b.shape[0] > bv[i] else np.arange(b.shape[0])
    cl = np.arange(1)
    sliced = b[np.ix_(rl, cl)]
    filled = np.pad(sliced, ((0, bv[i] - len(rl)), (0, 0)), 'constant', constant_values=np.nan)
    # print(f"b={b.shape} filled with (0, {bv[i] - len(rl)}), (0, 0)")
    return filled


def fill_array(arr, model_name, size=None):
    if model_name == "MLR":
        dws, dbs = zip(*arr)
        n_out = dbs[0].shape[0]
        if size is None:
            size = max(map(len, dws))
        cw = [np.concatenate((g, np.full((size - len(g), n_out), np.nan))) for g in dws]
        return cw, dbs
    else:
        if size is None:
            size = max(map(len, arr))
        empty = np.array([np.nan] * size).reshape(size, 10)
        return [np.concatenate((g, empty[:size - len(g)])) for g in arr]


def dynamic_lambda(workers, dynamic):
    capacity = [float(s) for s in dynamic.split(',')]
    batches = [int(np.random.normal(WEAK_DEVICE[0], WEAK_DEVICE[1])) for _ in np.arange(ceil(workers * capacity[0]))]
    batches += [int(np.random.normal(AVERAGE_DEVICE[0], AVERAGE_DEVICE[1])) for _ in
                np.arange(ceil(workers * capacity[1]))]
    batches += [int(np.random.normal(POWERFUL_DEVICE[0], POWERFUL_DEVICE[1])) for _ in
                np.arange(ceil(workers * capacity[2]))]
    batches = np.array(batches)
    batches[batches < 0] = 0
    return list(batches)


def softmax(x, eps=1e-7):
    # e = np.exp(x)
    e = np.exp(x - np.max(x))  # prevent overflow
    if e.ndim == 1:
        return e / (np.sum(e, axis=0) + eps)
    else:
        return e / (np.array([np.sum(e, axis=1)]).T + eps)


def flatten(arr):
    """
    Flatten an array or list of arrays
    """
    if isinstance(arr, np.ndarray):
        shape = arr.shape
        arr = [arr]
    elif isinstance(arr, (list, tuple)):
        shape = list(map(np.shape, arr))
    else:
        elog(f"Array type {type(arr)} is not supported>", style="error")
    try:
        _flattened = [f.ravel() for f in arr]
    except AttributeError:
        log(arr)
        elog([f.ravel() for f in arr])
    # _flattened = np.concatenate()
    return _flattened


def unflatten(arr, shapes):
    """
    Unflatten an array using a given shape or list of shapes
    """
    if isinstance(shapes, tuple):
        return arr.reshape(shapes)
    elif isinstance(shapes, list):
        parts = np.split(arr, np.cumsum([np.prod(s) for s in shapes]))
        parts = [p for p in parts if len(p) > 0]
        return [part.reshape(shapes[i]) for i, part in enumerate(parts)]
    else:
        elog(f"shapes type {type(arr)} is not supported>", style="error")


def number_grads(grads):
    size = 0
    for grad in grads:
        if isinstance(grad, np.ndarray):
            size += grad.size
        else:
            if isinstance(grad[0], np.ndarray):
                size += grad.size
            else:
                size += sum(g.size for g in grad[0])
    return size


if __name__ == '__main__':
    data = [np.random.randn(784, 30), np.random.randn(30, 10)]
    flattened, shape = flatten(data)
    log(flattened.shape)

    unflattened = unflatten(flattened, shape)
    log([u.shape for u in unflattened])
    elog("Done")
