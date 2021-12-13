import copy
import math
import pickle
import socket
import time
import traceback
from struct import unpack, pack
from threading import Thread

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
from tqdm import tqdm

from src import message
from src.aggregators import aggregate
from src.config import SOCK_TIMEOUT, TCP_SOCKET_SERVER_LISTEN, BYZ_ITER, EVAL_ROUND, FOE_EPS, LIE_Z, RECV_BUFFER, \
    ALLOW_DIFF_BATCH_SIZE, BARS_ROUND, USE_DIFFERENT_HARDWARE
from src.nonlinear_optimizers import DNNOptimizer
from src.optimizers import LROptimizer, RROptimizer, LNOptimizer, SVMOptimizer, MLROptimizer
from src.utils import Map, create_tcp_socket, dataset_split, wait_until, class_name, \
    flatten_grads, unflatten_grad, number_coordinates, w_slice, b_slice, fill_array, log, elog, flatten, unflatten, \
    number_grads


class ParamServer(Thread):

    def __init__(self, model, nb_workers, args: Map):
        super(ParamServer, self).__init__()
        self.mp = bool(args.mp)
        self.host = args.host
        self.port = args.port
        self.sock = None
        self.terminate = False
        self.status = Map({'active': 0, 'train': False, 'aggregate': False, 'started': False})
        self.raw_model = model
        self.model = copy.deepcopy(self.raw_model)
        self.nb_workers = nb_workers
        self.workers = []
        self.optimizer = None
        self.blocks = None
        self.block_size = None
        self.grads = []
        self.vg = {}
        self.gtime = []
        self.history = []
        self.grad_time = []
        self.x_time = []
        self.ngrads = []
        # default params
        self.params = Map({
            'rounds': args.rounds,
            'algo': args.algo,
            'optimizer': "sgd",
            'q': args.q,
            'v': args.v,
            'lr': args.lr,
            'tau': args.tau,
            'GAR': args.gar,
            'alpha': args.alpha,
            'batch_size': self.model.batch_size,
            'f': args.f,
            'attack': args.attack,
        })
        # define Byzantine workers
        self.byz_indices = []
        if self.mp:
            self.init_server()

    def init_server(self):
        log(f"Starting server on ({self.host}:{self.port})", style="info")
        self.sock = create_tcp_socket()
        self.sock.bind((self.host, self.port))
        self.sock.settimeout(SOCK_TIMEOUT)
        self.sock.listen(TCP_SOCKET_SERVER_LISTEN)

    def run(self):
        if self.mp:
            log(f"Server waiting for incoming connections ...", style="info")
            while not self.terminate:
                try:
                    conn, address = self.sock.accept()
                    if not self.terminate:
                        worker_thread = WorkerConnection(self, conn, address)
                        worker_thread.start()
                        self.workers.append(worker_thread)
                        # Start training if workers all joined
                        # self.update_status()
                        # self.start_train()
                except socket.timeout:
                    pass
                except Exception as e:
                    print(f"!! {self}: Exception!\n{e}")
                self.update_status()

            log(f"{self}: Terminating connections ...", style="warning")
            for w in self.workers:
                w.stop()
            time.sleep(1)
            for w in self.workers:
                w.join()
            self.sock.close()
            print(f"!! {self}: Stopped.")
        else:
            log(f"Server constructing the communication map ...", style="info")

    def start_train(self):
        for worker in self.workers:
            if self.mp:
                worker.send(message.join_train(self.model))
            else:
                worker.join_train({'model': self.model})

    def broadcast(self, msg, only=None):
        try:
            workers = only if only is not None else self.workers
            data = pickle.loads(msg)
            for worker in workers:
                if self.mp:
                    worker.send(msg)
                else:
                    worker.local_train(data['data'])
        except Exception as e:
            print(f"!! {self}: Exception in <Broadcast>\n{e}")
            traceback.print_exc()

    def reset(self, layer_dims=None):
        if layer_dims is None:
            self.model.reset()
        else:
            self.model.reset(layer_dims)
        self.optimizer = None
        self.blocks = None
        self.grads = []
        self.history = []
        self.grad_time = []
        self.x_time = []
        self.ngrads = []

    def init_byzantine_workers(self):
        if self.params.f > 0:
            b = np.random.choice(range(len(self.workers)), self.params.f, replace=False)
            for w in self.workers[b]:
                w.byzantine = Byzantine(w, self, self.params.attack)
            self.byz_indices = b
        else:
            self.byz_indices = []

    def init_optimizers(self):
        if self.params.optimizer == "momentum":
            self.vg["beta"] = 0.9
            if isinstance(self.model.W, list):
                self.vg["w"] = [np.zeros(w.shape) for w in self.model.W]
                self.vg["b"] = [np.zeros(w.shape) for w in self.model.b]
            else:
                self.vg["w"] = np.zeros(self.model.W.shape)
                self.vg["b"] = np.zeros(self.model.b.shape)
        elif self.params.optimizer == "adam":
            self.vg["b1"] = 0.9
            self.vg["b2"] = 0.999
            self.vg["a"] = 1e-8
            if isinstance(self.model.W, list):
                self.vg["vw"] = [np.zeros(w.shape) for w in self.model.W]
                self.vg["vb"] = [np.zeros(b.shape) for b in self.model.b]
                self.vg["sw"] = [np.zeros(w.shape) for w in self.model.W]
                self.vg["sb"] = [np.zeros(b.shape) for b in self.model.b]
            else:
                self.vg["vw"] = np.zeros(self.model.W.shape)
                self.vg["vb"] = np.zeros(self.model.b.shape)
                self.vg["sw"] = np.zeros(self.model.W.shape)
                self.vg["sb"] = np.zeros(self.model.b.shape)

    def train(self, X, y):
        t = time.time()
        time_laps = t
        acc_time = float('-inf')
        self.ngrads = []
        rounds = tqdm(range(self.params.rounds))
        if USE_DIFFERENT_HARDWARE:
            self.history.append(self.evaluate(X, y))
        self.x_time.append(0)
        # activate if you want to use momentum or adam
        # self.init_optimizers()
        for i in rounds:
            self.grads = []
            self.gtime = []
            if np.isnan(self.model.W[0]).any() or np.isnan(self.model.W[0]).any():
                elog(f"W in round {i} Contains nan!", style="error")
            # Select a portion of q workers that may include byzantine workers
            honest, byzantine = self.active_workers(i)
            # v <-- 1 if BYZ_ITER is not reached yet
            self.params.v = int(len(byzantine) / self.params.alpha) + 1
            # shuffle the gradient vector indices ({1,...,d})
            C = self.get_shuffled_grad_indices()
            self.broadcast(message.start_round(self.model.W, C, self.params.tau), only=self.workers[honest])
            t = time.time()
            # TODO restore
            # wait_until(self.grads_received, 1e-4, 1e-4, honest)
            if len(self.grads) < 1:
                log(f"Round {i}: Server received no gradients in {self.params.tau} unite(s) of time.", style="error")
                if not USE_DIFFERENT_HARDWARE:
                    if i % EVAL_ROUND == 0:
                        self.history.append(self.evaluate(X, y))
                    if i % BARS_ROUND == 0:
                        self.ngrads.append(len(self.grads))
                continue

            # t = round((time.time() - t), 4)
            # print(f">>{self.params.tau} --> took {t}s to receive {len(self.grads)} / {len(honest)} grads.")
            # Byzantine attack
            if self.params.attack != "NO":
                self.byzantine_attacks(byzantine)
            # self.grad_time.append(self.gtime)
            grads, block, bv, tblock = self.consistent_grads(C)
            round_grad = self.aggregate(grads, block)
            self.take_step(round_grad, block, i)
            # if self.params.algo == "HgO":
            #     if isinstance(self.model.W, np.ndarray):
            #         lr_max = 10
            #         self.params.lr = lr_max / bv
            #     else:
            #         lr_max = 2472
            #         self.params.lr = lr_max / np.linalg.norm(bv, ord=1)
            # laps_grads = LAPS_GRADS
            # if np.sum(self.grad_time) > laps_grads:
            #     laps_grads += LAPS_GRADS
            #     self.history.append(self.evaluate(X, y))

            # if time.time() - time_laps >= LAPS:
            #     self.history.append(self.evaluate(X, y))
            #     time_laps = time.time()

            if i % BARS_ROUND == 0:
                self.ngrads.append(len(self.grads))

            if USE_DIFFERENT_HARDWARE:
                round_time = np.max(self.gtime) + tblock
                self.grad_time.append(round_time)
                if i % EVAL_ROUND == 0:
                    self.history.append(self.evaluate(X, y))
                    self.x_time.append(self.x_time[-1] + round_time)
                # if self.x_time[-1] > max_time:
                #     break
            else:
                self.grad_time.append(np.max(self.gtime))
                if i % EVAL_ROUND == 0:
                    self.history.append(self.evaluate(X, y))

            # try:
            #     if acc_time == float('-inf') and self.history[-1][1] >= ACC_TIME:
            #         acc_time = time.time() - t
            # except:
            #     acc_time = float('-inf')

        t = time.time() - t
        log(f"Decentralized training finished in {t:.2f} seconds.", style="success")
        log(f"Number of grads: {np.sum(self.ngrads)}", style="info")

        return acc_time

    def byzantine_attacks(self, byzantine):
        byz_workers = self.workers[byzantine]
        if len(byz_workers) > 0:
            byz_grads = [byz_workers[0].attack()] * len(byz_workers)
            self.grads.extend(byz_grads)

    def byzantine_attacks_old(self, byzantine):
        byz_grads = []
        for byz in self.workers[byzantine]:
            byz_grads.append(byz.attack())
        self.grads.extend(byz_grads)

    def evaluate(self, X, y):
        return self.model.evaluate(X, y)

    def summary(self, train, test):
        cost, acc = self.evaluate(train.data, train.targets)
        log(f"Train >> Loss: {cost:.4f}, Accuracy: {(acc * 100):.2f}%.", style="warning")
        cost, acc = self.evaluate(test.data, test.targets)
        log(f"Test  >> Loss: {cost:.4f}, Accuracy: {(acc * 100):.2f}%.", style="warning")
        return self

    def aggregate(self, grads, block=None):
        model_name = class_name(self.model)
        if self.params.GAR == "average":
            return aggregate(model_name, grads, block, "average")
        elif self.params.GAR == "median":
            return aggregate(model_name, grads, block, "median")
        elif self.params.GAR == "tmean":
            return aggregate(model_name, grads, block, "tmean")
        elif self.params.GAR == "aksel":
            return aggregate(model_name, grads, block, "aksel")
        elif self.params.GAR == "krum":
            return aggregate(model_name, grads, block, "krum")
        else:
            raise NotImplementedError()

    def take_step(self, grad, block, i):
        name = class_name(self.model)
        if grad is None:
            return
        # t = i + 1
        # dw, db = grad
        # if self.params.optimizer == "momentum":
        #     if isinstance(self.vg["w"], list):
        #         for i in range(len(db)):
        #             self.vg["w"][i] = self.vg["beta"] * self.vg["w"][i] + (1 - self.vg["beta"]) * dw[i]
        #             self.vg["b"][i] = self.vg["beta"] * self.vg["b"][i] + (1 - self.vg["beta"]) * db[i]
        #     else:
        #         self.vg["w"] = self.vg["beta"] * self.vg["w"] + (1 - self.vg["beta"]) * dw
        #         self.vg["b"] = self.vg["beta"] * self.vg["b"] + (1 - self.vg["beta"]) * db
        #     vw = self.vg["w"]
        #     vb = self.vg["b"]
        #     grad = (vw, vb)
        # elif self.params.optimizer == "adam":
        #     if isinstance(self.vg["vw"], list):
        #         vw = []
        #         vb = []
        #         vw_corrected = []
        #         vb_corrected = []
        #         sw_corrected = []
        #         sb_corrected = []
        #         for i in range(len(db)):
        #             self.vg["vw"][i] = self.vg["b1"] * self.vg["vw"][i] + (1 - self.vg["b1"]) * dw[i]
        #             self.vg["vb"][i] = self.vg["b1"] * self.vg["vb"][i] + (1 - self.vg["b1"]) * db[i]
        #             vw_corrected.append(self.vg["vw"][i] / (1 - self.vg["b1"] ** t))
        #             vb_corrected.append(self.vg["vb"][i] / (1 - self.vg["b1"] ** t))
        #             self.vg["sw"][i] = self.vg["b2"] * self.vg["sw"][i] + (1 - self.vg["b2"]) * np.square(dw[i])
        #             self.vg["sb"][i] = self.vg["b2"] * self.vg["sb"][i] + (1 - self.vg["b2"]) * np.square(db[i])
        #             sw_corrected.append(self.vg["sw"][i] / (1 - self.vg["b2"] ** t))
        #             sb_corrected.append(self.vg["sb"][i] / (1 - self.vg["b2"] ** t))
        #             vw.append(vw_corrected[i] / (np.sqrt(sw_corrected[i]) + self.vg["a"]))
        #             vb.append(vb_corrected[i] / (np.sqrt(sb_corrected[i]) + self.vg["a"]))
        #         grad = (vw, vb)
        #     else:
        #         self.vg["vw"] = self.vg["b1"] * self.vg["vw"] + (1 - self.vg["b1"]) * dw
        #         self.vg["vb"] = self.vg["b1"] * self.vg["vb"] + (1 - self.vg["b1"]) * db
        #         vw_corrected = self.vg["vw"] / (1 - self.vg["b1"] ** t)
        #         vb_corrected = self.vg["vb"] / (1 - self.vg["b1"] ** t)
        #         self.vg["sw"] = self.vg["b2"] * self.vg["sw"] + (1 - self.vg["b2"]) * np.square(dw)
        #         self.vg["sb"] = self.vg["b2"] * self.vg["sb"] + (1 - self.vg["b2"]) * np.square(db)
        #         sw_corrected = self.vg["sw"] / (1 - self.vg["b2"] ** t)
        #         sb_corrected = self.vg["sb"] / (1 - self.vg["b2"] ** t)
        #         vw = vw_corrected / (np.sqrt(sw_corrected) + self.vg["a"])
        #         vb = vb_corrected / (np.sqrt(sb_corrected) + self.vg["a"])
        #         grad = (vw, vb)

        if name == "DNN":
            self._take_step_dnn(grad, block)
        elif name == "MLR":
            self._take_step_mlr(grad, block)
        else:
            self._take_step_linear(grad, block)

        return self

    def active_workers(self, i):
        if i >= BYZ_ITER:
            m = max(int(self.params.q * len(self.workers)), 1)
            active = np.random.choice(np.arange(len(self.workers)), m, replace=False)
            honest = active[~np.in1d(active, self.byz_indices)]
            byzantine = active[~np.in1d(active, honest)]
        else:
            m = max(int(self.params.q * len(self.workers) - self.params.f), 1)
            active = np.delete(np.arange(len(self.workers)), self.byz_indices)
            honest = np.random.choice(active, m, replace=False)
            byzantine = []
        # print(f">> {len(active)} active workers | {len(honest)} honest and {len(byzantine)} Byzantine.")
        return honest, byzantine

    def active_workers_old(self):
        m = max(int(self.params.q * len(self.workers) - self.params.f), 1)
        print(self.params.q * len(self.workers))
        print(self.params.f)
        print(m)
        active = np.delete(np.arange(len(self.workers)), self.byz_indices)
        print(len(active))
        print(np.random.choice(active, m, replace=False))
        exit()
        return np.random.choice(active, m, replace=False)

    def grads_received(self, active):
        if len(self.grads) == len(active):
            return True
        return False

    def workers_connected(self):
        if len(self.workers) == self.nb_workers:
            self.workers = np.array(self.workers)
            return True
        return False

    def get_workers(self):
        return self.workers

    def set_workers(self, workers):
        self.workers = workers
        return self

    def consistent_grads(self, C):
        t = time.time()
        model_name = class_name(self.model)
        # SGD algorithm
        if self.params.algo == "SGD":
            if model_name == "DNN":
                return self.grads, C, sum(len(c) for c in C), time.time() - t
            elif model_name == "MLR":
                return self.grads, C, len(C), time.time() - t
            else:
                return self.grads, C, len(C), time.time() - t
        # HgO algorithm
        if model_name == "DNN":
            dws, dbs = zip(*self.grads)
            output_R_w = np.array([dws[0][-1].shape[0]] * len(dws))
            R_w = [np.flip(np.sort([l.shape[1] for l in layer])) for layer in zip(*dws)]
            R_w.append(output_R_w)
            R_b = [np.flip(np.sort([l.shape[0] for l in layer])) for layer in zip(*dbs)]
            bv_w = [R[self.params.v - 1] for R in R_w]
            bv_b = [R[self.params.v - 1] for R in R_b]
            cw = [[w_slice(w, bv_w, i) for i, w in enumerate(dw)] for dw in dws]
            cb = [[b_slice(b, bv_b, i) for i, b in enumerate(db)] for db in dbs]
            block = [C[i][:bv] for i, bv in enumerate(bv_w)]
            return list(zip(cw, cb)), block, bv_w, time.time() - t
        elif model_name == "MLR":
            n_in, n_out = self.model.W.shape
            dws, dbs = zip(*self.grads)
            R = np.flip(np.sort([len(grad) for grad in dws]))
            bv = R[self.params.v - 1]
            cw = [grad[:bv] if len(grad) > bv else grad for grad in dws]
            cb = [grad[:bv] if len(grad) > bv else grad for grad in dbs]
            empty_w = np.array([np.nan] * bv * n_out).reshape(bv, n_out)
            ew = [np.concatenate((grad, empty_w[:bv - len(grad)])) for grad in cw]
            block = C[:bv]
            return list(zip(ew, cb)), block, bv, time.time() - t
        else:
            R = np.flip(np.sort([len(grad) for grad in self.grads]))
            bv = R[self.params.v - 1]
            cgrads = [grad[:bv] if len(grad) > bv else grad for grad in self.grads]
            empty = np.array([np.nan] * bv).reshape(bv, 1)
            dgrads = [np.concatenate((grad, empty[:bv - len(grad)])) for grad in cgrads]
            block = C[:bv]

            return dgrads, block, bv, time.time() - t

    def get_block(self):
        try:
            if class_name(self.model) in ['DNN', 'CNN']:
                output = []
                for block in self.blocks:
                    index = np.random.choice(len(block), replace=False)
                    output.append(block[index])
                return output
            else:
                index = np.random.choice(len(self.blocks), replace=False)
                return self.blocks[index]
        except:
            return None

    def get_shuffled_grad_indices(self):
        try:
            if class_name(self.model) == "DNN":
                dims = [layer.T.shape[0] for layer in self.model.W] + [self.model.W[-1].shape[0]]
                indxs = [np.random.permutation(list(range(dim))) for dim in dims[:-1]]
                return indxs + [np.arange(dims[-1])]
            else:
                indxs = np.random.permutation(list(range(self.model.W.shape[0])))
                return indxs
        except:
            return None

    def opt_strategy(self, args):
        if args.model == "LR":
            self.model.optimizer = LROptimizer(self.model.W)
        if args.model == "MLR":
            self.model.optimizer = MLROptimizer(self.model.W)
        elif args.model == "LN":
            self.model.optimizer = LNOptimizer(self.model.W)
        elif args.model == "RR":
            self.model.optimizer = RROptimizer(self.model.W)
        elif args.model == "SVM":
            self.model.optimizer = SVMOptimizer(self.model.W)
        elif args.model == "DNN":
            self.model.optimizer = DNNOptimizer(self.model.W, self.model.b)
        else:
            NotImplementedError()

    def plot(self, measure="loss"):
        xlabel = 'Rounds'
        ylabel = f' {measure.capitalize()}'
        title = f'{measure.capitalize()} vs. No. of rounds'
        if measure == "loss":
            data = [loss for loss, acc in self.history]
        elif measure == "acc":
            data = [acc for loss, acc in self.history]
        else:
            raise NotImplementedError("Only 'loss' and 'acc' are accepted")
        plt.plot(data)  # , '-x'
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.show()

    # -------------------------------------------------------------------------

    def update_status(self):
        pass
        # self.status.active = len(self.active_workers())
        # self.status.train = True if len(self.workers) == self.nb_workers else False
        # self.status.aggregate = self.status.train and len(self.grads) >= self.params.q * self.status.active

    def _take_step_linear(self, grad, block):
        grad = grad.reshape(-1, 1)
        if block is None:
            self.model.W = self.model.W - self.params.lr * grad
        else:
            self.model.W[block] = self.model.W[block] - self.params.lr * grad

    def _take_step_dnn(self, grad, block):
        dw, db = grad
        for idx, (w, b, gw, gb) in enumerate(zip(self.model.W, self.model.b, dw, db)):
            gw = np.nan_to_num(gw)
            w[np.ix_(block[idx + 1], block[idx])] -= self.params.lr * gw
            gb = np.nan_to_num(gb)
            b[np.ix_(block[idx + 1])] -= self.params.lr * gb

    def _take_step_mlr(self, grad, block):
        dw, db = grad
        self.model.W[block] -= self.params.lr * dw
        self.model.b -= self.params.lr * db

    # Special methods
    def __repr__(self):
        return f"Server()"

    def __str__(self):
        return f"Server()"


class WorkerConnection(Thread):

    def __init__(self, server, sock, address):
        super(WorkerConnection, self).__init__()
        self.server = server
        self.sock = sock
        self.address = address
        self.terminate = False
        self.byzantine = None

    def run(self):
        # Wait for messages from device
        while not self.terminate:
            try:
                (length,) = unpack('>Q', self.sock.recv(8))
                buffer = b''
                while len(buffer) < length:
                    to_read = length - len(buffer)
                    buffer += self.sock.recv(RECV_BUFFER if to_read > RECV_BUFFER else to_read)

                if buffer:
                    data = pickle.loads(buffer)
                    if data and data['mtype'] == message.TRAIN_INFO:
                        self.handle_epoch(data['data'])
                    elif data and data['mtype'] == message.DISCONNECT:
                        self.handle_disconnect()
                    else:
                        print(f"!! {self.server}: Unknown type of message: {data['mtype']}.")
            except pickle.UnpicklingError as e:
                print(f"!! {self.server}: Corrupted message : {e}")
            except socket.timeout:
                pass
            except Exception as e:
                self.terminate = True
                print(f">> workers1: {len(self.server.workers)}")
                self.server.workers.remove(self)
                print(f">> workers2: {len(self.server.workers)}")
                print(f"!! {self.server} WorkerConnection: Socket Exception\n{e}")
                traceback.print_exc()

        self.sock.close()
        print(f">> {self.server}: Worker disconnected")

    def send(self, msg):
        try:
            length = pack('>Q', len(msg))
            self.sock.sendall(length)
            self.sock.sendall(msg)
        except socket.error as e:
            self.terminate = True
            print(f"!! {self.server.name} WorkerConnection: Socket error\n{e}")
        except Exception as e:
            print(f"!! {self.server.name} WorkerConnection: Exception\n{e}")
            traceback.print_exc()

    def stop(self):
        self.terminate = True

    def handle_epoch(self, data):
        if data['grads'] is not None:
            self.server.grads.append(data['grads'])
            self.server.gtime.append(data['gtime'])

    def handle_disconnect(self):
        self.terminate = True
        self.sock.close()
        self.server.workers.remove(self)

    def attack(self):
        return self.byzantine.attack()


class Worker(Thread):

    def __init__(self, k, train, test, mask, args: Map, server: ParamServer = None):
        super(Worker, self).__init__()
        self.id = k
        self.mp = bool(args.mp)
        self.server = server
        self.sock = None
        self.terminate = False
        self.byzantine = None
        self.model = None
        self.grads = None
        self.gtime = None
        self.lamda = 1e3
        self.rho = 0.5
        self.train = mask['train'] if args.dataset == "femnist" else dataset_split(train, mask)
        # log(f"{self}: {len(self.train.targets)} samples with classes: {set(np.argmax(self.train.targets, axis=1))}")
        self.test = mask['test'] if args.dataset == "femnist" else test
        # default params
        self.params = Map({
            'server_host': args.host,
            'server_port': args.port,
            'algo': args.algo,
            'block_strategy': args.block_strategy,
        })
        if self.mp:
            # connect to server
            self.connect()
        else:
            server.workers.append(self)

    def connect(self):
        try:
            self.sock = create_tcp_socket()
            self.sock.settimeout(SOCK_TIMEOUT)
            self.sock.connect((self.params.server_host, self.params.server_port))
        except Exception as e:
            print(f"!! {self} not connected: {e}")

    def run(self):
        if self.mp:
            # Wait for messages from server
            while not self.terminate:
                try:
                    (length,) = unpack('>Q', self.sock.recv(8))
                    buffer = b''
                    while len(buffer) < length:
                        to_read = length - len(buffer)
                        buffer += self.sock.recv(RECV_BUFFER if to_read > RECV_BUFFER else to_read)

                    if buffer:
                        data = pickle.loads(buffer)
                        if data and data['mtype'] == message.TRAIN_JOIN:
                            self.join_train(data['data'])
                        elif data and data['mtype'] == message.TRAIN_START:
                            self.local_train(data['data'])
                        elif data and data['mtype'] == message.TRAIN_STOP:
                            self.stop_train(data['data'])
                        elif data and data['mtype'] == message.DISCONNECT:
                            self.stop()
                        else:
                            print(f"!! Unknown type of message: {data['mtype']}.")
                except pickle.UnpicklingError as e:
                    print(f"!! Corrupted message: {e}")
                    traceback.print_exc()
                except socket.timeout:
                    pass
                except Exception as e:
                    self.terminate = True
                    print(f"!! Socket Exception: {e}")
                    traceback.print_exc()

            self.sock.close()
            print(f">> Client disconnected")
        else:
            pass

    def send(self, msg):
        if self.mp:
            try:
                length = pack('>Q', len(msg))
                self.sock.sendall(length)
                self.sock.sendall(msg)
            except socket.error as e:
                self.terminate = True
                print(f"!! Socket error\n{e}")
            except Exception as e:
                print(f"!! Exception\n{e}")
        else:
            data = pickle.loads(msg)
            if data['mtype'] == message.TRAIN_INFO:
                if data['data']['grads'] is not None:
                    self.server.grads.append(data['data']['grads'])
                    self.server.gtime.append(data['data']['gtime'])
            else:
                print(f"!! Unknown type of message to be sent: {data['mtype']}.")
                exit()

    def stop(self):
        self.terminate = True

    def join_train(self, data):
        self.model = copy.deepcopy(data['model'])
        # if no batch_size is provided use all data
        # self.model.batch_size = len(self.train.data)

    def local_train(self, data):
        self.model.W = data['W']
        tau = data['tau']
        block, bi, si = self.round_complexity(data['C'], tau)
        if isinstance(block, str) and block == "incapable":
            self.send(message.train_info(None, None))
            return
        grads, gtime = self.model.one_epoch(self.train.data, self.train.targets, block, si)
        if USE_DIFFERENT_HARDWARE:
            # simulate weak hardware
            _, d = number_coordinates(self.model.W)
            gtime += bi / self.lamda
        self.send(message.train_info(grads, gtime))

    def round_complexity(self, C, tau):
        # number of coordinates
        dims, d = number_coordinates(self.model.W)
        if self.params.algo == "SGD":
            max_si = int((tau * self.lamda) / d)
            if d * self.model.batch_size <= tau * self.lamda:
                bi = d
                si = self.model.batch_size
            elif max_si > 0 and ALLOW_DIFF_BATCH_SIZE:
                self.model.batch_size = max_si
                bi = d
                si = max_si
            else:
                return "incapable", None, None
        else:
            # get bi, si based block selection strategy
            if self.params.block_strategy == "Hybrid":
                bi, si = self.get_block_hybrid(d, tau)
            elif self.params.block_strategy == "CoordinatesFirst":
                bi, si = self.get_block_coordinates_first(d, tau)
            elif self.params.block_strategy == "DataFirst":
                bi, si = self.get_block_data_first(d, tau)
            else:
                bi, si = self.get_block_for_batch(d, tau)
        # check for small values of si and bi
        if bi < 1 or si < 1:
            return "incapable", None, None
        # print(f">> Block strategy: {self.params.block_strategy}, bi: {bi}, si: {si}")
        # return a block from C and si data points from training
        block = self.get_block(C, dims, d, bi)

        return block, bi, si

    def get_block_for_batch(self, d, tau):
        if self.lamda * tau >= self.model.batch_size:
            bi = min(int((self.lamda * tau) / self.model.batch_size), d)
            si = self.model.batch_size
        elif ALLOW_DIFF_BATCH_SIZE:
            bi = 1
            si = int(self.lamda * tau)
        else:
            bi = si = -1

        return bi, si

    def get_block_coordinates_first(self, d, tau):
        S = len(self.train.data)
        if self.lamda * tau >= d:
            bi = d
            si = min(int((self.lamda * tau) / d), S)
        else:
            bi = int(self.lamda * tau)
            si = 1

        return bi, si

    def get_block_data_first(self, d, tau):
        S = len(self.train.data)
        if self.lamda * tau >= S:
            bi = min(int((self.lamda * tau) / S), d)
            si = S
        else:
            bi = 1
            si = int(self.lamda * tau)

        return bi, si

    def get_block_hybrid(self, d, tau):
        S = len(self.train.data)
        # verify self.rho
        min_rho = min(1 / (1 + self.lamda * tau / S ** 2), self.lamda * tau / (d * S))
        if self.rho >= 0.999:
            self.rho = 0.998
            print(f"!! rho must be less than 1, rho is set to {self.rho}")
        elif self.rho < min_rho:
            e = str(min_rho)[::-1].find('.') - 1
            e = e if e < 5 else 4
            self.rho = np.ceil(min_rho * 10 ** e) / 10 ** e
            print(f"!! rho must be greater than {min_rho}, rho is set to {self.rho}")
        # calculate bi, si
        bi = min(np.sqrt(self.rho * self.lamda * tau / (1 - self.rho)), d * self.rho)
        si = min((self.lamda * tau) / bi, len(self.train.data))
        if si < 1:
            print(f"!! si={si} is less than 1: wrong estimation")
        bi = np.floor(bi).astype('int')
        si = np.floor(si).astype('int')

        return bi, si

    def get_block(self, C, dims, d, bi):
        if isinstance(C, np.ndarray):
            # one layer: LN, LR, SVM ...
            return C[:bi]
        else:
            # DNN, CNN ...
            if bi == d:
                return [list(C[i][:k]) for i, k in enumerate(dims)]
            bi = bi - dims.pop()
            if bi < len(self.model.W):
                log("Number of coordinates too small to train the model", style="warning")
                return "incapable"
            probabilities = [dim / sum(dims) for dim in dims]
            block = list(np.random.multinomial(bi, probabilities))
            while 0 in block:
                block = list(np.random.multinomial(bi, probabilities))
            block.append(len(C[-1]))
            # print(f"Current block: {block}")
            return [list(C[i][:k]) for i, k in enumerate(block)]

    def stop_train(self, data):
        # stop training
        pass

    def disconnect(self):
        self.stop()
        self.sock.close()

    def attack(self):
        return self.byzantine.attack()

    def get_model(self):
        return self.model

    def set_model(self, model, batch_size=None):
        self.model = model
        if batch_size is not None:
            self.model.batch_size = batch_size
        return self

    # Special methods
    def __repr__(self):
        return f"Worker ({self.id})"

    def __str__(self):
        return f"Worker ({self.id})"


class Byzantine:

    def __init__(self, worker, knowledge: ParamServer, attack_type):
        self.worker = worker
        self.knowledge = knowledge
        self.f = knowledge.params.f
        self.attack_type = attack_type

    def attack(self):
        model_name = class_name(self.knowledge.model)
        if self.attack_type == "FOE":
            return self.fall_of_empires(model_name)
        elif self.attack_type == "LIE":
            return self.little_is_enough(model_name)
        elif self.attack_type == "CF":
            return self.crash_failure()
        else:
            NotImplementedError()

    def fall_of_empires(self, model, eps=FOE_EPS):
        if model == "DNN":
            grads = flatten_grads(self.knowledge.grads)
            try:
                mean = np.nanmean(grads, axis=0)
            except ValueError:
                log([g.shape for g in self.knowledge.grads])
                log([len(g) for g in grads])
                log(len(grads))
                elog("End")
            dims = [e.shape[-1] for e in self.knowledge.grads[0][0]] + [self.knowledge.grads[0][0][-1].shape[0]]
            return unflatten_grad(-eps * mean, dims)
        elif model == "MLR":
            Ws, Bs = fill_array(self.knowledge.grads, model)
            arWs = unflatten(-eps * np.nanmean(flatten(Ws), axis=0), Ws[0].shape)
            arBs = unflatten(-eps * np.nanmean(flatten(Bs), axis=0), Bs[0].shape)
            return arWs, arBs
        else:
            grads = fill_array(self.knowledge.grads, model)
            mean = np.nanmean(grads, axis=0)
            if np.isnan(mean).any():
                elog(np.isnan(mean).any())
            return -eps * mean

    def little_is_enough(self, model):
        n = len(self.knowledge.grads) + self.f
        s = math.floor(n / 2 + 1) - self.f
        if LIE_Z:
            z = LIE_Z
        else:
            z = st.norm.ppf((n - s) / n)

        if model == "DNN":
            grads = flatten_grads(self.knowledge.grads)
            # grads = fill_array(self.knowledge.grads)
            mean = np.nanmean(grads, axis=0)
            std = np.nanstd(grads, axis=0)
            dims = [e.shape[-1] for e in self.knowledge.grads[0][0]] + [self.knowledge.grads[0][0][-1].shape[0]]
            return unflatten_grad(mean + z * std, dims)
        elif model == "MLR":
            Ws, Bs = fill_array(self.knowledge.grads, model)
            mean_w = np.nanmean(flatten(Ws), axis=0)
            std_w = np.nanstd(flatten(Ws), axis=0)
            mean_b = np.nanmean(flatten(Bs), axis=0)
            std_b = np.nanstd(flatten(Bs), axis=0)
            arWs = unflatten(mean_w + z * std_w, Ws[0].shape)
            arBs = unflatten(mean_b + z * std_b, Bs[0].shape)
            return arWs, arBs
        else:
            grads = fill_array(self.knowledge.grads)
            mean = np.nanmean(grads, axis=0)
            std = np.std(grads, axis=0)
            return mean + z * std

    def crash_failure(self):
        n_features = self.knowledge.model.W.shape[0]
        return np.zeros((n_features, 1))
