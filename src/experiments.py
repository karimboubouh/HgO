import copy
import uuid

import numpy as np

from src.config import ACC_TIME, WEAK_DEVICE, AVERAGE_DEVICE, POWERFUL_DEVICE, ONE_DEVICE, EVAL_ROUND, TRACK_ACC
from src.plots import plot_std
from src.network import ParamServer
from src.utils import save, percentage, dynamic_lambda


def run(server: ParamServer, workers, train, test, config, args, log="acc", keep=True, debug=True, seed=None):
    logs = []
    tc = []
    tt = []
    for c in config:
        print(f">> Configuration : {c}")
        # load model
        server.model = copy.deepcopy(server.raw_model)
        # configure byzantine settings
        server.params.f = c['f']
        server.params.GAR = c['gar']
        # optional settings
        if c.get('lr', None):
            server.params.lr = c['lr']
        if c.get('tau', None):
            server.params.tau = c['tau']
        if c.get('q', None):
            server.params.q = c['q']
        if c.get('v', None):
            server.params.v = c['v']
        if c.get('strategy', None):
            for worker in workers:
                worker.params.block_strategy = c['strategy']
        if c.get('algo', None):
            server.params.algo = c['algo']
            for worker in workers:
                worker.params.algo = c['algo']
        if c.get('attack', None):
            server.params.attack = c['attack']
        if c.get('dynamic', None):
            d = dynamic_lambda(args.workers, c['dynamic'])
            print(f">> Distribution of the computation power <{c['dynamic']}>")
            for i, worker in enumerate(workers):
                worker.lamda = d[i]
        else:
            d = dynamic_lambda(args.workers, "0.3,0.4,0.3")
            print(f">> Distribution of the computation power <0.3,0.4,0.3>")
            for i, worker in enumerate(workers):
                worker.lamda = d[i]

        server.init_byzantine_workers()
        print(f">> Byzantine workers: {server.byz_indices}")
        # define the optimization strategy to use
        server.opt_strategy(args)
        # join decentralized training
        server.start_train(args)
        if TRACK_ACC == "Test":
            acc_time = server.train(test.data, test.targets)
        else:
            acc_time = server.train(train.data, train.targets)
        # evaluate the performance of the resulted model
        if log == "acc":
            histo = [acc for loss, acc in server.history]
        else:
            histo = [loss for loss, acc in server.history]
        logs.append(histo)
        grad_time = np.sum(server.grad_time)
        if debug:
            # summary of learning
            server.summary(train, test)
            # for block {c['block']}
            print(f"\033[1m\033[94m>> Time complexity to calculate gradients: {grad_time:.4f} seconds.\033[0m")
            print(f"\033[1m\033[94m>> Time complexity to reach 95% accuracy: {acc_time:.4f} seconds.\033[0m")

        print('----------------------------------------------------------------------\n')

        tc.append(grad_time)
        tt.append(acc_time)
        # reset server params
        server.reset()

    if keep:
        save(f"honest_GD_vs_CD_{args.model}_{uuid.uuid4().hex.upper()[0:6]}", logs)

    return logs, tc, tt


def multi_run(server, workers, train, test, config, args, runs=10, keep=True, debug=True, seed=True):
    print(f">> Evaluation {TRACK_ACC} data each {EVAL_ROUND} rounds.")
    general_logs = []
    general_tc = []  # time complexity
    general_tt = []  # time to reach {ACC_TIME} accuracy
    metric = "loss" if args.model == "LN" else "acc"
    # activate keep in case forgotten
    if runs > 1:
        keep = True
    # run the algorithm for {runs} time(s)
    for i in range(runs):
        print(f"\n********** RUN {i} *************************************************************")
        seed = args.seed if seed else None
        logs, tc, tt = run(server, workers, train, test, config, args, log=metric, keep=False, debug=debug)
        general_logs.append(logs)
        general_tc.append(tc)
        general_tt.append(tt)
    # construct logs for stats and plotting
    blocks = {i: [] for i, c in enumerate(config)}
    for exp in general_logs:
        for i, b in enumerate(exp):
            blocks[i].append(b)
    blocks_mean = [(np.mean(block, axis=0), config[index]["legend"]) for index, block in blocks.items()]
    blocks_std = [(np.std(block, axis=0)) for block in blocks.values()]
    # mean_tc = np.mean(general_tc, axis=0)
    # mean_tt = np.mean(general_tt, axis=0)
    # print("MEAN Time Complexity:")
    # b={c['block']}
    # tc_table = {f"{i}-->": (round(mean_tc[i], 8), percentage(mean_tc[i], mean_tc[0])) for i, c in enumerate(config)}
    # print(tc_table)
    # print(f"MEAN Time to reach {ACC_TIME * 100}%:")
    # b={c['block']}
    # tt_table = {f"{i}-->": (round(mean_tt[i], 8), percentage(mean_tt[i], mean_tt[0])) for i, c in enumerate(config)}
    # print(tt_table)
    # save logs
    unique = uuid.uuid4().hex.upper()[0:6]
    if keep:
        save(f"./out/EXP_{args.model}_{unique}.p", (blocks_mean, blocks_std))
    # Plot accuracies
    metric = "Loss" if args.model == "LN" else "Accuracy"
    info = {'ylabel': f"{TRACK_ACC} {metric}", 'xlabel': "Rounds", 'title': "SmartFed vs. Stochastic Gradient Descent"}
    plot_std(blocks_mean, blocks_std, info, unique=unique)

    return blocks_mean, blocks_std
