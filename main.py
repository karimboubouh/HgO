import numpy as np

from src.datasets import get_dataset
from src.experiments import multi_run
from src.models import load_model
from src.network import ParamServer, Worker
from src.utils import load_conf, wait_until, model_input, exp_details, nn_chunks

if __name__ == '__main__':
    # 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
    # load experiment configuration
    args = load_conf()
    # np.random.seed(args.seed)
    # enable or disable message passing
    # args.mp = False
    # args.iid = 1
    # args.niid_degree = 0
    # print experiment details
    exp_details(args)
    # load dataset and initialize user's data masks
    train, test, masks = get_dataset(args)
    # # load model
    mod_in = model_input(train, args)
    model = load_model(mod_in, args)
    # start server
    server = ParamServer(model, args.workers, args)
    server.start()
    # start workers
    workers = []
    for i, mask in masks.items():
        worker = Worker(i, train, test, mask, args, server)
        workers.append(worker)
        worker.start()
    # wait for all workers to connect
    wait_until(server.workers_connected)
    print(">> All workers connected.")

    # training configurations
    config = [
        {'f': 0, 'gar': "average", 'lr': 3, 'algo': "SGD", 'tau': 1000, 'legend': r"SGD, $\tau=\infty$"},
        {'f': 0, 'gar': "average", 'lr': 3, 'algo': "SGD", 'tau': 60, 'legend': r"SGD, $\tau=60$"},
        {'f': 0, 'gar': "average", 'lr': 3, 'algo': "HgO", 'tau': 60, 'legend': r"HgO, $\tau=60$"},
        {'f': 0, 'gar': "average", 'lr': 3, 'algo': "SGD", 'tau': 1, 'legend': r"SGD, $\tau=1$"},
        {'f': 0, 'gar': "average", 'lr': 3, 'algo': "HgO", 'tau': 1, 'legend': r"HgO, $\tau=1$"},
    ]

    # run the algorithm for {runs} times.
    multi_run(server, workers, train, test, config, args, debug=True, keep=True, runs=1)
