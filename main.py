from src.datasets import get_dataset
from src.experiments import multi_run
from src.models import load_model
from src.network import ParamServer, Worker
from src.utils import load_conf, wait_until, exp_details, log, elog

if __name__ == '__main__':
    # load experiment configuration
    args = load_conf()
    args.mp = False
    args.iid = 1
    # args.iid_degree = 1
    # np.random.seed(args.seed)
    # print experiment details
    exp_details(args)
    # load dataset and initialize user's data masks
    train, test, masks = get_dataset(args)
    # elog(train.data.shxape)
    # load model
    model = load_model(train, args)
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
    log("All workers connected.", style="success")

    # todo count number of coordinates instead of grads

    # training configurations
    """
        - $C_{weak}      : "1,0,0"
        - $C_{1}$        : "0.9,0.1,0"
        - $C_{2}$        : "0.7,0.3,0"
        - $C_{powerful}$ : "0,0,1"
    """

    config = [
        # {'algo': "SGD", 'f': 0, 'gar': "average", 'lr': 0.01, 'dynamic': "1,0,0", 'legend': "SGD, $C_{weak}$"},
        # {'algo': "SGD", 'f': 0, 'gar': "average", 'lr': 0.01, 'dynamic': "0.9,0.1,0", 'legend': "SGD, $C_{1}$"},
        # {'algo': "SGD", 'f': 0, 'gar': "average", 'lr': 0.01, 'dynamic': "0.7,0.3,0", 'legend': "SGD, $C_{2}$"},
        # {'algo': "SGD", 'f': 0, 'gar': "average", 'lr': 0.01, 'dynamic': "0,0,1", 'legend': "SGD, $C_{powerful}$"},
        {'algo': "SGD", 'f': 0, 'gar': "average", 'lr': 3, 'dynamic': "0,0,1", 'legend': "SGD, $C_{powerful}$"},
        {'algo': "HgO", 'f': 0, 'gar': "average", 'lr': 3, 'dynamic': "0,0,1", 'legend': "HgO, $C_{powerful}$"},

        # {'algo': "HgO", 'f': 0, 'gar': "average", 'lr': 0.01, 'dynamic': "1,0,0", 'legend': "HgO, $C_{weak}$"},
        # {'algo': "HgO", 'f': 0, 'gar': "average", 'lr': 0.01, 'dynamic': "0.9,0.1,0", 'legend': "HgO, $C_{1}$"},
        # {'algo': "HgO", 'f': 0, 'gar': "average", 'lr': 0.01, 'dynamic': "0.7,0.3,0", 'legend': "HgO, $C_{2}$"},
        # {'algo': "HgO", 'f': 0, 'gar': "average", 'lr': 0.01, 'dynamic': "0,0,1", 'legend': "HgO, $C_{powerful}$"},
    ]

    # run the algorithm for {runs} times.
    multi_run(server, workers, train, test, config, args, debug=True, keep=True, runs=1)
