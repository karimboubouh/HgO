# Democratizing Machine Learning

## Resilient Distributed Learning with Heterogeneous Participants

> **Important** :information_source:

Check the full version of the paper including additional details on the experiments as well as all the proofs of our theoretical results. 

Paper: `Democratizing_Machine_Learning___Extended_version.pdf`

---

## Experiments preparations

### Install requirements

```pip install -r requirements.txt```

### Download datasets

- **MNIST**: download `mnist.data` from http://tiny.cc/mnist and save it in `datasets/mnist/`.
- **CIFAR10**: Download `cifar10.data` from http://tiny.cc/cifar10 and save it in `datasets/cifar/`.
- **Fashion-MNIST**: download from https://github.com/zalandoresearch/fashion-mnist and save it in `datasets/fashion/`.
- **ADULT**: download from https://archive.ics.uci.edu/ml/machine-learning-databases/adult/
- **PHISHING**: download from https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/phishing
- **BOSTON**: loaded from `sklearn.datasets`

### Description

This report contains the experiments of `HgO` on multiple models and datasets. Each model is trained on a given dataset under different configurations. We use the following combination:

- **Linear Regression**: Boston
- **Logistic regression**: Binary MNIST/ Fashion-MNIST/ CIFAR10
- **SVM**: Phishing
- **Multinomial Logistic Regression**: MNIST/ Fashion-MNIST/ CIFAR10
- **DNN**: MNIST/ Fashion-MNIST/ CIFAR10

The different configuration are based on:

- $\tau$ The unites of time the parameter server allows in each iteration.
- $\lambda_i$ Thee computation rate associated with each worker.
- $b_i$ The size of the gradient block each worker chooses to perform the computation.
- $s_i$ The mini-batch size of the worker.

All experiments are conducted on either 50 or 100 workers with $q=80\%$ of them active at each round unless stated otherwise.

We show only the main experiments however, you can run any experiment you want following the configuration provided below.

### Learning rate

We first use cross-validation to select the best hyperparameters (learning rate) for each configuration according to SGD. HgO can have better learning rates (or selected dynamically) but we perform the experiments on the same values of the learning rate. We select the following learning rate for each model:

- **Linear Regression**:
  - Boston: `lr = 0.1`
- **Logistic regression**:
  - MNIST: `lr = 0.01`
  - Fashion-MNIST: `lr = 0.001`
- **SVM**:
  - Phishing: `lr = 0.0001`
- **Multinomial Logistic Regression**: 
  - MNIST: `lr = 0.01`
  - Fashion-MNIST:  `lr = 0.001`
  - CIFAR10:  `lr = 0.01`
- **DNN**: 
  - MNIST: `lr = 3`
  - Fashion-MNIST: `lr = 3`
  - CIFAR10: `lr = 3`

## Comparing HgO and SGD

### A – HgO vs. SGD in terms of accuracy, convergence rate and gradient exchange.

- **Configuration** 

```shell
# >> Command
python main.py --model=MLR --dataset=mnist --workers=100 --batch_size=32 --tau=32 --rounds=150

# >> Configurations 
METRIC = "acc"
EVAL_ROUND = 5
BARS_ROUND = 25
PLOT_GRADS = True
# the options bellow attributes \lambda values to each worker following a normal distribution (mean, std)
WEAK_DEVICE = [500, 250] # mean / std
AVERAGE_DEVICE = [10000, 2500]
POWERFUL_DEVICE = [50000, 5000]

# >> Experiments
args.iid = 0 # Non-IID
args.iid_degree = 1 # Degree of non-iid.ness: 100% Non-IID.
config = [
    {'algo': "SGD", 'f': 0, 'gar': "average", 'lr': 0.01, 'c': "1,0,0", 'legend': "SGD, $C_{weak}$"},
    {'algo': "SGD", 'f': 0, 'gar': "average", 'lr': 0.01, 'c': "0.9,0.1,0", 'legend': "SGD, $C_{1}$"},
    {'algo': "SGD", 'f': 0, 'gar': "average", 'lr': 0.01, 'c': "0.7,0.3,0", 'legend': "SGD, $C_{2}$"},

    {'algo': "HgO", 'f': 0, 'gar': "average", 'lr': 0.01, 'c': "1,0,0", 'legend': "HgO, $C_{weak}$"},
    {'algo': "HgO", 'f': 0, 'gar': "average", 'lr': 0.01, 'c': "0.9,0.1,0", 'legend': "HgO, $C_{1}$"},
    {'algo': "HgO", 'f': 0, 'gar': "average", 'lr': 0.01, 'c': "0.7,0.3,0", 'legend': "HgO, $C_{2}$"},
]
```

- **Output**: Non-IID data (left); IID data (right)

<img src="https://tva1.sinaimg.cn/large/008i3skNgy1gxa3otd6tsj31460u0gq9.jpg" width="45%" /><img src="https://tva1.sinaimg.cn/large/008i3skNgy1gxa3owx8onj31460u0428.jpg" width="45%" />

- **Options**
  
  You can configure the algorithm to use different configurations  including models, aggregation rules, attacks, computation profiles, Byzantine works, etc.

### B – HgO .vs SGD under different proportions of weak, average and powerful devices.

- **Configuration** 

```shell
# >> Command
python main.py --model=MLR --dataset=mnist --workers=100 --batch_size=32 --tau=32 --rounds=150

# >> Configurations 
METRIC = "acc"
EVAL_ROUND = 5
WEAK_DEVICE = [500, 250]
AVERAGE_DEVICE = [10000, 2500]
POWERFUL_DEVICE = [50000, 5000]

# >> Experiments
args.iid = 0 # Non-IID
args.iid_degree = 1 # Degree of non-iid.ness: 100% Non-IID.
config = [
    {'algo': "SGD", 'f': 0, 'gar': "average", 'lr': 0.01, 'c': "1,0,0", 'legend': "SGD, $C_{weak}$"},
    {'algo': "SGD", 'f': 0, 'gar': "average", 'lr': 0.01, 'c': "0.9,0.1,0", 'legend': "SGD, $C_{1}$"},
    {'algo': "SGD", 'f': 0, 'gar': "average", 'lr': 0.01, 'c': "0.7,0.3,0", 'legend': "SGD, $C_{2}$"},
    {'algo': "SGD", 'f': 0, 'gar': "average", 'lr': 0.01, 'c': "0,0,1", 'legend': "SGD, $C_{powerful}$"},

    {'algo': "HgO", 'f': 0, 'gar': "average", 'lr': 0.01, 'c': "1,0,0", 'legend': "HgO, $C_{weak}$"},
    {'algo': "HgO", 'f': 0, 'gar': "average", 'lr': 0.01, 'c': "0.9,0.1,0", 'legend': "HgO, $C_{1}$"},
    {'algo': "HgO", 'f': 0, 'gar': "average", 'lr': 0.01, 'c': "0.7,0.3,0", 'legend': "HgO, $C_{2}$"},
    {'algo': "HgO", 'f': 0, 'gar': "average", 'lr': 0.01, 'c': "0,0,1", 'legend': "HgO, $C_{powerful}$"},
]
```

- **Output**: Test Loss (left); Test Accuracy (right)

<img src="https://tva1.sinaimg.cn/large/008i3skNgy1gxa44eemyrj31460u0aer.jpg" width="45%" /><img src="https://tva1.sinaimg.cn/large/008i3skNgy1gxa44hsq2pj31460u0dkb.jpg" width="45%" />

**Options**

You can configure the algorithm to use different configurations  including models, aggregation rules, attacks, computation profiles, Byzantine works, etc.

### C - Evaluating runtime and accuracy of HgO and SGD.

- Configuration

```shell
# >> Command
python main.py --model=DNN --dataset=mnist --workers=100 --batch_size=32 --tau=32 --rounds=2001

# >> Configurations 
METRIC = "acc"
EVAL_ROUND = 50
# for DNN devices have more computation compared to MLR otherwis they are enable to participate.
WEAK_DEVICE = [5000, 500]
AVERAGE_DEVICE = [25000, 2500]
POWERFUL_DEVICE = [50000, 5000]

# >> Experiments
args.iid = 0 # Non-IID
args.iid_degree = 1 # Degree of non-iid.ness: 100% Non-IID.
config = [
    {'algo': "SGD", 'f': 0, 'gar': "average", 'lr': 3, 'tau': 32 * 100, 'c': "0.9,0.1,0",
     'legend': r"$SGD, C_{1}, \tau=\infty$"},
    {'algo': "SGD", 'f': 0, 'gar': "average", 'lr': 3, 'tau': 32, 'c': "0.9,0.1,0",
     'legend': r"$SGD, C_{1}, \tau=32$"},
    {'algo': "HgO", 'f': 0, 'gar': "average", 'lr': 3, 'tau': 32, 'c': "0.9,0.1,0",
     'legend': r"$HgO, C_{1}, \tau=32$"},
]
```

- **Output:** Runtime under Non-IID data (left); Runtime under IID data (right)

<img src="https://tva1.sinaimg.cn/large/008i3skNgy1gxa4eptwc3j31460u0q5x.jpg" width="45%" /><img src="https://tva1.sinaimg.cn/large/008i3skNgy1gxa4ethnb5j31460u0mzz.jpg" width="45%" />

- **Options**

You can configure the algorithm to use different configurations  including models, aggregation rules, attacks, computation profiles, Byzantine works, etc.

### D - Behavior of HgO in Byzantine scenarios.

- Configuration

```shell
# >> Command
python main.py --model=DNN --dataset=mnist --workers=100 --batch_size=32 --tau=32 --rounds=2001 --attack=FOE
# FOE: Fall of Empire. You can use the Little is enough attack as follow --attack=LIE

# >> Configurations 
METRIC = "acc"
EVAL_ROUND = 50
# for DNN devices have more computation compared to MLR otherwis they are enable to participate.
WEAK_DEVICE = [5000, 500]
AVERAGE_DEVICE = [25000, 2500]
POWERFUL_DEVICE = [50000, 5000]

# >> Experiments
args.iid = 1 #IID
config = [
    {'algo': "SGD", 'f': 30, 'gar': "median", 'lr': 3, 'c': "1,0,0", 'legend': "SGD, $C_{weak}$"},
    {'algo': "SGD", 'f': 30, 'gar': "median", 'lr': 3, 'c': "0.9,0.1,0", 'legend': "SGD, $C_{1}$"},
    {'algo': "SGD", 'f': 30, 'gar': "median", 'lr': 3, 'c': "0,0,1", 'legend': "SGD, $C_{powerful}$"},

    {'algo': "HgO", 'f': 30, 'gar': "median", 'lr': 3, 'c': "1,0,0", 'legend': "HgO, $C_{weak}$"},
    {'algo': "HgO", 'f': 30, 'gar': "median", 'lr': 3, 'c': "0.9,0.1,0", 'legend': "HgO, $C_{1}$"},
    {'algo': "HgO", 'f': 30, 'gar': "median", 'lr': 3, 'c': "0,0,1", 'legend': "HgO, $C_{powerful}$"}
]
```

- **Output:** 30 Byzantine workers (left); 10 Byzantine workers (right)

<img src="https://tva1.sinaimg.cn/large/008i3skNgy1gxa4ogs29dj31460u0436.jpg" width="45%" /><img src="https://tva1.sinaimg.cn/large/008i3skNgy1gxa4ocpahgj31460u0tck.jpg" width="45%" />

- **Options**

Other aggregation rules are available: `tmean`, `krum` or `aksel`.



>  You can use a combination of datasets and models to run more experiments on `HgO`



---

## Reproducing the results of the main paper 

### Figure 1,4

```shell
# >> Command
python main.py --model=MLR --dataset=mnist --workers=100 --batch_size=32 --tau=32 --rounds=150

# >> Configurations 
METRIC = "acc"
EVAL_ROUND = 5
WEAK_DEVICE = [500, 250]
AVERAGE_DEVICE = [10000, 2500]
POWERFUL_DEVICE = [50000, 5000]

# >> Experiments
# ------------ Figure 4(a)
args.iid = 1
args.iid_degree = 0
# OR --------- Figure 1 and 4(b)
args.iid = 0
args.iid_degree = 1

config = [
    {'algo': "SGD", 'f': 0, 'gar': "average", 'lr': 0.01, 'c': "1,0,0", 'legend': "SGD, $C_{weak}$"},
    {'algo': "SGD", 'f': 0, 'gar': "average", 'lr': 0.01, 'c': "0.9,0.1,0", 'legend': "SGD, $C_{1}$"},
    {'algo': "SGD", 'f': 0, 'gar': "average", 'lr': 0.01, 'c': "0.7,0.3,0", 'legend': "SGD, $C_{2}$"},

    {'algo': "HgO", 'f': 0, 'gar': "average", 'lr': 0.01, 'c': "1,0,0", 'legend': "HgO, $C_{weak}$"},
    {'algo': "HgO", 'f': 0, 'gar': "average", 'lr': 0.01, 'c': "0.9,0.1,0", 'legend': "HgO, $C_{1}$"},
    {'algo': "HgO", 'f': 0, 'gar': "average", 'lr': 0.01, 'c': "0.7,0.3,0", 'legend': "HgO, $C_{2}$"},
]
```

### Figure 3

```shell
# >> Command
python main.py --model=MLR --dataset=mnist --workers=100 --batch_size=32 --tau=32 --rounds=150

# >> Configurations 
METRIC = "acc"
EVAL_ROUND = 5
USE_DIFFERENT_HARDWARE = True
WEAK_DEVICE = [500, 250]
AVERAGE_DEVICE = [10000, 2500]
POWERFUL_DEVICE = [50000, 5000]

# >> Experiments
args.iid = 0
args.iid_degree = 1
config = [
    {'algo': "SGD", 'f': 0, 'gar': "average", 'lr': 0.01, 'tau': 32 * 100, 'c': "0.9,0.1,0",
     'legend': r"$SGD, C_{1}, \tau=\infty$"},
    {'algo': "SGD", 'f': 0, 'gar': "average", 'lr': 0.01, 'tau': 32, 'c': "0.9,0.1,0",
     'legend': r"$SGD, C_{1}, \tau=32$"},
    {'algo': "HgO", 'f': 0, 'gar': "average", 'lr': 0.01, 'tau': 32, 'c': "0.9,0.1,0",
     'legend': r"$HgO, C_{1}, \tau=32$"},
]
```

### Figure 5

```shell
# >> Command
python main.py --model=MLR --dataset=mnist --workers=100 --batch_size=32 --tau=32 --rounds=150

# >> Configurations 
METRIC = "acc"
EVAL_ROUND = 5
# Weaker devices than preevious experiments.
WEAK_DEVICE = [250, 50]
AVERAGE_DEVICE = [5000, 500]
POWERFUL_DEVICE = [50000, 1000]

# >> Experiments
args.iid = 0
args.iid_degree = 1
config = [
        {'algo': "HgO", 'f': 0, 'gar': "average", 'lr': 0.01, 'dynamic': "1,0,0", 'legend': "HgO, $C_{weak}$"},
        {'algo': "HgO", 'f': 0, 'gar': "average", 'lr': 0.01, 'dynamic': "0.9,0.1,0", 'legend': "HgO, $C_{1}$"},
        {'algo': "HgO", 'f': 0, 'gar': "average", 'lr': 0.01, 'dynamic': "0.7,0.3,0", 'legend': "HgO, $C_{2}$"},
        {'algo': "HgO", 'f': 0, 'gar': "average", 'lr': 0.01, 'dynamic': "0,0,1", 'legend': "HgO, $C_{powerful}$"},
    ]
```

### Figure 6 (a)

```shell
# >> Command
python main.py --model=MLR --dataset=mnist --workers=100 --batch_size=32 --tau=32 --rounds=150 --attack=FOE

# >> Configurations 
METRIC = "acc"
EVAL_ROUND = 5
WEAK_DEVICE = [500, 250]
AVERAGE_DEVICE = [10000, 2500]
POWERFUL_DEVICE = [50000, 5000]

# >> Experiments
args.iid = 0
args.iid_degree = 1
config = [
    {'algo': "HgO", 'f': 0, 'gar': "average", 'lr': 0.01, 'c': "1,0,0", 'legend': "HgO, $No Attack$"},
    {'algo': "HgO", 'f': 10, 'gar': "median", 'lr': 0.01, 'c': "0.9,0.1,0", 'legend': "HgO, $f=10$"},
    {'algo': "HgO", 'f': 10, 'gar': "krum", 'lr': 0.01, 'c': "0.9,0.1,0", 'legend': "HgO, $f=10$"},
    {'algo': "HgO", 'f': 10, 'gar': "aksel", 'lr': 0.01, 'c': "0.9,0.1,0", 'legend': "HgO, $f=10$"},
]
```

### Figure 6 (b)

```shell
# >> Command
python main.py --model=MLR --dataset=mnist --workers=100 --batch_size=32 --tau=32 --rounds=150 --attack=FOE
...
config = [
    {'algo': "HgO", 'f': 0, 'gar': "average", 'lr': 0.01, 'c': "1,0,0", 'legend': "HgO, No Attack"},
    {'algo': "HgO", 'f': 10, 'gar': "median", 'lr': 0.01, 'attack': "FOE", 'c': "0.9,0.1,0", 'legend': "HgO, f=10, FOE"},
    {'algo': "HgO", 'f': 10, 'gar': "median", 'lr': 0.01, 'attack': "LIE", 'c': "0.9,0.1,0", 'legend': "HgO, f=10, LIE"},
]
```

### Figure 6 (c)

```shell
# >> Command
python main.py --model=MLR --dataset=mnist --workers=100 --batch_size=32 --tau=32 --rounds=150
...
config = [
    {'algo': "HgO", 'f': 0, 'gar': "average", 'lr': 0.01, 'c': "1,0,0", 'legend': "HgO, $f=0, C_1$"},
    {'algo': "HgO", 'f': 5, 'gar': "median", 'lr': 0.01, 'c': "0.9,0.1,0", 'legend': "HgO, $f=5, C_1$"},
    {'algo': "HgO", 'f': 10, 'gar': "median", 'lr': 0.01, 'c': "0.9,0.1,0", 'legend': "HgO, $f=10, C_1$"},
    {'algo': "HgO", 'f': 20, 'gar': "median", 'lr': 0.01, 'c': "0.9,0.1,0", 'legend': "HgO, $f=20, C_1$"},
    {'algo': "HgO", 'f': 30, 'gar': "median", 'lr': 0.01, 'c': "0.9,0.1,0", 'legend': "HgO, $f=30, C_1$"},
]
```

### Figure 7 (a)

```shell
# >> Command
python main.py --model=DNN --dataset=mnist --workers=100 --batch_size=32 --tau=32 --rounds=2001

# >> Configurations 
METRIC = "acc"
EVAL_ROUND = 50
WEAK_DEVICE = [5000, 500]
AVERAGE_DEVICE = [25000, 2500]
POWERFUL_DEVICE = [50000, 5000]

# >> Experiments
args.iid = 0
args.iid_degree = 1
config = [
    {'algo': "SGD", 'f': 0, 'gar': "average", 'lr': 3, 'c': "1,0,0", 'legend': "SGD, $C_{weak}$"},
    {'algo': "SGD", 'f': 0, 'gar': "average", 'lr': 3, 'c': "0.9,0.1,0", 'legend': "SGD, $C_{1}$"},
    {'algo': "SGD", 'f': 0, 'gar': "average", 'lr': 3, 'c': "0.7,0.3,0", 'legend': "SGD, $C_{2}$"},
    {'algo': "SGD", 'f': 0, 'gar': "average", 'lr': 3, 'c': "0,0,1", 'legend': "SGD, $C_{powerful}$"},

    {'algo': "HgO", 'f': 0, 'gar': "average", 'lr': 3, 'c': "1,0,0", 'legend': "HgO, $C_{weak}$"},
    {'algo': "HgO", 'f': 0, 'gar': "average", 'lr': 3, 'c': "0.9,0.1,0", 'legend': "HgO, $C_{1}$"},
    {'algo': "HgO", 'f': 0, 'gar': "average", 'lr': 3, 'c': "0.7,0.3,0", 'legend': "HgO, $C_{2}$"},
    {'algo': "HgO", 'f': 0, 'gar': "average", 'lr': 3, 'c': "0,0,1", 'legend': "HgO, $C_{powerful}$"},
]
```

### Figure 7 (b)

```shell
# >> Command
python main.py --model=DNN --dataset=mnist --workers=100 --batch_size=32 --tau=32 --rounds=2001
...
USE_DIFFERENT_HARDWARE = True
...
config = [
    {'algo': "SGD", 'f': 0, 'gar': "average", 'lr': 3, 'tau': 32 * 100, 'c': "0.9,0.1,0",
     'legend': r"$SGD, C_{1}, \tau=\infty$"},
    {'algo': "SGD", 'f': 0, 'gar': "average", 'lr': 3, 'tau': 32, 'c': "0.9,0.1,0",
     'legend': r"$SGD, C_{1}, \tau=32$"},
    {'algo': "HgO", 'f': 0, 'gar': "average", 'lr': 3, 'tau': 32, 'c': "0.9,0.1,0",
     'legend': r"$HgO, C_{1}, \tau=32$"},
]
```

---

End.
