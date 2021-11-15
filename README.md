# Democratizing Machine Learning

## Resilient Distributed Learning with Heterogeneous Participants



> **Important** :information_source:

Check the full version of the paper including additional details on the experiments as well as all the proofs of our theoretical results. 

Paper: `Democratizing_Machine_Learning___Extended_version.pdf`



> **TODO**

- Include selection strategies in paper
- Include additional experiments in extended version of the paper
- Update the section on the android app
- Update the code of the android app

---

## Experiments preparations

### Install requirements

```pip install -r requirements.txt```

### Download datasets

- **MNIST**: download `mnist.data` from http://tiny.cc/mnist and save it in `datasets/mnist/`.
- **CIFAR10**: Download `cifar10.data` from http://tiny.cc/cifar10 and save it in `datasets/cifar/`.
- **ADULT**: download from https://archive.ics.uci.edu/ml/machine-learning-databases/adult/
- **PHISHING**: download from https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/phishing
- **BOSTON**: loaded from `sklearn.datasets`

### Description

This report contains extensive experiments of `HgO` on multiple models and datasets. Each model is trained on a given dataset under different configurations. We use the following combination:

- **Linear Regression**: Boston dataset (14 features)
- **Logistic regression**: MNIST/CIFAR10 dataset (784 features, 3072 features)
- **SVM**: Phishing dataset (68 features)
- **DNN**: MNIST/CIFAR10 dataset (784 features, 3072 features )

The different configuration are based on:

- $\tau$ The unites of time the parameter server allows in each iteration.
- $\lambda_i$ Thee computation rate associated with each worker.
- $b_i$ The size of the gradient block each worker chooses to perform the computation.
- $s_i$ The mini-batch size of the worker.

We first use cross-validation to select the best hyperparameters (learning rate) for each configuration.

All experiments are conducted on either 50 or 100 workers with $q=80\%$ of them active at each round unless stated otherwise.



## Comparing HgO and SGD



### A – SGD and HgO under different values of $\tau$

- **Configuration** 

```shell
python main.py --model=DNN --dataset=mnist --workers=100 --batch_size=64 --q=1 --rounds=300

options: 
- iid = 0
- niid_degree = 1

config2 = [
    {'f': 0, 'gar': "average", 'lr': 3, 'algo': "SGD", 'tau': 1000, 'legend': r"SGD, $\tau=\infty$"},
    {'f': 0, 'gar': "average", 'lr': 3, 'algo': "SGD", 'tau': 60, 'legend': r"SGD, $\tau=60$"},
    {'f': 0, 'gar': "average", 'lr': 3, 'algo': "HgO", 'tau': 60, 'legend': r"HgO, $\tau=60$"},
    {'f': 0, 'gar': "average", 'lr': 3, 'algo': "SGD", 'tau': 1, 'legend': r"SGD, $\tau=1$"},
    {'f': 0, 'gar': "average", 'lr': 3, 'algo': "HgO", 'tau': 1, 'legend': r"HgO, $\tau=1$"},
]
```

- **Output**: Non-IID data (left); IID data (right)

<img src="https://tva1.sinaimg.cn/large/008i3skNgy1gvhc5yqjtlj613t0u0wik02.jpg" alt="Screen Shot 2021-10-16 at 11.36.41" width="45%" /><img src="https://tva1.sinaimg.cn/large/008i3skNgy1gvhc8actyzj613t0u0ads02.jpg" alt="Screen Shot 2021-10-16 at 11.36.32" width="45%" />

### B – Number of received gradients in case of SGD and HgO under different values of $\tau$

- **Configuration** 

```shell
python main.py --model=DNN --dataset=mnist --workers=100 --batch_size=64 --q=1 --rounds=300

options: 
- iid = 0
- niid_degree = 1

config2 = [
    {'f': 0, 'gar': "average", 'lr': 3, 'algo': "SGD", 'tau': 1000, 'legend': r"SGD, $\tau=\infty$"},
    {'f': 0, 'gar': "average", 'lr': 3, 'algo': "SGD", 'tau': 60, 'legend': r"SGD, $\tau=60$"},
    {'f': 0, 'gar': "average", 'lr': 3, 'algo': "HgO", 'tau': 60, 'legend': r"HgO, $\tau=60$"},
    {'f': 0, 'gar': "average", 'lr': 3, 'algo': "SGD", 'tau': 1, 'legend': r"SGD, $\tau=1$"},
    {'f': 0, 'gar': "average", 'lr': 3, 'algo': "HgO", 'tau': 1, 'legend': r"HgO, $\tau=1$"},
]
```

- **Output:**
-  <img src="https://tva1.sinaimg.cn/large/008i3skNgy1gvhcbft9d1j613t0u0q4u02.jpg" alt="Screen Shot 2021-10-16 at 11.43.32" width="45%" />



## Byzantine impact vs convergence properties



### A - Behavior of HgO in the presence of Byzantine workers using different GARs and different attacks.

- Configuration

```
python main.py --model=LR --dataset=mnist --workers=50 --q=0.8 --rounds=300 --attack=FOE
python main.py --model=LR --dataset=mnist --workers=50 --q=0.8 --rounds=300 --attack=LIE

config = [
    {'f': 0, 'gar': "average", 'lr': 0.01, 'legend': "HgO, f=0, ar=avg"},
    {'f': 25, 'gar': "average", 'lr': 0.01, 'legend': "HgO, f=25, ar=avg"},
    {'f': 25, 'gar': "median", 'lr': 0.01, 'legend': "HgO, f=25, ar=median"},
    {'f': 25, 'gar': "aksel", 'lr': 0.01, 'legend': "HgO, f=25, ar=aksel"},
    {'f': 25, 'gar': "krum", 'lr': 0.01, 'legend': "HgO, f=25, ar=krum"},
]
```

- **Output:** using Fall Of Empire attack (left) and Little Is Enough attack (right)

  

  <img src="https://tva1.sinaimg.cn/large/008i3skNgy1gvhcjhj8v3j613t0u0q6902.jpg" alt="Screen Shot 2021-10-16 at 11.50.09" width="45%" /><img src="https://tva1.sinaimg.cn/large/008i3skNgy1gvhck422q6j613t0u0n0l02.jpg" alt="Screen Shot 2021-10-16 at 11.50.15" width="45%" />



### B – Behavior of HgO with different number of Byzantine workers

- **Configuration**

```
python main.py --model=LR --dataset=mnist --workers=100 --q=0.8 --rounds=150 --attack=FOE
python main.py --model=LR --dataset=mnist --workers=100 --q=0.8 --rounds=150 --attack=LIE

config = [
    {'f': 0, 'gar': "median", 'lr': 0.01, 'legend': "HgO, $f=0, q=0.8$"},
    {'f': 10, 'gar': "median", 'lr': 0.01, 'legend': "HgO, $f=10, q=0.8$"},
    {'f': 25, 'gar': "median", 'lr': 0.01, 'legend': "HgO, $f=25, q=0.8$"},
    {'f': 39, 'gar': "median", 'lr': 0.01, 'legend': "HgO, $f=39, q=0.8$"},
]
```

**Output**: using FOE (left) and LIE (right)

<img src="https://tva1.sinaimg.cn/large/008i3skNgy1gvhcmxe4dwj613t0u041c02.jpg" alt="Screen Shot 2021-10-16 at 11.53.56" width="45%" /><img src="https://tva1.sinaimg.cn/large/008i3skNgy1gvhcn1xmk7j613t0u0gor02.jpg" alt="Screen Shot 2021-10-16 at 11.54.02" width="45%" />



### C – Behavior of HgO under different attacks

**Configuration**

```
python main.py --model=LR --dataset=mnist --workers=100 --q=0.8 --rounds=150

config = [
    {'f': 0, 'gar': "median", 'lr': 0.01, 'legend': "HgO, $f=0, No attack$"},
    {'f': 10, 'gar': "median", 'lr': 0.01, 'attack': "FOE", 'legend': "HgO, $f=10, FOE$"},
    {'f': 10, 'gar': "median", 'lr': 0.01, 'attack': "LIE", 'legend': "HgO, $f=10, LIE$"},
]
```

**Output**: 

<img src="https://tva1.sinaimg.cn/large/008i3skNgy1gvhcookd8vj613t0u0n0302.jpg" alt="Screen Shot 2021-10-16 at 11.56.13" width="45%" />



## Varying the hardware profile of the network

### A - Behavior of HgO with heterogeneous workers (small, average, and powerful workers)

**Configuration**

```
python main.py --model=LR --dataset=mnist --workers=100 --q=0.8 --rounds=300

config = [
    {'f': 0, 'gar': "average", 'lr': 0.01, 'dynamic': "1.0,0.0,0.0", 'legend': "HgO, $C_{weak}$"},
    {'f': 0, 'gar': "average", 'lr': 0.01, 'dynamic': "0.9,0.0,0.1", 'legend': "HgO, $C_{min}$"},
    {'f': 0, 'gar': "average", 'lr': 0.01, 'dynamic': "0.7,0.2,0.1", 'legend': "HgO, $C_0$"},
    {'f': 0, 'gar': "average", 'lr': 0.01, 'dynamic': "0.3,0.5,0.2", 'legend': "HgO, $C_2$"},
    {'f': 0, 'gar': "average", 'lr': 0.01, 'dynamic': "0.1,0.2,0.7", 'legend': "HgO, $C_3$"},
    {'f': 0, 'gar': "average", 'lr': 0.01, 'dynamic': "0.0,0.0,1.0", 'legend': "HgO, $C_{powerful}$"},
]
# note : "0.3,0.5,0.2" represent the distribution of the connected workers: 30% weak, 50% average and 20% powerful
WEAK_DEVICE = 100
AVERAGE_DEVICE = 1000
POWERFUL_DEVICE = 100000
```

**Output**: 

<img src="https://tva1.sinaimg.cn/large/008i3skNgy1gvhcxquj0hj613t0u0tbn02.jpg" alt="Screen Shot 2021-10-16 at 12.02.17" width="45%" />





## Effect of block and mini-batch selection strategies

  

### A - Behavior of HgO with different block and mini-batch selection strategies: Coordinates First, Data First and Hybrid.



- Configuration

```
python main.py --model=LR --dataset=mnist --workers=100 --q=0.8 --rounds=300

==> weak
config = [
    {'f': 0, 'gar': "average", 'lr': 0.01, 'dynamic': "1.0,0.0,0.0", 'strategy': "CoordinatesFirst",
     'legend': r"HgO, $\lambda=weak, CoordinatesFirst$"},
    {'f': 0, 'gar': "average", 'lr': 0.01, 'dynamic': "1.0,0.0,0.0", 'strategy': "DataFirst",
     'legend': r"HgO, $\lambda=weak, DataFirst$"},
    {'f': 0, 'gar': "average", 'lr': 0.01, 'dynamic': "1.0,0.0,0.0", 'strategy': "Hybrid",
     'legend': r"HgO, $\lambda=weak, Hybrid, \rho=0.5$"},
]

==> average
config = [
    {'f': 0, 'gar': "average", 'lr': 0.01, 'dynamic': "0.0,1.0,0.0", 'strategy': "CoordinatesFirst",
     'legend': r"HgO, $\lambda_{average, CoordinatesFirst}$"},
    {'f': 0, 'gar': "average", 'lr': 0.01, 'dynamic': "0.0,1.0,0.0", 'strategy': "DataFirst",
     'legend': r"HgO, $\lambda_{average, DataFirst}$"},
    {'f': 0, 'gar': "average", 'lr': 0.01, 'dynamic': "0.0,1.0,0.0", 'strategy': "Hybrid",
     'legend': r"HgO, $\lambda_{average, Hybrid, \rho=0.5}$"},
]

==> powerful
config = [
    {'f': 0, 'gar': "average", 'lr': 0.01, 'dynamic': "0.0,1.0,0.0", 'strategy': "CoordinatesFirst",
     'legend': r"HgO, $\lambda_{powerful, CoordinatesFirst}$"},
    {'f': 0, 'gar': "average", 'lr': 0.01, 'dynamic': "0.0,1.0,0.0", 'strategy': "DataFirst",
     'legend': r"HgO, $\lambda_{powerful, DataFirst}$"},
    {'f': 0, 'gar': "average", 'lr': 0.01, 'dynamic': "0.0,1.0,0.0", 'strategy': "Hybrid",
     'legend': r"HgO, $\lambda_{powerful, Hybrid, \rho=0.5}$"},
]
```

**Output:** Weak workers (left), average workers (middle), powerful workers (right) 

<img src="https://tva1.sinaimg.cn/large/008i3skNgy1gvhdpu1gs5j613t0u0wi302.jpg" alt="Screen Shot 2021-10-16 at 12.12.53" width="30%" /><img src="https://tva1.sinaimg.cn/large/008i3skNgy1gvhdpkd9adj613t0u0gpr02.jpg" alt="Screen Shot 2021-10-16 at 12.13.02" width="30%" /><img src="https://tva1.sinaimg.cn/large/008i3skNgy1gvhdppvozhj613t0u078502.jpg" alt="Screen Shot 2021-10-16 at 12.13.07" width="30%" />



---

You can use a combination of datasets and models to run more experiments on `HgO`.

**End.** 
