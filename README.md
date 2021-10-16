# SmartFed: Robust and Lightweight Distributed Machine Learning

## Experiments preparations

### Install requirements

```pip install -r requirements.txt```

### Download datasets

- **MNIST**: download from https://mega.nz/file/3aA2VLLD#vdW4CjZg3TRqD2qECdGgLZ9Iu8N0w2EAobQV298Hqi8 and save it in `datasets/mnist/`.
- **ADULT**: download from https://archive.ics.uci.edu/ml/machine-learning-databases/adult/
- **PHISHING**: download from https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/phishing
- **BOSTON**: loaded from `sklearn.datasets`

### Description

This report contains extensive experiments of `SmartFed` on multiple models and datasets. Each model is trained on a given dataset under different configurations. We use the following combination:

- **Linear Regression**: Boston dataset (14 features)
- **Logistic regression**: MNIST dataset (784 features)
- **SVM**: Phishing dataset (68 features)
- **DNN**: MNIST dataset (784 features)

The different configuration are based on:

- Static and dynamic block sizes $b$.
- Static and dynamic mini-batch sizes $s$.

We first use cross-validation to select the best hyperparameters (learning rate) for each configuration.

The time complexity is measured as the sum of time performed to calculate the gradients over all rounds, because the time to calculate the gradients in one rounds is so small. However, the percentage is the same between different configurations.

All experiments are conducted on either 25 or 100 workers with 80% of them active at each round.



## SmartFed in a safe environment



### Light SmartFed (LSF)

In LSF, we consider the case of weak smartphones. Every active smartphone will calculate only one element of the gradient vector ($b = 1$) using only one data sample ($s = 1$) at each round.

We compare the performance of LSF with both SGD and SF($b=1, s=S$), where $S$ is the set of data samples available locally in each smartphone.

#### Linear regression: SGD ($s = S$)  vs. SF($b=1, s=S$)

**Configuration**

```python
# details
dataset    : boston (14 features)
block size : 1
batch size : S
rounds     : 150
# console
python main.py --model=LN --dataset=boston --batch_size=20 --workers=25 --frac=0.8 --rounds=150
# main.py
config = [
    {'block': 0, 'lr': 0.1, 'f': 0, 'gar': "median", 'legend': "SGD, GAR=median"},
    {'block': 1, 'lr': 0.6, 'f': 0, 'gar': "median", 'legend': "LSF, SAR=median"},
]
```

**Results**

- Train Loss

<img src="https://tva1.sinaimg.cn/large/008i3skNgy1gr6qn29bzlj31400u0jt3.jpg" alt="Figure_IMG_694" width="50%" />

#### Linear regression: SGD ($s = 1$) vs. LSF

**Configuration**

```python
# details
...
batch size : 1
# console
python main.py --model=LN --dataset=boston --batch_size=1 --workers=25 --frac=0.8 --rounds=150
...
```

**Results**

- Train Loss

<img src="https://tva1.sinaimg.cn/large/008i3skNgy1gr6qnogx7aj31400u00uw.jpg" alt="Figure_IMG_340" width="50%" />

##### Observations

- LSF convergence behavior is similar to SGD despite using only one coordinate of the gradient and one data sample per round.
- Training using mini-batch instead of only one data sample ($s > 1$) results in same convergence as SGD.

---

#### Logistic regression: SGD ($s = S$)  vs. SF($b=1, s=S$)

**Configuration**

```python
# details
dataset    : mnist (784 features)
block size : 1
batch size : S # S=508
rounds     : 150
# console
python main.py --model=LR --dataset=mnist --batch_size=508 --workers=25 --frac=0.8 --rounds=150
# main.py
config = [
    {'block': 0, 'lr': 0.01, 'f': 0, 'gar': "median", 'legend': "SGD, GAR=median"},
    {'block': 1, 'lr': 1, 'f': 0, 'gar': "median", 'legend': "LSF, SAR=median"},
]
```

**Results**

- Train Accuracy 

<img src="https://tva1.sinaimg.cn/large/008i3skNgy1gr6qqom4ntj31400u0q50.jpg" alt="Figure_IMG_225" width="50%" />

- Time complexity

| 25 workers                                     | SGD           | SF, b = 1         |
| ---------------------------------------------- | ------------- | ----------------- |
| Time complexity (seconds)                      | 0.0593 (100%) | 0.0022 **(3.8%)** |
| Time spent to reach **85%** accuracy (seconds) | 0.3013 (100%) | 2.1599 (730.86%)  |

#### Logistic regression: SGD ($s = 1$) vs. LSF

**Configuration**

```python
# details
...
batch size : 1
# console
python main.py --model=LR --dataset=mnist --batch_size=1 --workers=25 --frac=0.8 --rounds=300
...
```

**Results**

- Train Accuracy 

<img src="https://tva1.sinaimg.cn/large/008i3skNgy1gr6qrkh9xuj31400u0tb8.jpg" alt="Figure_IMG_236" width="50%" />

- Time complexity

| 25 workers                                     | SGD           | SF, b = 1           |
| ---------------------------------------------- | ------------- | ------------------- |
| Time complexity (seconds)                      | 0.0039 (100%) | 0.0028 **(72.13%)** |
| Time spent to reach **85%** accuracy (seconds) | —             | —                   |

##### Observations

- LSF convergence to a reasonable accuracy despite using only one coordinate of the gradient out of 784.
- The cost of one LSF iteration is significantly lower than SGD, specially in training on multiple data samples (3.8% of SGD cost).
- Convergence speed of LSF is much slower than SGD. That’s a tradeoff between computation and speed of convergence.

----

#### SVM: SGD ($s = S$)  vs. SF($b=1, s=S$)

**Configuration**

```python
# details
dataset    : phishing (68 features)
block size : 1
batch size : S # S=353
rounds     : 150
# console
python main.py --model=SVM --dataset=phishing --batch_size=353 --workers=25 --frac=0.8 --rounds=150
# main.py
config = [
    {'block': 0, 'lr': 0.01, 'f': 0, 'gar': "median", 'legend': "SGD, GAR=median"},
    {'block': 1, 'lr': 0.1, 'f': 0, 'gar': "median", 'legend': "LSF, SAR=median"},
]
```

**Results**

- Train Accuracy

<img src="https://tva1.sinaimg.cn/large/008i3skNgy1gr6qsy7shtj31400u076e.jpg" alt="Figure_IMG_300" width="50%" />

#### SVM: SGD ($s = 1$) vs. LSF

**Configuration**

```python
# details
...
batch size : 1
# console
python main.py --model=LR --dataset=mnist --batch_size=1 --workers=25 --frac=0.8 --rounds=300
...
```

**Results**

- Train Accuracy 

<img src="https://tva1.sinaimg.cn/large/008i3skNgy1gr6qtljou8j31400u0acp.jpg" alt="Figure_IMG_614" width="50%"/>

##### Observations

- SVM training is more stable with SmartFed.
- The bigger the feature space, the greater the gain in time complexity per round.
- Using SF($b=1, s=S$) results in almost similar convergence behavior.

---

#### Deep Neural Network: SGD ($s = S$)  vs. SF($b=1, s=S$)

**Configuration**

```python
# details
dataset    : mnist (784 features)
block size : 1
batch size : S # S=2400
rounds     : 5000
# console
python main.py --model=DNN --dataset=mnist --batch_size=2400 --workers=25 --frac=0.8 --rounds=5000
# main.py
config = [
    {'block': [784, 30, 10], 'lr': 3.0, 'f': 0, 'gar': "median", 'legend': "SGD, GAR=median"},
    {'block': [1, 1, 10], 'lr': 3.0, 'f': 0, 'gar': "median", 'legend': "LSF, SAR=median"},
]
# utils.py
EVAL_ROUND = 100
```

**Results**

- Train Accuracy

<img src="https://tva1.sinaimg.cn/large/008i3skNgy1gr6qv7w1hnj31400u0tao.jpg" alt="Figure_IMG_571" width="50%" />

#### Deep Neural Network: SGD ($s = 1$) vs. LSF

**Configuration**

```python
# details
dataset    : mnist (784 features)
block size : 1
batch size : 1
rounds     : 5000
# console
python main.py --model=DNN --dataset=mnist --batch_size=1 --workers=25 --frac=0.8 --rounds=5000
...
```

**Results**

- Train Accuracy

<img src="https://tva1.sinaimg.cn/large/008i3skNgy1gr6qwg6m1mj31400u0jtf.jpg" alt="Figure_IMG_667" width="50%" />

##### Observations

- LSF does not work with complex models, which is normal otherwise there will be no need for non-linear models.

---



### SmartFed (SF) under different smartphones configurations

A network of mobile phones contains a wide variety of smartphones with different computation and storage capabilities. Thus, it is not appropriate to require from smartphones participating in training to perform the same computation.

In SF, we assume that smartphones have different capacities ranging from weak to powerful. At each round a smartphone device will calculate a block of coordinates of the gradient vector ranging from one coordinate (LSF) to the full gradient vector (SGD) depending on its computation power.

We define tree types of smartphones configurations:

- Weak smartphones: `batch_size: 16`.
- Average smartphones:  `batch_size: 128`.
- Powerful smartphones: `batch_size: full data`.

In this set of experiments, we assume that all smartphones have the same configuration, and we measure how well they can do compared to SGD.

#### Linear regression: $b \in \{ 1, 4 ,8 \}$ and $s \in \{ 16, 128 ,S \}$

**Configuration**

```python
# details
dataset    : boston (14 features)
block size : 1, 4, 8
batch size : 16, 128, S
rounds     : 50
# console
python main.py --model=LN --dataset=boston --batch_size=16 --workers=25 --frac=0.8 --rounds=50
python main.py ... --batch_size=128
python main.py ... --batch_size=20 # value 0 means train on all local data (S).
# main.py
config = [
    {'block': 0, 'lr': 0.1, 'f': 0, 'gar': "median", 'legend': "SGD, GAR=median"},
    {'block': 1, 'lr': 0.9, 'f': 0, 'gar': "median", 'legend': "SF, SAR=median, b=1"},
    {'block': 4, 'lr': 0.6, 'f': 0, 'gar': "median", 'legend': "SF, GAR=median, b=4"},
    {'block': 8, 'lr': 0.3, 'f': 0, 'gar': "median", 'legend': "SF, GAR=median, b=8"},
]
```

**Results**

- Train Loss

<img src="https://tva1.sinaimg.cn/large/008i3skNgy1gr6r0e1dcjj31400u0jtr.jpg" alt="Figure_IMG_414" width="33%" /> <img src="https://tva1.sinaimg.cn/large/008i3skNgy1gq6270tiamj31400u00xe.jpg" alt="Figure_IMG_718" width="33%" /><img src="https://tva1.sinaimg.cn/large/008i3skNgy1gq6289197ij31400u00xe.jpg" alt="Figure_IMG_396" width="33%" />

- Time complexity


| 25 workers                 | SGD           | HSF, b = 1       | HSF, b = 4       | HSF, b = 8       |
| -------------------------- | ------------- | ---------------- | ---------------- | ---------------- |
| Weak — Time complexity     | 0.0003 (100%) | 0.0003 (103.71%) | 0.0003 (98.64%)  | 0.0003 (101.53%) |
| Average — Time complexity  | 0.0003 (100%) | 0.0003 (98.56%)  | 0.0003 ( 97.68%) | 0.0003 (101.63%) |
| Powefull — Time complexity | 0.0003 (100%) | 0.0003 (108.87%) | 0.0003 (102.33%) | 0.0003 (106.12%) |

##### Observations

- All variantes of HSF with a small data set (Boston) converge to the same training loss as SGD.

- HSF($b=8$) and HSF($b=4$) converge even faster than SGD.
- As Boston dataset has only 14 features, the difference in computation time is negligible.

---

#### Logistic regression: $b \in \{ 1, 32 ,128, 256 \}$ and $s \in \{ 16, 128 ,S \}$

**Configuration**

```python
# details
dataset    : mnist (784 features)
block size : 1, 32, 128, 256
batch size : 16, 128, S
rounds     : 150
# console
python main.py --model=LR --dataset=mnist --batch_size=16 --workers=25 --frac=0.8 --rounds=100
python main.py ... --batch_size=128
python main.py ... --batch_size=0
# main.py
config = [
    {'block': 0, 'lr': 0.01, 'f': 0, 'gar': "median", 'legend': "SGD, GAR=median"},
    {'block': 1, 'lr': 1, 'f': 0, 'gar': "median", 'legend': "SF, SAR=median, b=1"},
    {'block': 32, 'lr': 1, 'f': 0, 'gar': "median", 'legend': "SF, GAR=median, b=32"},
    {'block': 128, 'lr': 1, 'f': 0, 'gar': "median", 'legend': "SF, GAR=median, b=128"},
  	{'block': 256, 'lr': 1, 'f': 0, 'gar': "median", 'legend': "SF, GAR=median, b=256"},
]
```

**Results**

- Train Accuracy (weak, average and powerful)

<img src="https://tva1.sinaimg.cn/large/008i3skNgy1gq6mdb6p4oj31400u0tbl.jpg" alt="Figure_IMG_16" width="33%" /><img src="https://tva1.sinaimg.cn/large/008i3skNgy1gr6r1l1t2ij31400u0gog.jpg" alt="Figure_IMG_128" width="33%" /><img src="https://tva1.sinaimg.cn/large/008i3skNgy1gq6521nkbij31400u0795.jpg" alt="Figure_IMG_184" width="33%" />

- Time complexity

| 25 workers                     | SGD           | SF, b = 1           | SF, b = 32           | SF, b = 128         | SF, b = 256     |
| ------------------------------ | ------------- | ------------------- | -------------------- | ------------------- | --------------- |
| Weak — Time complexity         | 0.0023 (100%) | 0.0010 **(43.25%)** | 0.0011 **(46.88%)**  | 0.0012 **(52.97%)** | 0.0014 (61.3%)  |
| Time to reach **85%** accuracy | 0.2106 (100%) | 2.0614 (978.67%)    | 0.1087 **(51.63%)**  | 0.0562 **(26.72%)** | 0.0521 (24.75%) |
| Average — Time complexity      | 0.0103 (100%) | 0.0012 **(11.96%)** | 0.0018 **( 17.57%)** | 0.0032 **(31.39%)** | 0.0047 (45.68%) |
| Time to reach **85%** accuracy | 0.1725 (100%) | 1.4198 (823.12%)    | 0.1092 **(63.33%)**  | 0.0676 **(39.2%)**  | 0.0593 (34.43%) |
| Powefull — Time complexity     | 0.0313 (100%) | 0.0015 (**4.81%)**  | 0.0036 **(11.58%)**  | 0.0072 **(23.06%)** | 0.0124 (39.73%) |
| Time to reach **85%** accuracy | 0.2353 (100%) | —                   | 0.0928 **(39.43%)**  | 0.0749 **(31.82%)** | 0.0581 (24.69%) |

##### Observations

- All variants of SF converge and most of them converge even faster than SGD.
- The gain in time complexity when using SF is greater with large mini-batch size, this allows even weak smartphones to train on large batches of data.
- The combination SF($b=32, s=16$) trains the model with the least resources and it's even faster than SGD.

---

#### SVM: $b \in \{ 1,8, 32 \}$ and $s \in \{ 16, 128 ,S\}$

**Configuration**

```python
# details
dataset    : phishing (68 features)
block size : 1, 8, 32
batch size : 16, 128, S
rounds     : 150
# console
python main.py --model=SVM --dataset=phishing --batch_size=16 --workers=25 --frac=0.8 --rounds=150
python main.py ... --batch_size=128
python main.py ... --batch_size=0
# main.py
config = [
    {'block': 0, 'lr': 0.001, 'f': 0, 'gar': "median", 'legend': "SGD, GAR=median"},
    {'block': 1, 'lr': 0.1, 'f': 0, 'gar': "median", 'legend': "SF, SAR=median, b=1"},
    {'block': 8, 'lr': 0.01, 'f': 0, 'gar': "median", 'legend': "SF, GAR=median, b=8"},
    {'block': 32, 'lr': 0.01, 'f': 0, 'gar': "median", 'legend': "SF, GAR=median, b=32"},
]
```

**Results**

- Train Accuracy

<img src="https://tva1.sinaimg.cn/large/008i3skNgy1gq6mkve2cgj31400u0jtz.jpg" width="33%" /><img src="https://tva1.sinaimg.cn/large/008i3skNgy1gq6bhfq33ej31400u0aeq.jpg" alt="Figure_IMG_295" width="33%" /><img src="https://tva1.sinaimg.cn/large/008i3skNgy1gr6r6j409vj31400u0go7.jpg" width="33%" />

- Time complexity

| 25 workers                     | SGD           | HSF, b = 1       | HSF, b = 8      | HSF, b = 32         |
| ------------------------------ | ------------- | ---------------- | --------------- | ------------------- |
| Weak — Time complexity         | 0.0158 (100%) | 0.0145 (91.99%)  | 0.0145 (91.88%) | 0.0128 **(81.49%)** |
| Time to reach **85%** accuracy | 0.1605 (100%) | 0.3244 (202.09%) | 0.1369 (85.33%) | 0.0324 **(20.19%)** |
| Average — Time complexity      | 0.1436 (100%) | 0.1381 (96.16%)  | 0.1323 (92.13%) | 0.1135 (79.06%)     |
| Time to reach **85%** accuracy | 0.8092 (100%) | 1.1747 (145.16%) | 0.5271 (65.13%) | 0.1225 (15.14%)     |
| Powefull — Time complexity     | 0.3055 (100%) | 0.2840 (92.98%)  | 0.2797 (91.58%) | 0.2450 (80.2%)      |
| Time to reach **85%** accuracy | 1.5774 (100%) | 2.8118 (178.25%) | 1.3655 (86.57%) | 0.3119 (19.78%)     |

##### Observations

- All variantes of HSF converge and most of them converge even faster than SGD.
- The combination HSF($b=32, s=16$) trains the model with the least resources and it is even faster than SGD.

---

#### DNN: $s \in \{ 16, 128 ,S\}$

In a deep neural network, a block refer to a random sub-network that is trained by a worker during each training round. For example the the block $b = [98, 16, 10]$ refers to a sub-network with 98 input features, 16 hidden neurons in the first hidden layer and 10 classes in the output layer. At each round the server randomly selects a sub-network and send it to the workers to train on that sub-network only.

<img src="https://tva1.sinaimg.cn/large/008i3skNgy1gr6r88db5ej30fl0flgm1.jpg" alt="subNN" width="  25%" />

**Configuration**

```python
# details
dataset    : mnist (784 features)
block size : [392, 20, 10], [196, 18, 10], [98, 16, 10], [49, 14, 10], [20, 12, 10]
batch size : 16, 128, S
rounds     : 5000
# console
python main.py --model=DNN --dataset=mnist --batch_size=16 --workers=25 --frac=0.8 --rounds=5000
python main.py ... --batch_size=128
python main.py ... --batch_size=0
# main.py
config = [
    {'block': [], 'lr': 3, 'f': 0, 'gar': "median", 'legend': "SGD, GAR=median"},
    {'block': [392, 20, 10], 'lr': 6, 'f': 0, 'gar': "median", 'legend': "SF-392, GAR=median"},
    {'block': [196, 18, 10], 'lr': 9, 'f': 0, 'gar': "median", 'legend': "SF-196, GAR=median"},
    {'block': [98, 16, 10], 'lr': 12, 'f': 0, 'gar': "median", 'legend': "SF-98, GAR=median"},
    {'block': [49, 14, 10], 'lr': 15, 'f': 0, 'gar': "median", 'legend': "SF-49, GAR=median"},
    {'block': [20, 12, 10], 'lr': 18, 'f': 0, 'gar': "median", 'legend': "SF-20, GAR=median"},
]
# utils.py
EVAL_ROUND = 100
```

**Results**

- Train Accuracy

<img src="https://tva1.sinaimg.cn/large/008i3skNgy1gr6r8ahc9rj31400u0q6v.jpg" alt="Figure_IMG_190" width="33%" /><img src="https://tva1.sinaimg.cn/large/008i3skNgy1gq7jfzvojrj31400u0tfu.jpg" alt="Figure_IMG_212" width="33%" /><img src="https://tva1.sinaimg.cn/large/008i3skNgy1gq7jgjgw6mj31400u0qa1.jpg" alt="Figure_IMG_304" width="33%" />

- Time complexity

| 25 workers                     | SGD,Net [748, 30, 10] | SF, subNet [392, 20, 10] | SF, subNet [196, 18, 10] | SF, subNet [98, 16, 10] | SF, subNet [49, 14, 10] | SF, subNet [20, 12, 10] |
| ------------------------------ | --------------------- | ------------------------ | ------------------------ | ----------------------- | ----------------------- | ----------------------- |
| Weak — Time complexity         | 1.0707 (100%)         | 0.8651 (80.8%)           | 0.7814 (72.98%)          | 0.7230 (67.52%)         | 0.7109 (66.4%)          | 0.6861(64.08%)          |
| Time to reach **85%** accuracy | —                     | —                        | —                        | —                       | —                       | —                       |
| Average — Time complexity      | 1.6405 (100%)         | 1.2429 (75.76%)          | 1.0258 **(62.53%)**      | 0.9294 (56.65%)         | 0.8843 (53.91%)         | 0.8061 (49.14%)         |
| Time to reach **85%** accuracy | 164.71 (100%)         | 235.44 (142.94%)         | 224.19 **(136.11%)**     | —                       | —                       | —                       |
| Powefull — Time complexity     | 19.331 (100%)         | 19.835 (102.61%)         | 12.146 (62.84%)          | 9.9679 (51.57%)         | 7.4512 (38.55%)         | 6.3721 (32.96%)         |
| Time to reach **85%** accuracy | 314.93 (100%)         | 612.81 (194.58%)         | —                        | 723.77 (229.81%)        | —                       | —                       |

##### Observations

- SmartFed enables even weak smartphones to participate in training complexe models.
- Training on sub-networks with reasonable small batch-sizes (e.g., 128) can result in models with good performance.
- SmartFed presents a tradeoff between accuracy and computation for weak and average smartphones.

---



### Hybrid SmartFed (HSF)

In this set of experiments, we assume that smartphones participating in the training have different computation capabilities. Thus, require different configurations.

For simplicity, we assume that the training network contains three types of smartphones, weak, average and powerful.

We want to measure how well they can do compared to SGD with different proportions of these types of smartphones.

The experiments evaluate the training of the following computation profiles :

- **C~1~**: 70% weak, 20% average and 10% powerful.
- **C~2~**: 30% weak, 60% average and 10% powerful.
- **C~3~**: 10% weak, 40% average and 50% powerful.

We compare these profiles with both extreme cases; (1) **C~weak~**: all devices are weak, and (2) **C~powerful~**: all devices are powerful.

The aim of the experiments is to show that in a Hybrid network even with very few powerful devices, HSF can achieve good results.

#### Logistic regression: C~1~, C~2~, C~3~ vs. C~weak~,C~powerful~

**Configuration**

```python
# details
dataset    : mnist (784 features)
block size : 32
rounds     : 150
# console
python main.py --model=LR --dataset=mnist --workers=25 --frac=0.8 --rounds=150
# main.py
# dbs defines the computation profile: [%weak, %average, %powerful]
config = [
    {'block': 32, 'lr': 0.01, 'f': 0, 'gar': "median", 'legend': "HSF, b=32, $c_{weak}$", "dbs": "1,0,0"},
    {'block': 32, 'lr': 0.01, 'f': 0, 'gar': "median", 'legend': "HSF, b=32, $c_1$", "dbs": "0.7,0.2,0.1"},
    {'block': 32, 'lr': 0.01, 'f': 0, 'gar': "median", 'legend': "HSF, b=32, $c_2$", "dbs": "0.3,0.6,0.1"},
    {'block': 32, 'lr': 0.01, 'f': 0, 'gar': "median", 'legend': "HSF, b=32, $c_3$", "dbs": "0.1,0.4,0.5"},
    {'block': 32, 'lr': 0.01, 'f': 0, 'gar': "median", 'legend': "HSF, b=32, $c_{powerful}$", "dbs": "0,0,1"},
]
# utils.py
WEAK_DEVICE = 1 # weak devices use: batch_size = 1
AVERAGE_DEVICE = 64
POWERFUL_DEVICE = 508
```

**Results**

- Train Accuracy

<img src="https://tva1.sinaimg.cn/large/008i3skNgy1gr6reb3kwij31400u0jud.jpg" alt="Figure_IMG_773" width="50%" />

- Time complexity

| 25 workers                     | HSF, b = 32, C~weak~ | HSF, b = 32, C~1~ | HSF, b = 32, C~2~ | HSF, b = 32, C~3~ | HSF, b = 32, C~powerful~ |
| ------------------------------ | -------------------- | ----------------- | ----------------- | ----------------- | ------------------------ |
| Weak — Time complexity         | 0.0013 (100%)        | 0.0014            | 0.0014            | 0.0016            | 0.0017                   |
| Time to reach **85%** accuracy | —                    | 4.1736            | 4.1591            | 3.9351            | 3.9935                   |

##### Observations

- All configurations  (**C~1~, C~2~, C~3~**) achieve good accuracy similar to the extreme case **C~powerful~**.
- Introducing only 10% of powerful devices in **C~1~** increased accuracy from 84% to 96%.

---

#### SVM: C~1~, C~2~, C~3~ vs. C~weak~,C~powerful~

**Configuration**

```python
# details
dataset    : phishing (68 features)
block size : 8
rounds     : 150
# console
python main.py --model=SVM --dataset=phishing --workers=25 --frac=0.8 --rounds=150
# main.py
# dbs defines the capacity profile: [%weak, %average, %powerful]
config = [
    {'block': 8, 'lr': 0.01, 'f': 0, 'gar': "median", 'legend': "HSF, b=8, $c_{weak}$", "dbs": "1,0,0"},
    {'block': 8, 'lr': 0.01, 'f': 0, 'gar': "median", 'legend': "HSF, b=8, $c_1$", "dbs": "0.7,0.2,0.1"},
    {'block': 8, 'lr': 0.01, 'f': 0, 'gar': "median", 'legend': "HSF, b=8, $c_2$", "dbs": "0.3,0.6,0.1"},
    {'block': 8, 'lr': 0.01, 'f': 0, 'gar': "median", 'legend': "HSF, b=8, $c_3$", "dbs": "0.1,0.4,0.5"},
    {'block': 8, 'lr': 0.01, 'f': 0, 'gar': "median", 'legend': "HSF, b=8, $c_{powerful}$", "dbs": "0,0,1"},
]
# utils.py
WEAK_DEVICE = 1
AVERAGE_DEVICE = 64
POWERFUL_DEVICE = 353
```

**Results**

- Train Accuracy

<img src="https://tva1.sinaimg.cn/large/008i3skNgy1gr6rf0g6jpj31400u00x0.jpg" alt="Figure_IMG_378" width="50%" />

- Time complexity

| 25 workers                     | HSF, b = 8, C~weak~ | HSF, b = 8, C~1~ | HSF, b = 8, C~2~ | HSF, b = 8, C~3~ | HSF, b = 8, C~powerful~ |
| ------------------------------ | ------------------- | ---------------- | ---------------- | ---------------- | ----------------------- |
| Weak — Time complexity         | 0.0031              | 0.0418           | 0.0588           | 0.14961          | 0.2693                  |
| Time to reach **85%** accuracy | —                   | 0.6255           | 0.4731           | 0.8183           | 1.2347                  |

##### Observations

- Configurations (**C~2~, C~3~**) perform almost like the extreme case **C~powerful~**.
- **C~1~** achieves less accuracy compared to  (**C~2~, C~3~**), yet its way better than **C~weak~**.
- **C~2~** converges in 0.47s with iteration cost 0.05s similar to **C~powerful~** that takes 1.23s with iteration cost of 0.26s.

---

#### DNN: C~1~, C~2~, C~3~ vs. C~weak~,C~powerful~

**Configuration**

```python
# details
dataset    : mnist (784 features)
block size : [98, 16, 10]
rounds     : 5000
# console
python main.py --model=DNN --dataset=mnist --workers=25 --frac=0.8 --rounds=5000
# main.py
# dbs defines the capacity profile: [%weak, %average, %powerful]
config = [
    {'block': [98, 16, 10], 'lr': 12, 'f': 0, 'gar': "median", 'legend': "HSF-98, $c_{weak}$", "dbs": "1,0,0"},
    {'block': [98, 16, 10], 'lr': 12, 'f': 0, 'gar': "median", 'legend': "HSF-98, $c_1$", "dbs": "0.7,0.2,0.1"},
    {'block': [98, 16, 10], 'lr': 12, 'f': 0, 'gar': "median", 'legend': "HSF-98, $c_2$", "dbs": "0.3,0.6,0.1"},
    {'block': [98, 16, 10], 'lr': 12, 'f': 0, 'gar': "median", 'legend': "HSF-98, $c_3$", "dbs": "0.1,0.4,0.5"},
    {'block': [98, 16, 10], 'lr': 12, 'f': 0, 'gar': "median", 'legend': "HSF-98, $c_{powerful}$", "dbs": "0,0,1"},
]
# utils.py
EVAL_ROUND = 100
WEAK_DEVICE = 16
AVERAGE_DEVICE = 256
POWERFUL_DEVICE = 2400
```

**Results**

- Train Accuracy

<img src="https://tva1.sinaimg.cn/large/008i3skNgy1gr6rhhafttj31400u0whq.jpg" alt="Figure_IMG_211" width="50%" />

- Time complexity

| 25 workers                     | HSF-98, C~weak~ | HSF-98, C~1~ | HSF-98 = 8, C~2~ | HSF-98 = 8, C~3~ | HSF-98, C~powerful~ |
| ------------------------------ | --------------- | ------------ | ---------------- | ---------------- | ------------------- |
| Weak — Time complexity         | 0.5115          | 1.4810       | 1.9344           | 5.0801           | 8.4752              |
| Time to reach **85%** accuracy | — (69.35%)      | — (83.29%)   | **463.01**       | **486.61**       | **620.10**          |

##### Observations

- A direct observation is that all configurations (**C~1~, C~2~, C~3~**)  perform almost like the extreme case **C~powerful~**.
- HSF achieves good results with significantly less iteration cost (1,48s vs. 8,47s).
- Even in a network containing mostly weak devices, using HSF we achieve comparable results to a powerful network.
- Introducing only 10% of powerful devices in **C~1~** increased accuracy from 69.35% to 83.29%.

----



## SmartFed in a Byzantine environment

Here, we assume a Byzantine setting, where Byzantine workers can send artificially construct gradients to compromise the learning task. We consider two famous attacks, `Fall of Empires (FOE)` and `Little Is Enough (LIE)`, where 25% of active workers are Byzantine.

We asses the performance of SmartFed under multiple configurations using different robust aggregation rules; namely *median*, *krum* and *aksel* with *average* as reference.

We run the following experiments for each model:

- **SmartFed (SF)**: SmartFed with a selected block size from previous experiments. Each round of training is done using full device’s data ($s=S$).
- **Light SmartFed (LSF)**: Light SmartFed with block size of one ($b=1$) and one training sample ($s=1$).
- **Hybrid SmartFed (HSF)**: Hybrid SmartFed support training between devices of different computation profiles (**C~1~, C~2~, C~3~** for simplicity).

All experiments were done using 100 workers 25 among them are Byzantine.



#### Logistic regression: Evaluation of SmartFed under `FOE ` and `LIE `attacks.



##### Byzantine resilience of SmartFed (SF)

**Configuration**

```python
# details
dataset    : mnist (784 features)
block size : 32
rounds     : 150
# console
python main.py --model=LR --dataset=mnist --batch_size=127 --workers=100 --rounds=300 --attack=FOE # Fall of Empires attack
python main.py ... --attack=LIE # Little Is Enough attack
# main.py
config = [
    {'block': 32, 'lr': 0.01, 'f': 0, 'gar': "average", 'legend': "SF, NO ATTACK, b=32"},
    {'block': 32, 'lr': 0.01, 'f': 25, 'gar': "average", 'legend': "SF, GAR=avg, b=32"},
    {'block': 32, 'lr': 0.01, 'f': 25, 'gar': "median", 'legend': "SF, GAR=median, b=32"},
    {'block': 32, 'lr': 0.01, 'f': 25, 'gar': "krum", 'legend': "SF, GAR=krum, b=32"},
    {'block': 32, 'lr': 0.01, 'f': 25, 'gar': "aksel", 'legend': "SF, GAR=aksel, b=32"},
]
# utils.py
EVAL_ROUND = 10
BYZ_ITER = 20
FOE_EPS = 10
LIE_Z = 10
```

**Results**

- Train Accuracy (FOE, LIE)

<img src="https://tva1.sinaimg.cn/large/008i3skNgy1gr6ro3pvuyj31400u00vn.jpg" alt="Figure_IMG_143" width="50%" /><img src="https://tva1.sinaimg.cn/large/008i3skNgy1gqdmnq47pjj31400u0dmo.jpg" alt="Figure_IMG_260" width="50%" />

- Time complexity

| 100 workers                          | SF, NO ATTACK | SF, GAR=avg | SF, GAR=meidan | SF, GAR=krum | SF, GAR=aksel |
| ------------------------------------ | ------------- | ----------- | -------------- | ------------ | ------------- |
| FOE — Time complexity                | 0.0042        | 0.0041      | 0.0044         | 0.0044       | 0.0042        |
| FOE — Time to reach **85%** accuracy | 1.6572        | —           | 1.6049         | 5.5368       | **1.3535**    |
| LIE — Time complexity                | 0.0041        | 0.0041      | 0.0041         | 0.0042       | 0.0041        |
| LIE — Time to reach **85%** accuracy | 1.4146        | —           | 1.2276         | —            | **1.1601**    |

##### Observations

- **SF** defends agains both Byzantine attacks when combined with robust gradients aggregation rules (GARs).
- **SF** converges even faster than the case with no attacks when using the Aksel GAR.

---

##### Byzantine resilience of Light SmartFed (LSF)

**Configuration**

```python
# details
dataset    : mnist (784 features)
block size : 32
rounds     : 150
# console
python main.py --model=LR --dataset=mnist --batch_size=1 --workers=100 --rounds=500 --attack=FOE # Fall of Empires attack
python main.py ... --attack=LIE # Little Is Enough attack
# main.py
config = [
    {'block': 1, 'lr': 0.1, 'f': 0, 'gar': "average", 'legend': "LSF, NO ATTACK"},
    {'block': 1, 'lr': 0.1, 'f': 25, 'gar': "average", 'legend': "LSF, SAR=avg"},
    {'block': 1, 'lr': 0.1, 'f': 25, 'gar': "median", 'legend': "LSF, SAR=median"},
    {'block': 1, 'lr': 0.1, 'f': 25, 'gar': "krum", 'legend': "LSF, SAR=krum"},
    {'block': 1, 'lr': 0.1, 'f': 25, 'gar': "aksel", 'legend': "LSF, SAR=aksel"},
]
# utils.py
EVAL_ROUND = 10
BYZ_ITER = 50
FOE_EPS = 10
LIE_Z = 10 
```

**Results**

- Train Accuracy (FOE, LIE)

<img src="https://tva1.sinaimg.cn/large/008i3skNgy1gr6rq5t37oj31400u0dj6.jpg" alt="Figure_IMG_480" width="50%" /><img src="https://tva1.sinaimg.cn/large/008i3skNgy1gqdn7svuooj31400u0n22.jpg" alt="Figure_IMG_391" width="50%" />

- Time complexity

| 100 workers                          | LSF, NO ATTACK | LSF, SAR=avg | LSF, SAR=meidan | LSF, SAR=krum | LSF, SAR=aksel |
| ------------------------------------ | -------------- | ------------ | --------------- | ------------- | -------------- |
| FOE — Time complexity                | 0.0051         | 0.0050       | 0.0051          | 0.0053        | 0.0052         |
| FOE — Time to reach **85%** accuracy | 7.0094         | —            | —               | —             | —              |
| LIE — Time complexity                | 0.0042         | 0.0042       | 0.0042          | 0.0042        | 0.0042         |
| LIE — Time to reach **85%** accuracy | 6.0630         | —            | —               | —             | —              |

##### Observations

- **LSF** cannot defend against any of the Byzantine attacks.
- **LSF**  demonstrate the tradeoff between block size, batch size and the Byzantine effect.

---

##### Byzantine resilience of Hybrid SmartFed (HSF)

**Configuration**

```python
# details
dataset    : mnist (784 features)
block size : 32
rounds     : 150
# console
python main.py --model=LR --dataset=mnist --batch_size=127 --workers=100 --rounds=500 --attack=FOE # Fall of Empires attack
python main.py ... --attack=LIE # Little Is Enough attack
# main.py
config = [
    {'block': 32, 'lr': 0.01, 'f': 0, 'gar': "average", 'legend': "SF, NO ATTACK"},
    {'block': 32, 'lr': 0.01, 'f': 25, 'gar': "average", 'legend': "SF, GAR=avg"},
    {'block': 32, 'lr': 0.01, 'f': 25, 'gar': "aksel", 'legend': "HSF, GAR=aksel, $c_{weak}$", "dbs": "1,0,0"},
    {'block': 32, 'lr': 0.01, 'f': 25, 'gar': "aksel", 'legend': "HSF, GAR=aksel, $c_1$", "dbs": "0.7,0.2,0.1"},
    {'block': 32, 'lr': 0.01, 'f': 25, 'gar': "aksel", 'legend': "HSF, GAR=aksel, $c_2$", "dbs": "0.3,0.6,0.1"},
    {'block': 32, 'lr': 0.01, 'f': 25, 'gar': "aksel", 'legend': "HSF, GAR=aksel, $c_3$", "dbs": "0.1,0.4,0.5"},
    {'block': 32, 'lr': 0.01, 'f': 25, 'gar': "aksel", 'legend': "HSF, GAR=aksel, $c_{powerful}$", "dbs": "0,0,1"},
]
# utils.py
EVAL_ROUND = 10
BYZ_ITER = 20
FOE_EPS = 10
LIE_Z = 10
WEAK_DEVICE = 1
AVERAGE_DEVICE = 32
POWERFUL_DEVICE = 127
```

**Results**

- Train Accuracy (FOE, LIE)

<img src="https://tva1.sinaimg.cn/large/008i3skNgy1gqdnlrc3yqj31400u0agp.jpg" alt="Figure_IMG_344" width="50%" /><img src="https://tva1.sinaimg.cn/large/008i3skNgy1gr6rp5s5fgj31400u042c.jpg" alt="Figure_IMG_219" width="50%" />

- Time complexity

| 100 workers                          | SF, NO ATTACK   | SF, GAR=avg      | HSF, GAR=aksel, C~weak~ | HSF, GAR=aksel, $c_1$ | HSF, GAR=aksel, $c_2$ | HSF, GAR=aksel, $c_3$ | HSF, GAR=aksel, $c_{powerful}$ |
| ------------------------------------ | --------------- | ---------------- | ----------------------- | --------------------- | --------------------- | --------------------- | ------------------------------ |
| FOE — Time complexity                | 0.0065 (100%)   | 0.0065 (100%)    | 0.0045 (68.77%)         | 0.0051 (78.48%)       | 0.0054 (83.7%)        | 0.0060 (92.29%)       | 0.0065 (100%)                  |
| FOE — Time to reach **85%** accuracy | 2.1436 (100%)   | —                | —                       | —                     | 2.0515 (95.7%)        | 1.4860 (69.32%)       | 1.6280  (75.95%)               |
| LIE — Time complexity                | 0.0066 (100%)   | 0.0072 (107.98%) | 0.0047 (71.4%)          | 0.0051 (77.33%)       | 0.0054 (81.75%)       | 0.0061 (92.16%)       | 0.0069 (103.38%)               |
| LIE — Time to reach **85%** accuracy | 2.1173 (100.0%) | —                | —                       | —                     | 2.4476 (115.6%)       | 1.4322 (67.64%)       | 1.7261 (81.53%)                |

##### Observations

- **HSF** experiments clearly demonstrate the tradeoff between block size, batch size and the Byzantine effect.
- **HSF** with configuration **C~weak~** is not resilient to the byzantine attacks.
- Introducing few powerful devices as in **C~1~** drastically reduces the effect of the Byzantine attacks.
- **HSF** with configurations **C~2~** and **C~3~** defend agains both Byzantine attacks.

---

#### SVM: Evaluation of SmartFed under `FOE ` and `LIE `attacks.

##### Byzantine resilience of SmartFed (SF)

**Configuration**

```python
# details
dataset    : phishing (68 features)
block size : 8
rounds     : 150
# console
python main.py --model=SVM --dataset=phishing --batch_size=88 --workers=100 --rounds=150 --attack=FOE # Fall of Empires attack
python main.py ... --attack=LIE # Little Is Enough attack
# main.py
config = [
    {'block': 8, 'lr': 0.01, 'f': 0, 'gar': "average", 'legend': "SF, NO ATTACK, b=8"},
    {'block': 8, 'lr': 0.01, 'f': 25, 'gar': "average", 'legend': "SF, GAR=avg, b=8"},
    {'block': 8, 'lr': 0.01, 'f': 25, 'gar': "median", 'legend': "SF, GAR=median, b=8"},
    {'block': 8, 'lr': 0.01, 'f': 25, 'gar': "krum", 'legend': "SF, GAR=krum, b=8"},
    {'block': 8, 'lr': 0.01, 'f': 25, 'gar': "aksel", 'legend': "SF, GAR=aksel, b=8"},
]
# utils.py
EVAL_ROUND = 10
BYZ_ITER = 20
FOE_EPS = 10
LIE_Z = 10
```

**Results**

- Train Accuracy (FOE, LIE)

<img src="https://tva1.sinaimg.cn/large/008i3skNgy1gr6rs2h3jqj31400u0dim.jpg" alt="Figure_IMG_512" width="50%" /><img src="https://tva1.sinaimg.cn/large/008i3skNgy1gqdrn1t303j31400u0afp.jpg" alt="Figure_IMG_152" width="50%" />

- Time complexity

| 100 workers                          | SF, NO ATTACK                | SF, GAR=avg     | SF, GAR=meidan   | SF, GAR=krum     | SF, GAR=aksel   |
| ------------------------------------ | ---------------------------- | --------------- | ---------------- | ---------------- | --------------- |
| FOE — Time complexity                | 0.0663 (100%)                | 0.069 (104.84%) | 0.0681 (102.76%) | 0.0664 (100.24%) | 0.0662 (99.85%) |
| FOE — Time to reach **85%** accuracy | 1.3976 (100%)                | —               | 1.0425 (74.59%)  | 2.1292 (152.35%) | 1.1190 (80.07%) |
| LIE — Time complexity                | 0.0689 (100.0%)              | 0.0725 (105.2%) | 0.0688 (99.84%)  | 0.0701 (101.7%)  | 0.0682 (99.02%) |
| LIE — Time to reach **85%** accuracy | 1.6507 (100%)0.8125 (49.33%) | 0.8125 (49.33%) | 0.9162 (55.5%)   | 1.2613 (76.41%)  | 1.1292 (68.41%) |

##### Observations

- **SF** defends agains both Byzantine attacks when combined with robust gradients aggregation rules (GARs).

---

#### Byzantine resilience of Light SmartFed (LSF)

**Configuration**

```python
# details
dataset    : phishing (68 features)
block size : 8
rounds     : 150
# console
python main.py --model=SVM --dataset=phishing --batch_size=1 --workers=100 --rounds=150 --attack=FOE # Fall of Empires attack
python main.py ... --attack=LIE # Little Is Enough attack
# main.py
config = [
    {'block': 1, 'lr': 0.1, 'f': 0, 'gar': "average", 'legend': "LSF, NO ATTACK"},
    {'block': 1, 'lr': 0.1, 'f': 25, 'gar': "average", 'legend': "LSF, GAR=avg"},
    {'block': 1, 'lr': 0.1, 'f': 25, 'gar': "median", 'legend': "LSF, GAR=median"},
    {'block': 1, 'lr': 0.1, 'f': 25, 'gar': "krum", 'legend': "LSF, GAR=krum"},
    {'block': 1, 'lr': 0.1, 'f': 25, 'gar': "aksel", 'legend': "LSF, GAR=aksel"},
]
# utils.py
EVAL_ROUND = 10
BYZ_ITER = 20
FOE_EPS = 10
LIE_Z = 10
```

**Results**

- Train Accuracy (FOE, LIE)

<img src="https://tva1.sinaimg.cn/large/008i3skNgy1gqdryijhq4j31400u0grh.jpg" alt="Figure_IMG_620" width="50%" /><img src="https://tva1.sinaimg.cn/large/008i3skNgy1gr6rs3svd3j31400u00vj.jpg" alt="Figure_IMG_208" width="50%" />

- Time complexity

| 100 workers                          | LSF, NO ATTACK  | LSF, SAR=avg    | LSF, SAR=meidan  | LSF, SAR=krum    | LSF, SAR=aksel  |
| ------------------------------------ | --------------- | --------------- | ---------------- | ---------------- | --------------- |
| FOE — Time complexity                | 0.0030 (100%)   | 0.0030 (98.69%) | 0.0030 (99.17%)  | 0.0030 (99.59%)  | 0.0030 (99.08%) |
| FOE — Time to reach **85%** accuracy | —               | —               | —                | —                | —               |
| LIE — Time complexity                | 0.0029 (100.0%) | 0.0029 (101.9%) | 0.0030 (103.25%) | 0.0029 (101.65%) | 0.00 (101.7%)   |
| LIE — Time to reach **85%** accuracy | —               | —               | —                | —                | —               |

##### Observations

- **LSF** cannot defend against any of the Byzantine attacks.
- **LSF** with Aksel performs better than the other GARs and achieves more than 60% accuracy despite the attacks.

---

#### Byzantine resilience of Hybrid SmartFed (HSF)

**Configuration**

```python
# details
dataset    : phishing (68 features)
block size : 8
rounds     : 150
# console
python main.py --model=SVM --dataset=phishing --batch_size=88 --workers=100 --rounds=150 --attack=FOE # Fall of Empires attack
python main.py ... --attack=LIE # Little Is Enough attack
# main.py
config = [
    {'block': 8, 'lr': 0.01, 'f': 0, 'gar': "average", 'legend': "SF, NO ATTACK"},
    {'block': 8, 'lr': 0.01, 'f': 25, 'gar': "average", 'legend': "SF, GAR=avg"},
    {'block': 8, 'lr': 0.01, 'f': 25, 'gar': "aksel", 'legend': "HSF, GAR=aksel, $c_{weak}$", "dbs": "1,0,0"},
    {'block': 8, 'lr': 0.01, 'f': 25, 'gar': "aksel", 'legend': "HSF, GAR=aksel, $c_1$", "dbs": "0.7,0.2,0.1"},
    {'block': 8, 'lr': 0.01, 'f': 25, 'gar': "aksel", 'legend': "HSF, GAR=aksel, $c_2$", "dbs": "0.3,0.6,0.1"},
    {'block': 8, 'lr': 0.01, 'f': 25, 'gar': "aksel", 'legend': "HSF, GAR=aksel, $c_3$", "dbs": "0.1,0.4,0.5"},
    {'block': 8, 'lr': 0.01, 'f': 25, 'gar': "aksel", 'legend': "HSF, GAR=aksel, $c_{powerful}$", "dbs": "0,0,1"},
]
# utils.py
EVAL_ROUND = 10
BYZ_ITER = 20
FOE_EPS = 10
LIE_Z = 10
WEAK_DEVICE = 1
AVERAGE_DEVICE = 16
POWERFUL_DEVICE = 88
```

**Results**

- Train Accuracy (FOE, LIE)

<img src="https://tva1.sinaimg.cn/large/008i3skNgy1gqdsjvh1wmj31400u0ahw.jpg" alt="Figure_IMG_992" width="50%" /><img src="https://tva1.sinaimg.cn/large/008i3skNgy1gr6rs5ighhj31400u00w0.jpg" alt="Figure_IMG_657" width="50%" />

- Time complexity

| 100 workers                          | SF, NO ATTACK   | SF, GAR=avg       | HSF, GAR=aksel, C~weak~ | HSF, GAR=aksel, C~1~ | HSF, GAR=aksel, C~2~ | HSF, GAR=aksel, C~3~ | HSF, GAR=aksel, C~powerful~ |
| ------------------------------------ | --------------- | ----------------- | ----------------------- | -------------------- | -------------------- | -------------------- | --------------------------- |
| FOE — Time complexity                | 0.1871 (100%)   | 0.21928 (117.16%) | 0.0096 (5.13%)          | 0.04080 (21.8%)      | 0.05034 (26.9%)      | 0.11746 (62.76%)     | 0.18646 (99.62%)            |
| FOE — Time to reach **85%** accuracy | 1.3666 (100%)   | —                 | —                       | —                    | 0.4955 (36.26%)      | 0.9041 (66.16%)      | 1.0835 (79.29%)             |
| LIE — Time complexity                | 0.1931 (100.0%) | 0.2187 (113.33%)  | 0.0117 (6.09%)          | 0.0459 (23.78%)      | 0.0505 (26.17%)      | 0.1216 (62.97%)      | 0.2061 (106.73%)            |
| LIE — Time to reach **85%** accuracy | 1.6951 (100.0%) | 0.8683 (51.23%)   | 0.8244 (48.63%)         | 0.6995 (41.27%)      | 0.4918 (29.02%)      | 0.8185 (48.29%)      | 1.1724 (69.17%)             |

##### Observations

- **HSF** with configuration **C~weak~** is not resilient to the byzantine attacks.
- Introducing few powerful devices as in **C~1~** reduces the effect of the Byzantine attacks.
- **HSF** with configurations **C~2~** and **C~3~** defend agains both Byzantine attacks.

---

#### DNN: Evaluation of SmartFed under `FOE ` and `LIE `attacks.

##### Byzantine resilience of SmartFed (SF)

**Configuration**

```python
# details
dataset    : mnist (784 features)
block size : [98, 16, 10]
rounds     : 5000
# console
python main.py --model=DNN --dataset=mnist --batch_size=600 --workers=100 --rounds=5000 --attack=FOE # Fall of Empires attack
python main.py ... --attack=LIE # Little Is Enough attack
# main.py
config = [
    {'block': [98, 16, 10], 'lr': 12, 'f': 0, 'gar': "average", 'legend': "SF-98, NO ATTACK"},
    {'block': [98, 16, 10], 'lr': 12, 'f': 25, 'gar': "average", 'legend': "SF-98, GAR=avg"},
    {'block': [98, 16, 10], 'lr': 12, 'f': 25, 'gar': "median", 'legend': "SF-98, GAR=median"},
    {'block': [98, 16, 10], 'lr': 12, 'f': 25, 'gar': "krum", 'legend': "SF-98, GAR=krum"},
    {'block': [98, 16, 10], 'lr': 12, 'f': 25, 'gar': "aksel", 'legend': "SF-98, GAR=aksel"},
]
# utils.py
EVAL_ROUND = 100
BYZ_ITER = 500
FOE_EPS = 10
LIE_Z = 10
```

**Results**

- Train Accuracy (FOE, LIE)

<img src="https://tva1.sinaimg.cn/large/008i3skNgy1gr6rtllahzj31400u0tbr.jpg" alt="Figure_IMG_615" width="50%" /><img src="https://tva1.sinaimg.cn/large/008i3skNgy1gqdt8gf7yyj31400u00z2.jpg" alt="Figure_IMG_335" width="50%" />

- Time complexity

| 100 workers                          | SF-98, NO ATTACK | SF-98, GAR=avg  | SF-98, GAR=meidan | SF-98, GAR=krum  | SF-98, GAR=aksel |
| ------------------------------------ | ---------------- | --------------- | ----------------- | ---------------- | ---------------- |
| FOE — Time complexity                | 2.5291 (100%)    | 2.5251 (99.84%) | 2.9438 (116.4%)   | 2.8896 (114.26%) | 2.7231 (107.67%) |
| FOE — Time to reach **85%** accuracy | —                | —               | —                 | —                | 374.57           |
| LIE — Time complexity                | 2.6690 (100%)    | 2.3567 (88.3%)  | 2.6956 (101%)     | 2.6807 (100.44%) | 2.7183 (101.85%) |
| LIE — Time to reach **85%** accuracy | —                | —               | 604.31            | 762.79           | 423.72           |

##### Observations

- **SF** defends agains both Byzantine attacks even when using a complex model such as a deep neural network.
- Aksel works really well with **SF** to defend agains attacks but also it converges faster.

---

##### Byzantine resilience of Light SmartFed (LSF)

**Configuration**

```python
# details
dataset    : mnist (784 features)
block size : [98, 16, 10]
rounds     : 5000
# console
python main.py --model=DNN --dataset=mnist --batch_size=1 --workers=100 --rounds=5000 --attack=FOE # Fall of Empires attack
python main.py ... --attack=LIE # Little Is Enough attack
# main.py
config = [
    {'block': [1, 1, 10], 'lr': 21, 'f': 0, 'gar': "average", 'legend': "LSF, NO ATTACK"},
    {'block': [1, 1, 10], 'lr': 21, 'f': 25, 'gar': "average", 'legend': "LSF, GAR=avg"},
    {'block': [1, 1, 10], 'lr': 21, 'f': 25, 'gar': "median", 'legend': "LSF, GAR=median"},
    {'block': [1, 1, 10], 'lr': 21, 'f': 25, 'gar': "krum", 'legend': "LSF, GAR=krum"},
    {'block': [1, 1, 10], 'lr': 21, 'f': 25, 'gar': "aksel", 'legend': "LSF, GAR=aksel"},
]
# utils.py
EVAL_ROUND = 100
BYZ_ITER = 500
FOE_EPS = 10
LIE_Z = 10
```

**Results**

- Train Accuracy (FOE, LIE)

<img src="https://tva1.sinaimg.cn/large/008i3skNgy1gr6rtoo7gnj31400u0q6q.jpg" alt="Figure_IMG_427" width="50%" /><img src="https://tva1.sinaimg.cn/large/008i3skNgy1gqdtfhprulj31400u0q8h.jpg" alt="Figure_IMG_937" width="50%" />

- Time complexity

| 100 workers                          | LSF, NO ATTACK | LSF, SAR=avg     | LSF, SAR=meidan | LSF, SAR=krum   | LSF, SAR=aksel   |
| ------------------------------------ | -------------- | ---------------- | --------------- | --------------- | ---------------- |
| FOE — Time complexity                | 0.3787 (100%)  | 0.3793 (100.16%) | 0.3328 (87.9%)  | 0.3325 (87.81%) | 0.3299 (87.13%)  |
| FOE — Time to reach **85%** accuracy | —              | —                | —               | —               | —                |
| LIE — Time complexity                | 0.4139 (100%)  | 0.4108 (99.26%)  | 0.4074 (98.44%) | 0.3989 (96.39%) | 0.4333 (104.69%) |
| LIE — Time to reach **85%** accuracy | —              | —                | —               | —               | —                |

##### Observations

- **LSF** cannot defend against any of the Byzantine attacks.
- In DNN, even when using Aksel, **LSF** could not do better than 20%/40% in terms of accuracy while the other GARs totally crashed.

---

##### Byzantine resilience of Hybrid SmartFed (HSF)

**Configuration**

```python
# details
dataset    : mnist (784 features)
block size : [98, 16, 10]
rounds     : 5000
# console
python main.py --model=DNN --dataset=mnist --batch_size=0 --workers=100 --rounds=5000 --attack=FOE # Fall of Empires attack
python main.py ... --attack=LIE # Little Is Enough attack
# main.py
config = [
    {'block': [98, 16, 10], 'lr': 12, 'f': 0, 'gar': "average", 'legend': "SF, NO ATTACK"},
    {'block': [98, 16, 10], 'lr': 12, 'f': 25, 'gar': "average", 'legend': "SF, GAR=avg"},
    {'block': [98, 16, 10], 'lr': 12, 'f': 25, 'gar': "aksel", 'legend': "HSF-98, GAR=aksel, $c_{weak}$", "dbs": "1,0,0"},
    {'block': [98, 16, 10], 'lr': 12, 'f': 25, 'gar': "aksel", 'legend': "HSF-98, GAR=aksel, $c_1$", "dbs": "0.7,0.2,0.1"},
    {'block': [98, 16, 10], 'lr': 12, 'f': 25, 'gar': "aksel", 'legend': "HSF-98, GAR=aksel, $c_2$", "dbs": "0.3,0.6,0.1"},
    {'block': [98, 16, 10], 'lr': 12, 'f': 25, 'gar': "aksel", 'legend': "HSF-98, GAR=aksel, $c_3$", "dbs": "0.1,0.4,0.5"},
    {'block': [98, 16, 10], 'lr': 12, 'f': 25, 'gar': "aksel", 'legend': "HSF-98, GAR=aksel, $c_{powerful}$", "dbs": "0,0,1"},
]
# utils.py
EVAL_ROUND = 10
BYZ_ITER = 20
FOE_EPS = 10
LIE_Z = 10
WEAK_DEVICE = 16
AVERAGE_DEVICE = 64
POWERFUL_DEVICE = 600
```

**Results**

- Train Accuracy (FOE, LIE)

<img src="https://tva1.sinaimg.cn/large/008i3skNgy1gr6rtnpogoj31400u0dj1.jpg" alt="Figure_IMG_602" width="50%" /><img src="https://tva1.sinaimg.cn/large/008i3skNgy1gqdwebgo1gj31400u0qax.jpg" alt="Figure_IMG_870" width="50%" />

- Time complexity

| 100 workers                          | SF, NO ATTACK | SF, GAR=avg     | HSF, GAR=aksel, $c_{weak}$ | HSF, GAR=aksel, $c_1$ | HSF, GAR=aksel, $c_2$ | HSF, GAR=aksel, $c_3$ | HSF, GAR=aksel, $c_{powerful}$ |
| ------------------------------------ | ------------- | --------------- | -------------------------- | --------------------- | --------------------- | --------------------- | ------------------------------ |
| FOE — Time complexity                | 2.5596 (100%) | 2.8692 (112.1%) | 0.5250 (20.51%)            | 0.782 (30.56%)        | 0.8312 (32.48%)       | 1.6575 (64.76%)       | 2.7175 (106.17%)               |
| FOE — Time to reach **85%** accuracy | —             | —               | —                          | —                     | —                     | —                     | 448.5113                       |
| LIE — Time complexity                | 3.0593 (100%) | 2.8210 (92.21%) | 0.5046 (16.5%)             | 0.7821 (25.56%)       | 0.8851 (28.93%)       | 1.6913 (55.29%)       | 2.6669 (87.17%)                |
| LIE — Time to reach **85%** accuracy | —             | —               | 238.64                     | 294.09                | 289.98                | 381.01                | 467.96                         |

##### Observations

- Interestingly **HSF** with configuration **C~weak~** and **C~1~** are resilient to both byzantine attacks when using a sub-network of size $b=[98, 16, 10]$.
- Using the block size $b=[98, 16, 10]$ resulted in even  **C~weak~** being resilient.
- **HSF** with all configurations **C~1~**, **C~2~** and **C~3~** defend agains both Byzantine attacks.

----

END.

