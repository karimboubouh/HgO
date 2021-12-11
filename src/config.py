SERVER_HOST = "0.0.0.0"
SERVER_PORT = 45000
ROUNDS = 150
TCP_SOCKET_BUFFER_SIZE = 500000
TCP_SOCKET_SERVER_LISTEN = 10
SOCK_TIMEOUT = 20
RECV_BUFFER = 100 * 4096

METRIC = "acc"
TRACK_ACC = "Test"
EVAL_ROUND = 5
BARS_ROUND = 1
PLOT_GRADS = False
ALLOW_DIFF_BATCH_SIZE = False
USE_DIFFERENT_HARDWARE = False
MIN_DATA_SAMPLE = 128

BYZ_ITER = 10
FOE_EPS = 10
LIE_Z = 10

ONE_DEVICE = 1
# ===================================== MNIST =================================
# LR/LN/SVM
# WEAK_DEVICE = [500, 250]
# AVERAGE_DEVICE = [1000, 500]
# POWERFUL_DEVICE = [5000, 1000]
# MLR
WEAK_DEVICE = [500, 250]
AVERAGE_DEVICE = [10000, 2500]
POWERFUL_DEVICE = [50000, 5000]
# DNN
# WEAK_DEVICE = [5000, 500]
# AVERAGE_DEVICE = [25000, 2500]
# POWERFUL_DEVICE = [50000, 5000]
# ===================================== CIFAR-10 ==============================
# cifar10 # 3072

# DNN
# WEAK_DEVICE = [20000, 2000]
# AVERAGE_DEVICE = [100000, 5000]
# POWERFUL_DEVICE = [200000, 10000]

# WEAK_DEVICE = [100, 10]
# AVERAGE_DEVICE = [500, 100]
# POWERFUL_DEVICE = [5000, 500]

# boston
# WEAK_DEVICE = 20
# AVERAGE_DEVICE = 100
# POWERFUL_DEVICE = 500


LAYER_DIMS_MNIST = [784, 30, 10]
LAYER_DIMS_CIFAR10 = [3072, 30, 10]
LAYER_DIMS_FEMNIST = [784, 2048, 62]
# LAYER_DIMS_FEMNIST = [784, 512, 256, 62]

LAPS = 1
LAPS_GRADS = 0.02
