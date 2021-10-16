import numpy as np
import matplotlib.pyplot as plt

from src.config import EVAL_ROUND
from src.utils import load


def plot_vs(x, y, info):
    xlabel = info.get('xlabel', "X")
    ylabel = info.get('ylabel', "Y")
    title = info.get('title', f"{xlabel} vs {ylabel}")
    plt.rc('legend', fontsize=14)
    plt.plot(x, color='blue')
    plt.plot(y, color='orange')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(loc='best', shadow=True)
    plt.show()


def plot_many(data, info):
    if isinstance(data, str):
        data = load(data)
    xlabel = info.get('xlabel', "X")
    ylabel = info.get('ylabel', "Y")
    title = info.get('title', f"{xlabel} vs {ylabel}")
    colors = ['blue', 'orange', 'red', 'black', 'pink', 'aqua', 'olive', 'tan']
    for i, (d, l) in enumerate(data):
        plt.plot(d, color=colors[i], label=l)
    # plt.rc('legend', fontsize=14)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc='best', shadow=True)
    plt.title(title)
    plt.show()


def plot_std(mean, std, info, unique=None):
    if isinstance(mean, str):
        redo = True
        mean, std = load(mean)
    else:
        redo = False
        # mean1, std1 = load(f"../out/EXP_DNN_0E90D3.p")
    # plt.style.use('seaborn')
    xlabel = info.get('xlabel', "Rounds")
    ylabel = info.get('ylabel', "Test Accuracy")
    title = info.get('title', f"{xlabel} vs {ylabel}")
    colors = ['red', 'orange', 'green', 'black', 'aqua', 'blue', 'tan', 'grey', 'navy', 'pink']
    # line_styles = ['-', '--', '-.', ':', '-', '--', '-.', ':', '-', '--', '-.', ':']
    line_styles = ['-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', ]
    # markers = ['.', 'o', 'v', '^', '<', '>', 's', 'p', '*', 'h', 'H', '+', 'x', 'D', 'd', '|', '_', 'P', 'X']

    # mean[4] = mean1[0]
    # std[4] = std1[0]

    for i in range(len(mean)):
        x = range(0, len(mean[i][0]) * EVAL_ROUND, EVAL_ROUND)
        plt.plot(x, mean[i][0], color=colors[i], label=mean[i][1], linestyle=line_styles[i])
        # plt.plot(mean[i][0], color=colors[i], label=mean[i][1], linestyle=line_styles[i], marker=markers[i])
        plt.fill_between(x, mean[i][0] - std[i], mean[i][0] + std[i], color=colors[i], alpha=.1)
    plt.rc('legend', fontsize=12)
    plt.xlabel(xlabel, fontsize=13)
    plt.xticks(fontsize=13, )
    plt.ylabel(ylabel, fontsize=13)
    plt.yticks(fontsize=13, )
    loc = 'lower right'  # lower right
    plt.legend(loc=loc, shadow=True)
    plt.grid(linestyle='dashed')
    # plt.title(title)
    if not unique:
        unique = np.random.randint(100, 999)
    print(f"Saving files with code {unique}...")
    if redo:
        plt.savefig(f"../out/EXP_{unique}.pdf")
    else:
        plt.savefig(f"./out/EXP_{unique}.pdf")
    # plt.savefig(f"./out/EXP_{unique}.png", dpi=300)
    plt.show()


def plot_app(data, info={}):
    acc = load(data)
    print(acc)
    xlabel = info.get('xlabel', "Rounds")
    ylabel = info.get('ylabel', "Train Accuracy")

    plt.rc('legend', fontsize=14)
    plt.plot(acc, color='blue')
    # plt.plot(y, color='orange')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc='best', shadow=True)

    code = np.random.randint(100, 999)
    print(f"Saving files {code}...")
    plt.savefig(f"../out/Figure_PDF_{code}.pdf")
    # plt.savefig(f"../out/Figure_IMG_{code}.png", dpi=300)
    plt.show()


def grads_number():
    plt.style.use('ggplot')
    x = ["SGD\n" + r"$\tau=\infty$", "HgO\n" + r"$\tau=\infty$", "SGD\n" + r"$\tau=60$", "HgO\n" + r"$\tau=60$",
         "SGD\n" + r"$\tau=1$", "HgO\n" + r"$\tau=1$"]
    energy = [30000, 30000, 20700, 30000, 9000, 30000]

    x_pos = [i for i, _ in enumerate(x)]

    plt.bar(x_pos, energy, color=['blue', 'green', 'blue', 'green', 'blue', 'green'])
    plt.xlabel("")
    plt.ylabel("Received Gradients")
    # plt.title("Energy output from various fuel sources")

    plt.xticks(x_pos, x)

    plt.show()


if __name__ == '__main__':
    # grads_number()
    file = "EXP_LR_F8028E"
    info = {'ylabel': f"Test Accuracy", 'xlabel': "Rounds", 'title': "SmartFed vs. Stochastic Gradient Descent"}

    plot_std(f"../out/{file}.p", None, info)
