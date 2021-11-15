import matplotlib.pyplot as plt
import numpy as np

from src.config import EVAL_ROUND, BARS_ROUND
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
        mean, std, _ = load(mean)
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

    # mean[0] = (mean[0][0], "HgO, $f=0, ar=median$")

    for i in range(len(mean)):
        x = range(0, len(mean[i][0]) * EVAL_ROUND, EVAL_ROUND)
        plt.plot(x, mean[i][0], color=colors[i], label=mean[i][1], linestyle=line_styles[i])
        # plt.plot(mean[i][0], color=colors[i], label=mean[i][1], linestyle=line_styles[i], marker=markers[i])
        plt.fill_between(x, mean[i][0] - std[i]/2, mean[i][0] + std[i]/2, color=colors[i], alpha=.1)

    plt.rc('legend', fontsize=12)
    plt.xlabel(xlabel, fontsize=13)
    plt.xticks(fontsize=13, )
    plt.ylabel(ylabel, fontsize=13)
    plt.yticks(fontsize=13, )
    loc = 'lower right'  # lower right
    plt.legend(loc="best", shadow=True)
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


def plot_std_bar(mean, std, bars, info, unique=None):
    # plt.style.use(['dark_background'])
    if isinstance(mean, str):
        redo = True
        mean, std, bars = load(mean)
    else:
        redo = False
    xlabel = info.get('xlabel', "Rounds")
    ylabel_left = info.get('ylabel_left', "Test Accuracy")
    ylabel_right = info.get('ylabel_right', "Number of Gradients")
    # colors = ['red', 'orange', 'green', 'black', 'aqua', 'blue', 'tan', 'grey', 'navy']
    colors = ['orange', 'blue', 'red', 'black', 'green', 'aqua', 'tan', 'grey', 'navy']
    line_styles = ['-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', ]
    fig, ax1 = plt.subplots()
    ax1.set_xlabel(xlabel, fontsize=13)
    ax1.set_ylabel(ylabel_left, fontsize=13)
    # data for axis 1
    # x = list(range(0, len(mean[0][0]) * EVAL_ROUND, EVAL_ROUND))
    x = np.arange(0, len(mean[0][0]) * EVAL_ROUND, EVAL_ROUND)
    xbar = np.arange(0, len(bars[0]) * BARS_ROUND, BARS_ROUND)
    for i in range(len(mean)):
        ax1.plot(x, mean[i][0], color=colors[i], label=mean[i][1], linestyle=line_styles[i])
        ax1.fill_between(x, mean[i][0] - std[i]/2, mean[i][0] + std[i]/2, color=colors[i], alpha=.2)
    # data for axis 2
    ax2 = ax1.twinx()
    # maxi = np.max(bars) + 0.2 * np.max(bars)
    # ax2.set_ylim([0, maxi])
    ax2.set_ylabel(ylabel_right, fontsize=13)
    width = int(BARS_ROUND / (len(bars)))
    # linestyle = line_styles[i]
    for i in range(len(bars)):
        ax2.bar(xbar + i * width, bars[i], color=colors[i], width=width, alpha=0.6, zorder=-i)
    # show figure
    ax1.set_zorder(1)
    ax1.set_frame_on(False)
    ax2.set_frame_on(True)
    ax1.legend(loc="lower right", shadow=True, fontsize=10)
    # plt.grid(linestyle='dashed')
    fig.tight_layout()
    # plt.title(title)
    if not unique:
        unique = np.random.randint(100, 999)
    print(f"Saving files with code {unique}...")
    if redo:
        plt.savefig(f"../out/EXP_{unique}.pdf")
    else:
        plt.savefig(f"./out/EXP_{unique}.pdf")
    plt.show()


def plot_app(data, info=None):
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
    # x = np.arange(0, 100, 1)
    # xbars = np.arange(0, 100, 20)
    # y = np.exp(x)
    # bars = [50] * len(xbars)
    # fig, ax1 = plt.subplots()
    # ax1.plot(x, y)
    # ax2 = ax1.twinx()
    # ax2.bar(xbars, bars, width=10)
    # fig.tight_layout()
    # plt.show()
    # exit()

    # grads_number()
    file = "EXP_MLR_A8F842"
    info_ = {'ylabel': f"Test Accuracy", 'xlabel': "Rounds", 'title': "SmartFed vs. Stochastic Gradient Descent"}

    # plot_std_bar(f"../out/{file}.p", [], [], {})
    plot_std(f"../out/{file}.p", [], {})
