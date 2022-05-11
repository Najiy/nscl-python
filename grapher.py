from asyncio.windows_events import NULL
from cProfile import label
import json
from tracemalloc import start
from jinja2 import Undefined
import matplotlib.pyplot as plt
import csv, seaborn as sns
from pyparsing import col
import datetime
import numpy as np
from scipy.interpolate import interp1d
from scipy.interpolate import make_interp_spline
import os

# plt.rc('font', size=30)
# plt.rcParams.update({'font.size': 9})
plt.style.use("seaborn")
sns.set(font_scale=1.2)
plt.rcParams["font.size"] = "32"


def smooth(x, y, num=500, capzero=True):
    X_Y_Spline = make_interp_spline(x, y)

    X_ = np.linspace(min(x), max(x), num)
    Y_ = X_Y_Spline(X_)

    if capzero:
        for i in range(0, len(X_)):
            if Y_[i] < 0:
                Y_[i] = 0
    return (X_, Y_)


def neurone_profile(fname="defparams.json"):
    with open(fname) as jsonfile:
        defparams = json.loads("\n".join(jsonfile.readlines()))

        x = [-2, -1]
        y = [0.75, 0.75]

        x_t = [-2, -1]
        y_t = [
            defparams["FiringThreshold"],
            defparams["FiringThreshold"],
        ]

        x_z = [-2, -1]
        y_z = [
            defparams["ZeroingThreshold"],
            defparams["ZeroingThreshold"],
        ]

        value = 0.8
        refractory = 0

        for i in range(0, 12):
            print(i, value)

            x.append(i)
            y.append(value)

            x_t.append(i)
            y_t.append(defparams["FiringThreshold"])

            x_z.append(i)
            y_z.append(defparams["ZeroingThreshold"])

            if value < defparams["ZeroingThreshold"]:
                value = 0
            elif value >= defparams["FiringThreshold"] and refractory == 0:
                refractory = defparams["RefractoryPeriod"]
                value = 1.0
                # x.append(i)
                # y.append(value)
                value *= defparams["PostSpikeFactor"]
            else:
                value *= defparams["DecayFactor"]

            if refractory > 0:
                refractory = -1

        # x = np.linspace(-2, 12, num=11, endpoint=True)
        # y = np.cos(-x**2/9.0)
        # xnew = np.linspace(0, 15, num=41, endpoint=True)

        # f_cubic = interp1d(x, y, kind='cubic')
        # x, y = smooth(x, y)

        for i in range(0, len(x)):
            if y[i] < defparams["ZeroingThreshold"]:
                y[i] = 0

        return (
            x,
            y,
            x_t,
            y_t,
            x_z,
            y_z,
            defparams["BindingThreshold"],
        )  # ,xnew, f_cubic(xnew))


def compilenetmetagraph(fname="states/networks.meta", col="neurones", ylabel=NULL):
    network = {}
    cols = []

    with open(fname, "r") as csvfile:
        plots = csv.reader(csvfile, delimiter=",")
        head = False
        for row in plots:
            if head == False:
                cols = row
                head = True
                continue
            if row[1] not in network:
                # input(row[0].split("_")[1:-1])
                network[row[1]] = [[], [], row[0].split("_")[0], []]
                print(network[row[1]])
            network[row[1]][0].append(datetime.datetime.fromtimestamp(int(row[3])))
            network[row[1]][1].append(int(row[cols.index(col)]))
            network[row[1]][3].append(int(row[3]))

    # 1598434227

    xstart = 0
    xend = 0

    for key in network:
        if network[key][2] not in [
            # "b2l2",
            # "b2l3",
            # "b2l4",
            # "b3l2",
            # "b3l3",
            # "b3l4",
            # "b4l2",
            # "b4l3",
            # "b4l4"
        ]:
            # x, y = smooth(network[key][3], network[key][1])
            x, y = network[key][3], network[key][1]
            # plt.plot(network[key][3], network[key][1], label=network[key][2], linewidth=2)
            plt.plot(x, y, label=network[key][2], linewidth=2)
            if xstart == 0 and xend == 0:
                xstart = network[key][3][0]
                xend = network[key][3][-1]
                print(xstart, xend)

    plt.xlabel("Time")
    if ylabel == NULL:
        plt.ylabel(str.title(col))
    else:
        plt.ylabel(str.title(ylabel))
    # plt.title("Neurone Composites Generated over Time")

    handles, labels = plt.gca().get_legend_handles_labels()
    # sort both labels and handles by labels
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
    plt.legend(
        handles,
        labels,
        loc="lower left",
        bbox_to_anchor=(0, 1.02, 1, 0.2),
        mode="expand",
        ncol=3,
    )
    date_ranges = np.linspace(xstart, xend, 6)
    dates = [datetime.datetime.fromtimestamp(x).date() for x in date_ranges]
    for xc in date_ranges:
        plt.axvline(x=xc, color="grey", linestyle="--")
    plt.xticks(date_ranges, dates)
    plt.tight_layout()
    # plt.show()
    plt.savefig(f"figures/{col}.png", dpi=300)
    plt.show()


def compileneuronegraph(fname="defparams.json"):
    (nprof_x, nprof_y, nthres_x, nthres_y, nzero_x, nzero_y, binding_threshold) = neurone_profile(fname)

    # plt.grid()
    xcoords = [x for x in range(-2, 12)]
    for xc in xcoords:
        plt.axvline(x=xc, color="grey", linestyle="--")

    plt.plot(nthres_x, nthres_y, label="Firing Threshold", color="orange", linewidth=2)
    plt.plot(nzero_x, nzero_y, label="Zeroing Threshold", color="blue", linewidth=2)
    plt.plot(nprof_x, nprof_y, label="Action Potential", color="black", linewidth=3)

    plt.plot(
        [1],
        binding_threshold,
        marker="X",
        markersize=12,
        markeredgecolor="red",
        markerfacecolor="blue",
    )

    plt.xlabel("Time")
    plt.ylabel("Action Potential")
    plt.legend(
        loc="lower left",
        bbox_to_anchor=(0, 1.02, 1, 0.2),
        mode="expand",
        ncol=6,
        # frameon=True,
    )

    plt.tight_layout()
    # plt.show()
    plt.savefig(f"figures/neurone_profile.png", dpi=300)
    plt.show()


if not os.path.isdir("figures"):
    os.mkdir("figures")

# compilenetmetagraph(col='composites', ylabel='Composites Generated')
# compilenetmetagraph(col='npruned', ylabel='Composites Pruned')
# compilenetmetagraph(col='synapses', ylabel='Synapses Formed')
compileneuronegraph()
