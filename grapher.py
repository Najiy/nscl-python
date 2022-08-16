# from asyncio.windows_events import NULL
from cProfile import label
import json
import math
# from msilib.schema import File
from threading import activeCount
from tracemalloc import start
from jinja2 import Undefined
import matplotlib
import matplotlib.pyplot as plt
import csv
import seaborn as sns
from pyparsing import col
import datetime
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.interpolate import make_interp_spline
from matplotlib.ticker import MaxNLocator
import os
# from datetime import datetime

# plt.rc('font', size=30)
# plt.rcParams.update({'font.size': 9})
plt.style.use("seaborn")
sns.set(font_scale=1.2)
sns.set_style("white")
sns.set_style("ticks")
plt.rcParams["font.size"] = "32"


def getData(data, cols=None):

    data = pd.read_csv(data)
    if cols != None:
        data = data[cols]
    data = data.fillna(0)

    neuralData = []

    for h in data:
        neuralData.append([])

    for index, row in data.iterrows():
        for i, h in enumerate(data):
            if float(row[i]) != 0.0:
                # input(f'yes {h} {i} {index}')
                neuralData[i].append(index)
            # neuralData.append([row["sensorA"], row["sensorB"], row["sensorC"], row["fsensorA"]])

    # neuralData = np.array(neuralData)

    return neuralData


def dataset(data="dataset\dataset.csv", data2="dataset\dataset2.csv", data3="dataset\dataset3.csv", data4="dataset\dataset4.csv", colls=["S0", "S1", "S2", "S3", "S4", "S5"], maxx=60):

    def colors1(c=colls):
        return ['C{}'.format(i) for i in range(len(c))]

    fig, axs = plt.subplots(3, 1)

    # create a horizontal plot
    d1 = getData(data, colls)
    xcoords = [x for x in range(0, 90)]
    for xc in xcoords:
        axs[0].axvline(x=xc, color="grey", linestyle="dotted", alpha=0.2)
    axs[0].eventplot(d1, colors=colors1(), linelengths=0.8)
    axs[0].set_yticks(range(len(colls)))
    # axs[0].set_xticks(range(90))
    axs[0].set_xlim([0, maxx])
    axs[0].set_yticklabels(colls)

    d2 = getData(data2, colls)
    # create a vertical plot
    for xc in xcoords:
        axs[1].axvline(x=xc, color="grey", linestyle="dotted", alpha=0.2)
    axs[1].eventplot(d2, colors=colors1(), linelengths=0.8)
    axs[1].set_yticks(range(len(colls)))
    # axs[1].set_xticks(range(0,1,0.01))
    axs[1].set_xlim([0, maxx])
    axs[1].set_yticklabels(colls)
    axs[1].set_ylabel("Event Streams")

    c = ["S4", "S5"]

    d3 = getData(data3, c)
    # create a vertical plot
    for xc in xcoords:
        axs[2].axvline(x=xc, color="grey", linestyle="dotted", alpha=0.2)
    axs[2].eventplot(d3, colors=colors1(c), linelengths=0.8)
    axs[2].set_yticks(range(len(c)))
    # axs[1].set_xticks(range(0,1,0.01))
    axs[2].set_xlim([0, maxx])
    axs[2].set_yticklabels(c)

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.4)
    plt.show()

    c = ["S0", "S1"]


def dataset_singly(data="dataset\dataset.csv", data2="dataset\dataset2.csv", data3="dataset\dataset3.csv", data4="dataset\dataset4.csv",  data5="dataset\dataset5.csv", colls=["S0", "S1", "S2", "S3", "S4", "S5"], maxx=90, xres=8, yres=4):

    def colors1(c=colls):
        return ['C{}'.format(i) for i in range(len(c))]

    fig, axs = plt.subplots(1, 1)

    # create a horizontal plot
    d1 = getData(data, colls)
    xcoords = [x for x in range(0, 90)]
    for xc in xcoords:
        axs.axvline(x=xc, color="grey", linestyle="dotted", alpha=0.2)
    axs.eventplot(d1, colors=colors1(), linelengths=0.8)
    axs.set_yticks(range(len(colls)))
    # axs[0].set_xticks(range(90))
    axs.set_xlim([0, maxx])
    axs.set_yticklabels(colls)
    axs.set_ylabel("Event Stream")
    axs.set_xlabel("Timesteps")

    fig.set_size_inches(xres, yres)
    plt.tight_layout()
    plt.show()

    fig, axs = plt.subplots(1, 1)
    d2 = getData(data2, colls)
    # create a vertical plot
    for xc in xcoords:
        axs.axvline(x=xc, color="grey", linestyle="dotted", alpha=0.2)
    axs.eventplot(d2, colors=colors1(), linelengths=0.8)
    axs.set_yticks(range(len(colls)))
    # axs[1].set_xticks(range(0,1,0.01))
    axs.set_xlim([0, maxx])
    axs.set_yticklabels(colls)
    axs.set_ylabel("Event Streams")
    axs.set_xlabel("Timesteps")

    fig.set_size_inches(xres, yres)
    plt.tight_layout()
    plt.show()

    fig, axs = plt.subplots(1, 1)
    # c = ["S4", "S5"]

    d3 = getData(data3, colls)
    # create a vertical plot
    for xc in xcoords:
        axs.axvline(x=xc, color="grey", linestyle="dotted", alpha=0.2)
    axs.eventplot(d3, colors=colors1(colls), linelengths=0.8)
    axs.set_yticks(range(len(colls)))
    # axs[1].set_xticks(range(0,1,0.01))
    axs.set_xlim([0, maxx])
    axs.set_yticklabels(colls)
    axs.set_ylabel("Event Streams")
    axs.set_xlabel("Timesteps")

    fig.set_size_inches(xres, yres)
    plt.tight_layout()
    plt.show()

    fig, axs = plt.subplots(1, 1)
    c = ["S0", "S1", "fS0"]

    d4 = getData(data4, c)

    # create a vertical plot
    for xc in xcoords:
        axs.axvline(x=xc, color="grey", linestyle="dotted", alpha=0.2)

    axs.eventplot(d4, colors=colors1(c), linelengths=0.8)
    axs.set_yticks(range(len(c)))
    # axs[1].set_xticks(range(0,1,0.01))
    axs.set_xlim([0, maxx])
    axs.set_yticklabels(c)
    axs.set_ylabel("Event Streams")
    axs.set_xlabel("Timesteps")

    
    flts = [380, 420, 860]

    for i, v in enumerate(d4[2]):
        axs.annotate(flts[i], (v, 2),  fontsize=12)

    # for i, txt in enumerate(d4["fS0"]):
    #     axs.annotate(txt, (1, 1))

    fig.set_size_inches(xres, yres)
    plt.tight_layout()
    plt.show()

    # D5
    fig, axs = plt.subplots(1, 1)
    c = ["S0", "S1", "fS0", "fS1", "fS2"]
    d5 = getData(data5, c)

    # create a vertical plot
    for xc in xcoords:
        axs.axvline(x=xc, color="grey", linestyle="dotted", alpha=0.2)

    axs.eventplot(d5, colors=colors1(c), linelengths=0.8)
    axs.set_yticks(range(len(c)))
    # axs[1].set_xticks(range(0,1,0.01))
    axs.set_xlim([0, maxx])
    axs.set_yticklabels(c)
    axs.set_ylabel("Event Streams")
    axs.set_xlabel("Timesteps")

    flts = [400, 420, 860]
    flts2 = [220, 880, 300]
    flts3 = [400]

    for i, v in enumerate(d5[2]):
        axs.annotate(flts[i], (v, 2),  fontsize=12)

    for i, v in enumerate(d5[3]):
        y = 3
        if i == 1:
            y = 2
        axs.annotate(flts2[i], (v, y),  fontsize=12)

    for i, v in enumerate(d5[4]):
        axs.annotate(flts3[i], (v, 4),  fontsize=12)

    # for i, txt in enumerate(d5["fS0"]):
    #     axs.annotate(txt, (1, 1))

    fig.set_size_inches(xres, yres)
    plt.tight_layout()
    plt.show()


def smooth(x, y, num=500, capzero=True):
    X_Y_Spline = make_interp_spline(x, y)

    X_ = np.linspace(min(x), max(x), num)
    Y_ = X_Y_Spline(X_)

    if capzero:
        for i in range(0, len(X_)):
            if Y_[i] < 0:
                Y_[i] = 0
    return (X_, Y_)


def compilenetmetagraph(fname="states/networks.meta", col="neurones", ylabel=None, filterout=[], xres=8, yres=4, reticky=False, smoothed=False, lblreplace=[]):

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
                name = row[0].split("_")[0]
                ex = False

                for i in filterout:
                    if i in name:
                        ex = True

                if ex == True:
                    continue

                # input(row[0].split("_")[1:-1])
                network[row[1]] = [[], [], name, []]
                # print(network[row[1]])
            network[row[1]][0].append(
                datetime.datetime.fromtimestamp(int(row[3])))
            network[row[1]][1].append(int(row[cols.index(col)]))
            network[row[1]][3].append(int(row[3]))

    # 1598434227

    xstart = 0
    xend = 0

    fig, axs = plt.subplots(1, 1)

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
            x, y = [], []

            if smoothed:
                x, y = smooth(network[key][3], network[key][1])
            else:
                x, y = network[key][3], network[key][1]
            # plt.plot(network[key][3], network[key][1], label=network[key][2], linewidth=2)
            plt.plot(x, y, label=network[key][2], linewidth=2)
            if xstart == 0 and xend == 0:
                xstart = network[key][3][0]
                xend = network[key][3][-1]
                # print(xstart, xend)

    plt.xlabel("Time")
    if ylabel == None:
        plt.ylabel(str.title(col))
    else:
        plt.ylabel(str.title(ylabel))
    # plt.title("Neurone Composites Generated over Time")

    handles, labels = plt.gca().get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))

    if len(lblreplace) == 2:
        labels = tuple(i.replace(lblreplace[0], lblreplace[1]) for i in labels)

    plt.legend(
        handles,
        labels,
        loc="lower left",
        bbox_to_anchor=(0, 1.02, 1, 0.2),
        mode="expand",
        ncol=3,
    )

    date_ranges = np.linspace(xstart-1, xend, 10)
    # dates = [datetime.datetime.fromtimestamp(x).date() for x in date_ranges]
    for xc in date_ranges:
        plt.axvline(x=xc, color="grey", linestyle="--")
    # plt.xticks(date_ranges, dates)

    if reticky:
        yint = range(min(y), math.ceil(max(y))+1)
        plt.yticks(yint)

    fig.set_size_inches(xres, yres)
    plt.tight_layout()
    # plt.show()
    plt.savefig(f"figures/{col}.png", dpi=300)
    plt.show()


def compileneuronegraph(fname="defparams.json", ticks=15, xres=8, yres=4):

    fig, axs = plt.subplots(1, 1)

    def neurone_profile(fname="defparams.json", start=0.30, ticks=15):
        with open(fname) as jsonfile:
            defparams = json.loads("\n".join(jsonfile.readlines()))

            x = [-4, -3, -2, -1]
            y = [0.11, 0.12, 0.15, 0.17]

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

            x_b = [-2, -1]
            y_b = [
                defparams["BindingThreshold"],
                defparams["BindingThreshold"],
            ]

            value = start
            refractory = 0

            for i in range(0, ticks):
                # print(i, value)

                x.append(i)
                y.append(value)

                x_t.append(i)
                y_t.append(defparams["FiringThreshold"])

                x_z.append(i)
                y_z.append(defparams["ZeroingThreshold"])

                x_b.append(i)
                y_b.append(defparams["BindingThreshold"])

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
                x_b,
                y_b,
                defparams
                # defparams["BindingThreshold"],
            )  # ,xnew, f_cubic(xnew))

    (nprof_x, nprof_y, nthres_x, nthres_y, nzero_x,
     nzero_y, binding_x, binding_y, defparams) = neurone_profile(fname, ticks=ticks)

    # plt.grid()
    xcoords = [x for x in range(-2, ticks)]
    active = [x+1 for x in xcoords if nprof_x[x]
              < defparams["BindingThreshold"]]
    active.pop()
    inactive = [x for x in xcoords if x not in active]

    for xc in active:
        plt.axvline(x=xc, color="green", linestyle="--", alpha=0.6)

    for xc in inactive:
        plt.axvline(x=xc, color="grey", linestyle="dotted", alpha=0.2)

    # for xc in xcoords:
    #     plt.axvline(x=xc, color="grey", linestyle="--")

    plt.plot(nthres_x, nthres_y, label="Firing Threshold",
             color="orange", linewidth=1.5)
    plt.plot(nzero_x, nzero_y, label="Zeroing Threshold",
             color="blue", linewidth=1.5)
    plt.plot(binding_x, binding_y, label="Binding Threshold",
             color="green", linewidth=1.5)
    plt.plot(nprof_x, nprof_y, label="Action Potential",
             color="black", linewidth=2)

    # plt.plot(
    #     [1],
    #     binding_threshold,
    #     marker="X",
    #     markersize=12,
    #     markeredgecolor="red",
    #     markerfacecolor="blue",
    # )

    plt.xlabel("Spike-relative Timestep")
    plt.ylabel("Neurone Potential")
    plt.legend(
        loc="best",
        bbox_to_anchor=(0, 1.02, 1, 0.2),
        mode="expand",
        ncol=1
    )
    plt.ylim(ymin=-0.1, ymax=1.1)
    plt.xlim([-2, 22])

    # plt.show()
    fig.set_size_inches(xres, yres)
    plt.tight_layout()
    # plt.savefig(f"figures/neurone_profile.png", dpi=300)
    plt.show()


if not os.path.isdir("figures"):
    os.mkdir("figures")


filtout = ["b2", "D2", "D3", "D4", "D5"]

# compilenetmetagraph(col='neurones', ylabel='Neurosymbols Generated',
#                     filterout=filtout, yres=5, lblreplace=["D5", "D4"])
# # compilenetmetagraph(col='composites', ylabel='Composites Generated', filterout=filtout)
# compilenetmetagraph(col='synapses', ylabel='Synapses Formed',
#                     filterout=filtout, yres=5, lblreplace=["D5", "D4"])
# compilenetmetagraph(col='npruned', ylabel='Acc. Prunes',
#                     filterout=filtout, reticky=True, yres=5, lblreplace=["D5", "D4"])

# compilenetmetagraph(col='npruned', ylabel='Composites Pruned')
# compilenetmetagraph(col='synapses', ylabel='Synapses Formed')
# compileneuronegraph(ticks=25, xres=5, yres=5)
# dataset()
# dataset_singly(yres=3)



