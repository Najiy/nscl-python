# from asyncio.windows_events import None
from cProfile import label
from copy import deepcopy
from distutils import file_util
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
    include_headers = []

    # input(cols)
    if cols != None:
        data = data[cols]
    else:
        include_headers = [x for x in data.head() if "S" in x or "F" in x]
        data = data[include_headers]
        print(data)
    data = data.fillna(0)

    neuralData = []
    valuesDict = {}

    for h in data:
        neuralData.append([])

    for index, row in data.iterrows():
        for i, h in enumerate(data):
            if float(row[i]) != 0.0:
                if h not in valuesDict.keys():
                    valuesDict[h] = row[i]
                # else:
                #     valuesDict[h].add(row[i])
                # input(f'yes {h} {i} {index}')
                neuralData[i].append(index)
            # neuralData.append([row["sensorA"], row["sensorB"], row["sensorC"], row["fsensorA"]])

    # neuralData = np.array(neuralData)
    # input(len(neuralData))

    # !!! neuralData is spikes at specific times for each streams (occurences, not values of floats)
    # input(valuesDict)

    return neuralData, include_headers, valuesDict


# def dataset(data="dataset\dataset.csv", data2="dataset\dataset2.csv", data3="dataset\dataset3.csv", data4="dataset\dataset4.csv", colls=["S0", "S1", "S2", "S3", "S4", "S5"], maxx=60):

#     def colors1(c=colls):
#         return ['C{}'.format(i) for i in range(len(c))]

#     fig, axs = plt.subplots(3, 1)

#     # create a horizontal plot
#     d1 = getData(data, colls)
#     xcoords = [x for x in range(0, 90)]
#     for xc in xcoords:
#         axs[0].axvline(x=xc, color="grey", linestyle="dotted", alpha=0.2)
#     axs[0].eventplot(d1, colors=colors1(), linelengths=0.8) # colors=colors1()
#     axs[0].set_yticks(range(len(colls)))
#     # axs[0].set_xticks(range(90))
#     axs[0].set_xlim([0, maxx])
#     axs[0].set_yticklabels(colls)

#     d2 = getData(data2, colls)
#     # create a vertical plot
#     for xc in xcoords:
#         axs[1].axvline(x=xc, color="grey", linestyle="dotted", alpha=0.2)
#     axs[1].eventplot(d2, colors=colors1(), linelengths=0.8)
#     axs[1].set_yticks(range(len(colls)))
#     # axs[1].set_xticks(range(0,1,0.01))
#     axs[1].set_xlim([0, maxx])
#     axs[1].set_yticklabels(colls)
#     axs[1].set_ylabel("Event Streams")

#     c = ["S4", "S5"]

#     d3 = getData(data3, c)
#     # create a vertical plot
#     for xc in xcoords:
#         axs[2].axvline(x=xc, color="grey", linestyle="dotted", alpha=0.2)
#     axs[2].eventplot(d3, colors=colors1(c), linelengths=0.8)
#     axs[2].set_yticks(range(len(c)))
#     # axs[1].set_xticks(range(0,1,0.01))
#     axs[2].set_xlim([0, maxx])
#     axs[2].set_yticklabels(c)

#     plt.tight_layout()
#     plt.subplots_adjust(hspace=0.4)
#     plt.show()

#     c = ["S0", "S1"]


def dataset_one(data=["./dataset/dataset_sin.csv", "./dataset/dataset_sin_float.csv"], colls=None, maxx=80, xres=16, yres=6, color=None):

    # fig, axs = plt.subplots(1, 1)

    for index, path in enumerate(data):

        file_title = path.split('/')[-1].split('.')[0]
        # input(file_title)

        fig, axs = plt.subplots(1, 1)

        # create a horizontal plot
        d1, headers, headersValues = getData(path, colls)
        sensors_headers = deepcopy(headers)

        for i,v in enumerate(headers):
            if "F" in v:
                # input(headersValues[sensors_headers[i]])
                sensors_headers[i] = f"({int(headersValues[sensors_headers[i]])}) " + sensors_headers[i]
        
        # input(sensors_headers)

        def colors1(c=headers, color=color):
            if color == None:
                return ['C{}'.format(i) for i in range(len(c))]
            else:
                return color

        xcoords = [x for x in range(0, maxx)]
        for xc in xcoords:
            axs.axvline(x=xc, color="grey", linestyle="dotted", alpha=0.2)
        axs.eventplot(d1, colors=colors1(), linelengths=0.8)
        axs.set_yticks(range(len(headers)))
        # axs[index].set_xticks(range(90))
        axs.set_xlim([0, maxx])
        axs.set_yticklabels(sensors_headers)
        axs.set_ylabel("Event Streams")
        axs.set_xlabel("Timesteps")

        # flts = [380, 420, 860]
        # for i, v in enumerate(d1[5]):
        #     axs[index].annotate(flts[i], (v, 2),  fontsize=12)

        fig.set_size_inches(xres, yres)
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.4)
        plt.savefig(f'./figures_paper2/{file_title}')
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


if not os.path.isdir("figures"):
    os.mkdir("figures")


# filtout = ["b2", "D2", "D3", "D4", "D5"]

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


# dataset_one(data=["./dataset/dataset_sin_S3F3.csv",
#             "./dataset/dataset_sin_S5F5.csv", "./dataset/dataset_sin_S10F10.csv","./dataset/dataset_sin_S6.csv",
#             "./dataset/dataset_sin_S10.csv", "./dataset/dataset_sin_S20.csv" ])


dataset_one(data=["./dataset/dataset_sin_S10F10.csv" ])


# dataset_one("./dataset/dataset_sin_float.csv", colls=["S0","S1","S2","S3","S4","F0","F1","F2","F3","F4" ])
# print(getData('./dataset/dataset_sin.csv', cols=None))
