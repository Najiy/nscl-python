# from asyncio.windows_events import NULL
import array
from cProfile import label
from copy import copy
import json
import math
from multiprocessing.dummy import Array
from operator import index
from random import uniform
from secrets import token_bytes
# from msilib.schema import File
from threading import activeCount
from tracemalloc import start
from jinja2 import Undefined
from copy import deepcopy
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
import pprint as pp


# from datetime import datetime

# plt.rc('font', size=30)
# plt.rcParams.update({'font.size': 9})
plt.style.use("seaborn")
sns.set(font_scale=1.2)
sns.set_style("white")
sns.set_style("ticks")
plt.rcParams["font.size"] = "32"


def convert_traces(filename, scount, fcount, annot={'F3': 14, 'F1': 15}):

    fs = open(filename)
    tdata = json.load(fs)
    fs.close()

    headers = ["unix_time"]

    for i in range(scount):
        headers.append(f'S{i}')

    for i in range(fcount):
        headers.append(f'F{i}')

    neuralData = {}    

    for t in tdata["tscores_sum"]:
        entries = tdata["tscores_sum"][t]
        
        for e in entries:
            # print(t, e)
            
            offset = int(e[0].split('@')[1])
            evid = str(e[0].split('@')[0])

            t=int(t)

            if t-offset not in neuralData:
                neuralData[t-offset] = {}

            if evid not in neuralData[t-offset]:
                neuralData[t-offset][evid] = float(e[1])
            else:
                neuralData[t-offset][evid] += float(e[1])

        # cd /mnt/Data/Dropbox/PhD\ Stuff/Najiy/sourcecodes/nscl-python/feedtraces
        # python graph_traces.py

    # pp.pprint(neuralData)

    tkeys = list(neuralData.keys())
    tkeys.sort()

    lines = []

    for t in tkeys:
        entry = ["0" for x in range(len(headers))]
        entry[0] = str(t)

        for i in neuralData[t]:
            sensor = i.split('~')[0]
            entry[headers.index(sensor)] = str(neuralData[t][i])
            # print(t, sensor, neuralData[t][i])

        # print(entry)
        str_entry = ",".join(entry) + "\n"
        lines.append(str_entry)


        # print(t, neuralData[t])
        # stuff


    filename = filename.replace('.json', '.csv')
    fss = open(filename, 'w+')
    fss.write(",".join(headers) + "\n")
    fss.writelines(lines)
    fss.close()

    sensors = {}
    lencounter = 0

    for i in range(1, len(headers)):
        sensors[headers[i]] = []

    for t in tkeys:
        lencounter += 1

        for i in neuralData[t]:
            sensor = i.split('~')[0]
            sensors[sensor].append(round(float(neuralData[t][i]),3))

            # print(t, sensor, neuralData[t][i])

        for i in sensors:
            if len(sensors[i]) < lencounter:
                sensors[i].append(float(0))

    pp.pprint(sensors)

    df_data = pd.DataFrame(data=sensors)
    df_data = df_data[df_data.columns[::-1]].transpose()
    df_labels = deepcopy(df_data)

    for col in df_labels:
        df_labels[col].values[df_labels[col].values < 0.4] = 0
        df_labels[col] = df_labels[col].apply(str)
        df_labels[col].values[df_labels[col].values == '0.0'] = ''

    # flights = flights.pivot("month", "year", "passengers")
    ax = sns.heatmap(df_data, xticklabels=tkeys, annot_kws={"size": 10}, annot=df_labels, fmt='')
    plt.show()

    # print(flights)

    return filename

    # https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html
    

def load_data(filename):
    data = pd.read_csv(filename)
    print(data)


def get_data(data, cols=None):

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



def dataset_one(data=["./dataset/dataset_sin.csv", "./dataset/dataset_sin_float.csv"], colls=None, maxx=60, xres=16, yres=6, color=None):

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
                sensors_headers[i] = f"({headersValues[sensors_headers[i]]}) " + sensors_headers[i]
        
        # input(sensors_headers)

        def colors1(c=headers, color=color):
            if color == None:
                return ['C{}'.format(i) for i in range(len(c))]
            else:
                return color

        xcoords = [x for x in range(0, 90)]
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
        # plt.show()


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

convert_traces('DSB2L2_S10F10_W4_non_directional.json', 10, 10)
# load_data('DSB2L2_S10F10_W4.json')