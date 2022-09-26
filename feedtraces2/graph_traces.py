# from asyncio.windows_events import NULL
import array
from cProfile import label
from copy import copy
import json
import math
from multiprocessing.dummy import Array
from operator import index
from pickle import NONE
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
# sns.set(font_scale=1.2)
sns.set_style("white")
sns.set_style("ticks")
# sns.set_context("paper", font_scale=0.9)
plt.rcParams["font.size"] = "16"

xres = 10
yres = 4

annotLabels = {'F0': '(1) F0',
               'F1': '(144) F1',
               'F2': '(207) F2',
               'F3': '(243) F3',
               'F4': '(134) F4',
               'F5': '(116) F5',
               'F6': '(141) F6',
               'F7': '(90) F7',
               'F8': '(168) F8',
               'F9': '(113) F9'
               }

sawAnnotLabels = {'F0': '(1) F0',
                  'F1': '(191) F1',
                  'F2': '(87) F2',
                  'F3': '(146) F3',
                  'F4': '(139) F4',
                  'F5': '(41) F5',
                  'F6': '(97) F6',
                  'F7': '(20) F7',
                  'F8': '(144) F8',
                  'F9': '(225) F9'
                  }


def convert_traces(filename, scount, fcount, annotLabels, xlabel=None, ylabel=None, ax=None, maxx=None, labelfontsize=13):

    fs = open(filename)
    tdata = json.load(fs)
    fs.close()

    headers = ["unix_time"]

    for i in range(scount):
        headers.append(f'S{i}')

    for i in range(fcount):
        headers.append(f'F{i}')

    neuralData = tdata['tscores_prod']

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

    if maxx != None and len(tkeys) != maxx:
        for i in range(int(tkeys[-1])+1, int(tkeys[-1]) + (maxx+1 - len(tkeys)+1)+1):
            tkeys.append(str(i))

    for t in tkeys:
        lencounter += 1

        try:
            for i in neuralData[t]:
                sensor = i.split('~')[0]
                sensors[sensor].append(round(float(neuralData[t][i]), 3))

                # print(t, sensor, neuralData[t][i])

            for i in sensors:
                if len(sensors[i]) < lencounter:
                    sensors[i].append(float(0))
        except:
            for i in neuralData[tkeys[0]]:
                sensor = i.split('~')[0]
                sensors[sensor].append(round(0, 3))

            for i in sensors:
                if len(sensors[i]) < lencounter:
                    sensors[i].append(float(0))

    # tkeys[:] = [str(int(t)+1) for t in tkeys]

    pp.pprint(sensors)

    df_data = pd.DataFrame(data=sensors)
    df_data = df_data[df_data.columns[::-1]].transpose()
    df_labels = deepcopy(df_data)

    for col in df_labels:
        df_labels[col].values[df_labels[col].values < 0.0010] = 0
        # df_labels[col].values[df_labels[col].values > 1.0] = 1
        df_labels[col] = df_labels[col].apply(lambda x: round(x, 2))
        df_labels[col] = df_labels[col].apply(str)
        df_labels[col].values[df_labels[col].values == '0.0'] = ''

    # flights = flights.pivot("month", "year", "passengers")
    # ax =
    # plt.show()

    # print(flights)

    # input(df_labels.head())
    yticks = headers[1:]
    yticks.reverse()

    for i, v in enumerate(yticks):
        if 'F' in v:
            yticks[i] = annotLabels[v]

    # sns.lineplot(x=x, y=y*num)

    # plt.xlabel('hello')
    s = sns.heatmap(df_data, xticklabels=tkeys, yticklabels=yticks, annot_kws={
                    "size": labelfontsize-2}, annot=df_labels, fmt='', ax=ax)
    if xlabel != None:
        s.set_xlabel('Timestep', fontsize=labelfontsize)
    if ylabel != None:
        s.set_ylabel('Event Streams', fontsize=labelfontsize)

    return s

    # https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html


if not os.path.isdir("figures"):
    os.mkdir("figures")


def plotgraph(datafile, ax=None, maxx=None, subres=None, lblfontsize=16, xlabel="Timesteps", ylabel="Event Streams",  annotLabels=annotLabels):
    fig = matplotlib.pyplot.gcf()
    if subres != None:
        fig.set_size_inches(subres[0], subres[1])
        # plt.tight_layout()
    convert_traces(datafile, 10, 10,
                   annotLabels=annotLabels, ax=ax, maxx=maxx, labelfontsize=lblfontsize, xlabel=xlabel, ylabel=ylabel)
    print('plotting', datafile)
    if ax == None:
        plt.show()


def subplotgraph(datafiles=[], maxx=None, lblfontsize=10, xlabel=None, ylabel=None):
    fig, axes = plt.subplots(4, 2)
    fig.set_size_inches(12, 14)
    sns.set_context("paper", font_scale=0.7)
    for i, d in enumerate(datafiles):
        plotgraph(d, axes[int(i/2)][i % 2], maxx=maxx, lblfontsize=lblfontsize,
                  xlabel=xlabel, ylabel=ylabel)
    plt.tight_layout()
    plt.subplots_adjust(left=0.1,
                        bottom=0.03,
                        right=0.97,
                        top=0.97,
                        wspace=0.2,
                        hspace=0.15)
    plt.savefig(
        '../figures_paper2/DSB2L2_SIN2_24_33_Progressive_Unfolding.png', dpi=300)
    # plt.figure(dpi=1200)
    plt.show()


datafiles = ['B2L3_RSIN3_26_36.json',
             'B2L3_RSIN3_55_64.json',
            #  'B2L3_RSAW3_15_24.json'
             ]

for f in datafiles:
    plotgraph(f, subres=(xres, yres), lblfontsize=11,
              annotLabels=sawAnnotLabels)

datafiles2 = ['B2L3_RSIN3_26_27.json', 'B2L3_RSIN3_26_31.json',
              'B2L3_RSIN3_26_28.json', 'B2L3_RSIN3_26_32.json',
              'B2L3_RSIN3_26_29.json', 'B2L3_RSIN3_26_34.json',
              'B2L3_RSIN3_26_30.json', 'B2L3_RSIN3_26_35.json']

subplotgraph(datafiles2, maxx=10, xlabel=None, ylabel=None, lblfontsize=11)
