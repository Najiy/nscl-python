#!/usr/bin/python

# from msilib.schema import File
from audioop import reverse
from cgi import print_environ
from copy import deepcopy
from distutils import errors
from genericpath import exists
from tracemalloc import start
import typing

from curses import meta
from datetime import date, datetime
from decimal import DivisionByZero
from inspect import trace
from re import search
import math
import hashlib

# from nscl_algo import NSCLAlgo
# from nscl_algo import NSCLAlgo

# from nscl_algo import NSCLAlgo
import pprint
import os
from re import split, template
import time
import sys
import json
import pprint
import subprocess

# from subprocess import Popen
from typing import NewType

# from jinja2.defaults import NEWLINE_SEQUENCE

from networkx.algorithms.planarity import Interval
from networkx.generators.geometric import random_geometric_graph
from pandas.core.algorithms import take
from nscl import NSCL
import matplotlib.pyplot as plt
from nscl_algo import NSCLAlgo
import seaborn as sns
import numpy as np
import pandas as pd
import networkx as nx
from nscl_predict import NSCLPredict as npredict
from pyvis.network import Network

import itertools
from itertools import islice


# test = [None, None, 2, None, None]
# exists = any(x != None for x in test)
# input(exists)


# dt = datetime.now().isoformat(timespec="minutes")
# input(dt)

# from networkx.drawing.nx_agraph import graphviz_layout

pathdiv = "/" if os.name == "posix" else "\\"
# pathdiv = "/" if os.name == "posix" else "\\"


def clear():
    if os.name == "posix":
        return os.system("clear")
    elif os.name == "nt":
        return os.system("cls")


themes = ["viridis", "mako_r", "flare"]

eng = NSCL.Engine()
# synapses = eng.network.synapses
# neurones = eng.network.neurones


def jprint(obj) -> None:
    js = json.dumps(obj, default=lambda x: x.__dict__, indent=4)
    print(js)


def reshape_trace(eng) -> list:
    print(len(eng.traces))
    timelength = len(eng.traces[-1])

    r = []
    for t in eng.traces:
        n = [0 for i in range(0, timelength)]
        for i, v in enumerate(t):
            n[i] = v
        r.append(n)

    # print("trace shape %s \n" % str(np.shape(r)))
    return r


def flatten(traces) -> list:
    return [item for sublist in traces for item in sublist]


def heatmap(
    result_path,
    traces,
    neurones,
    figsize=(24, 12),
    save=False,
    ax=None,
    theme="viridis",
):
    print(" -heatplot")

    my_colors = [(0.2, 0.3, 0.3), (0.4, 0.5, 0.4),
                 (0.1, 0.7, 0), (0.1, 0.7, 0)]

    tlength = len(traces)
    arr_t = np.array(traces).T
    # c = sns.color_palette("vlag", as_cmap=True)
    # c = sns.cubehelix_palette(start=2, rot=0, dark=0, light=.95, reverse=True, as_cmap=YlGnBu)

    if save == True:
        plt.figure(figsize=figsize)

    heatplot = sns.heatmap(
        arr_t,
        # linewidth= (0.01 if tlength < 100 else 0.0),
        linewidths=0.01,
        yticklabels=neurones,
        # cmap="YlGnBu",
        cmap=theme,
        # cmap=my_colors,
        vmin=0,
        vmax=1,
        # linewidths=0.01,
        square=True,
        linecolor="#222",
        annot_kws={"fontsize": 11},
        # linecolor=(0.1,0.2,0.2),
        xticklabels=10,
        cbar=save,
        ax=ax,
    )

    # colorbar = ax.collections[0].colorbar
    # M=dt_tweet_cnt.max().max()
    # colorbar.set_ticks([1/8*M,3/8*M,6/8*M])
    # colorbar.set_ticklabels(['low','med','high'])

    if save == True:
        plt.tight_layout()
        plt.savefig(result_path + pathdiv + r"Figure_1.png", dpi=120)
        plt.clf()

    return heatplot


def lineplot(
    result_path, data_plot, figsize=(15, 8), save=False, ax=None, theme="flare"
):
    print(" -lineplot")
    if save == True:
        plt.figure(figsize=figsize)

    lplot = sns.lineplot(
        x="time", y="ncounts", data=data_plot, hue="ntype", ax=ax
    )  # , palette=theme)

    # for i in data_plot:
    #     input(data_plot[i])

    # lplot = plt.stackplot(x=range(0,60), y=data_plot, labels=["inputs", "generated", "total"])

    if save == True:
        plt.tight_layout()
        plt.savefig(result_path + pathdiv + r"Figure_2.png", dpi=120)
        plt.clf()

    return lplot


def networkx(result_path, synapses, figsize=(24, 12), save=False, ncolor="skyblue"):
    print(" -networkplot")
    plt.figure(figsize=figsize)
    # Build a dataframe with your connections
    df = pd.DataFrame(
        {
            "from": [synapses[s].rref for s in synapses],
            "to": [synapses[s].fref for s in synapses],
            "value": [synapses[s].wgt for s in synapses],
        }
    )

    # Build your graph
    G = nx.from_pandas_edgelist(df, "from", "to", create_using=nx.DiGraph())

    # pos = graphviz_layout(G, prog='dot')

    # Custom the nodes:
    nx.draw(
        G,
        # pos,
        with_labels=True,
        node_color=ncolor,
        node_size=1200,
        edge_color=df["value"],
        width=5.0,
        alpha=0.9,
        edge_cmap=plt.cm.Blues,
        arrows=True,
        # pos=nx.layout.planar_layout(G)
    )
    # plt.show()
    # plt.figure(figsize=figsize)
    if save == True:
        plt.savefig(result_path + pathdiv + r"Figure_3.png", dpi=120)
        plt.clf()

    return G


def networkx_pyvis(
    result_path, synapses, figsize=(24, 12), save=False, ncolor="skyblue"
):
    print(" -networkplot")
    plt.figure(figsize=figsize)
    # Build a dataframe with your connections
    df = pd.DataFrame(
        {
            "from": [synapses[s].rref for s in synapses],
            "to": [synapses[s].fref for s in synapses],
            "value": [synapses[s].wgt for s in synapses],
        }
    )

    # Build your graph
    G = nx.from_pandas_edgelist(df, "from", "to", create_using=nx.DiGraph())

    # pos = graphviz_layout(G, prog='dot')

    # Custom the nodes:
    nx.draw(
        G,
        # pos,
        with_labels=True,
        node_color=ncolor,
        node_size=1200,
        edge_color=df["value"],
        width=5.0,
        alpha=0.9,
        edge_cmap=plt.cm.Blues,
        arrows=True,
        # pos=nx.layout.planar_layout(G)
    )
    # plt.show()
    # plt.figure(figsize=figsize)
    if save == True:
        plt.savefig(result_path + pathdiv + r"Figure_3.png", dpi=120)
        plt.clf()

    #####################

    net = Network(notebook=False, width=1600, height=900)
    net.toggle_hide_edges_on_drag(False)
    net.barnes_hut()
    net.from_nx(nx.davis_southern_women_graph())
    net.show("ex.html")

    return G


def jsondump(result_path, fnameext, jdata):
    if not exists(result_path):
        os.mkdir(result_path)
    with open(result_path + pathdiv + fnameext, "w") as outfile:
        json.dump(jdata, outfile)


def graphout(eng, flush=True):
    tt = str(datetime.now().replace(microsecond=0)).replace(":", "_")
    rpath = r"results%s%s" % (pathdiv, tt)

    if os.name != "nt":
        rpath = "results%s%s" % (pathdiv, tt)
    eng.traces = reshape_trace(eng)

    print("%s" % rpath)

    if not os.path.exists("results"):
        os.mkdir("results")

    if not os.path.exists(rpath):
        os.mkdir(rpath)

    fig, ax = plt.subplots(2, 1, figsize=(24, 12))
    df = pd.DataFrame(
        {"time": eng.ntime, "ncounts": eng.ncounts, "ntype": eng.nmask})
    sns.set_theme()
    heatmap(rpath, eng.traces, eng.network.neurones, ax=ax[0])
    lineplot(rpath, df, ax=ax[1])
    plt.savefig(f"{rpath}{pathdiv}Figure_1-2.png", dpi=120)
    plt.clf()
    heatmap(rpath, eng.traces, eng.network.neurones, save=True)
    lineplot(rpath, df, save=True)
    networkx(rpath, eng.network.synapses, save=True)
    networkx_pyvis(rpath, eng.network.synapses, save=True)

    if flush == True:
        eng.clear_traces()


def stream(streamfile, trace=True):

    # text = "Top Cat! The most effectual Top Cat! Who’s intellectual close friends get to call him T.C., providing it’s with dignity. Top Cat! The indisputable leader of the gang. He’s the boss, he’s a pip, he’s the championship. He’s the most tip top, Top Cat."
    # txt_arr = text.lower().split(' ')
    # for i,v in enumerate(txt_arr):
    #     inputs[i] = [v]
    # input(inputs)

    filecontent = json.loads(
        open(f"dataset{pathdiv}{streamfile}.json", "r").read())

    interv = filecontent["interval"]
    inputs = filecontent["activity_stream"]

    temp = eng.tick
    eng.tick = 0

    maxit = min(len(inputs) - 1, interv)
    running = True

    while running and eng.tick <= maxit:
        try:
            clear()

            print()
            print(" ###########################")
            print("     NSCL_python  t =", eng.tick)
            print(" ###########################")
            print()

            eng.algo(inputs[eng.tick], meta, trace)

            if eng.tick == maxit:
                graphout(eng)

        except KeyboardInterrupt:
            running = False

    eng.tick += temp

    print("\n\n test streaming done.")
    print()


def normaliser(data, minn, maxx, scaling=1):
    try:
        return (data - minn) / (maxx - minn) * scaling
    except DivisionByZero:
        return 0


def csvstream(streamfile, metafile, trace=False, fname="default", iterations=1):
    # def take(n, iterable):
    #     # "Return first n items of the iterable as a list"
    #     return list(islice(iterable, n))

    data = {}

    metafile = open(metafile, "r")
    headers = metafile.readline().split(",")
    metadata = metafile.readlines()

    for line in metadata:

        row = line.split(",")
        # print(row)
        eng.meta[row[0]] = {
            "min": float(row[9]),
            "max": float(row[10]),
            "res": float(eng.network.params["DefaultEncoderResolution"]),
        }

        # eng.meta[row[0]] = {
        #     "min": eng.network.params["DefaultEncoderFloor"],
        #     "max": eng.network.params["DefaultEncoderCeiling"],
        #     "res": eng.network.params["DefaultEncoderResolution"],
        # }

    file = open(streamfile, "r")
    sensors = file.readline().split(",")[1:]
    rawdata = []
    rawcp = deepcopy(file.readlines())

    # print(len(rawdata))
    for i in range(iterations):
        rawdata += deepcopy(rawcp)
    # print(len(rawdata))
    # input()

    for i in range(len(rawdata)):
        # print(i, rawdata[i])
        entry = rawdata[i].split(',')
        entry[0] = str(i)
        rawdata[i] = ",".join(entry)

        # input(entry)

    # print(len(rawdata), rawdata[-1])
    # pp.pprint(rawdata[595: 615])

    # for line in rawdata:
    #     sline = line.replace("\n", "").split(",")
    #     for i in range(1, 1 + len(sensors)):
    #         if sline[i] != "": ## filters sensors
    #             sline[i] = f"{sensors[i-1]}~{sline[i]}"
    #     data[int(sline[0])] = [x for x in sline[1:] if x != ""]

    for line in rawdata:
        try:
            sline = line.replace("\n", "").split(",")
            for i in range(1, 1 + len(sensors)):
                if sline[i] != "":  # filters sensors
                    name = sensors[i - 1]
                    value = float(sline[i])

                    if (
                        search("current", name)
                        or search("humidity", name)
                        or search("temperature", name)
                    ):
                        sline[i] = ""
                    else:
                        maxx = eng.network.params["DefaultEncoderCeiling"]
                        minn = eng.network.params["DefaultEncoderFloor"]
                        res = eng.network.params["DefaultEncoderResolution"]

                        if name in eng.meta:
                            maxx = eng.meta[name]["max"]
                            minn = eng.meta[name]["min"]
                            res = eng.meta[name]["res"]

                        newval = math.floor(normaliser(value, minn, maxx, res))
                        # input(f"normaliser({value},{minn},{maxx},{res}) = {newval}")
                        sline[i] = f"{name}~{newval}-{newval+1}"
            data[int(sline[0])] = [x for x in sline[1:] if x != ""]
        except:
            print(line)

    start = int(rawdata[0].split(",")[0])
    end = int(rawdata[-1].split(",")[0])

    del rawdata

    # for i, v in enumerate(eng.meta):
    #     print(i, v["min"], v["max"], v["res"])

    print(eng.network.hash_id)
    print(start, end)
    save_range = [x for x in range(
        start + 1, end, int(round((end - start) / 20, 1)))]
    print(save_range)

    input("csv dataset loaded, now processing [enter] ")

    temp = eng.tick
    eng.tick = start
    maxit = end

    # maxit = min(len(inputs) - 1, interv)
    running = True
    # netmeta = open(f"{fname}.netmeta", "w+")

    starttime = datetime.now().isoformat(timespec="minutes")

    next_prompt = True

    while running and eng.tick <= maxit:

        # print("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

        # if not skip:
        #     for n in eng.network.neurones:
        #         print(f"{n}", end=" ")
        #     print()
        #     for n in eng.network.neurones:
        #         print(f"{eng.network.neurones[n].potential:0.2f}", end=" ")
        #     print()

        try:
            if eng.prune == 0:
                eng.prune = eng.network.params["PruneInterval"]
            eng.prune -= 1

            if eng.tick % 5000 == 0 or eng.tick == maxit or eng.prune == 0:
                clear()

                print()
                print(" ###########################")
                print(
                    f"     NSCL_python \n streming file {fname} at t(eng) {eng.tick}")
                print(f"hashid = {eng.network.hash_id}")
                print(f"start = {starttime}")
                print(f"saverange = {save_range}")
                print(
                    f"progress = {(eng.tick - start) / (end - start) * 100 : .1f}%")
                print(f"neurones = {len(eng.network.neurones)}")
                print(f"synapses = {len(eng.network.synapses)}")
                print(f"bindings = {eng.network.params['BindingCount']}")
                print(f"proplevel = {eng.network.params['PropagationLevels']}")
                print(
                    f"encres = {eng.network.params['DefaultEncoderResolution']}")
                print(f"npruned = {len(eng.npruned)}")
                print(f"spruned = {len(eng.spruned)}")
                print(f"prune_ctick = {eng.prune}")
                print(" ###########################")
                print()

            print('\n', eng.tick)

            if eng.tick not in data:
                eng.algo([])
                print('input', [])
            else:
                # pp.pprint(data[eng.tick])
                # input()
                # input(data[eng.tick])
                eng.algo(data[eng.tick])
                print('input', data[eng.tick-1])

            pp.pprint(eng.network.neurones.keys())

            if next_prompt and input("[enter] for next step ('r' for no-prompt): ") == 'r':
                next_prompt = False

            if eng.tick == maxit and trace == True:
                graphout(eng)

            if eng.tick in save_range:
                save_range.remove(eng.tick)
                # os.mkdir(f'state{pathdiv}{fname}')
                # eng.save_state(f'{fname}{pathdiv}{eng.tick}')
                eng.save_state(f"{fname}_{eng.tick}")

            if eng.tick == maxit:
                input("csv stream done! [enter]")

        except KeyboardInterrupt:
            running = False

        # if input(f"\n{eng.tick-1}") == "s" and skip == False:
        #     skip = True

    # netmeta.close()
    eng.tick += temp

    print("\n csv streaming done.")
    print()


def load_data(streamfile, metafile, trace=False, fname="default", exclude_values=["", "0"], exclude_headers=["current", "humidity", "temperature"]):
    # def take(n, iterable):
    #     # "Return first n items of the iterable as a list"
    #     return list(islice(iterable, n))

    data = {}

    metafile = open(metafile, "r")
    headers = metafile.readline().split(",")
    metadata = metafile.readlines()

    for line in metadata:

        row = line.split(",")
        eng.meta[row[0]] = {
            "min": float(row[9]),
            "max": float(row[10]),
            "res": float(eng.network.params["DefaultEncoderResolution"])
        }

    file = open(streamfile, "r")
    sensors = file.readline().split(",")[1:]
    rawdata = file.readlines()

    for line in rawdata:
        sline = line.replace("\n", "").split(",")
        for i in range(1, 1 + len(sensors)):
            if sline[i] != "":  # filters sensors
                name = sensors[i - 1]
                value = float(sline[i])

                exclude = all(n != None for n in [
                              search(x, name) for x in exclude_headers])

                if (exclude):
                    sline[i] = ""
                else:
                    maxx = eng.network.params["DefaultEncoderCeiling"]
                    minn = eng.network.params["DefaultEncoderFloor"]
                    res = eng.network.params["DefaultEncoderResolution"]

                    if name in eng.meta:
                        maxx = eng.meta[name]["max"]
                        minn = eng.meta[name]["min"]
                        res = eng.meta[name]["res"]

                    newval = math.floor(normaliser(value, minn, maxx, res))
                    # input(f"normaliser({value},{minn},{maxx},{res}) = {newval}")
                    sline[i] = f"{name}~{newval}-{newval+1}"

        data[int(sline[0])] = [x for x in sline[1:] if x != ""]

    del rawdata
    return data


def feed_trace_old(eng, tfirst, data, ticks=[], p_ticks=0, pot_threshold=0.8, reset_potentials=True):

    binding_window, refractory_period, reverberating, reverb_window = eng.network.check_params(
        prompt_fix=False)
    tick = tfirst

    if ticks == []:
        ticks = [t for t in range(max(data.keys()))]

    htraces = {}
    tscores = {}
    datainputs = {}
    refract_suppress = {}

    if reset_potentials:
        eng.reset_potentials()

    def neurones_levels(name):
        total = {}
        for lvls in eng.network.neurones[name].heirarcs.keys():
            if lvls not in total:
                total[lvls] = len(eng.network.neurones[name].heirarcs[lvls])
        return total

    def score(potential, ncount, heirarcs):
        scores = {}

        for lvls in heirarcs:
            for hns in heirarcs[lvls]:
                if hns not in scores:
                    scores[f"{hns}@{lvls+1}"] = potential / ncount
                    # scoring based on composites potential over number of connected synapses.

        return scores

    for t in ticks:

        # print('#####################################')
        activated = []

        # refract_suppress[t]

        for n in refract_suppress:
            refract_suppress[n] -= 1
            if refract_suppress[n] == 0:
                del refract_suppress[n]

        if t not in data.keys():
            r, e, activated = eng.algo([])
            # print(f"{tick} data", [])
        else:
            # print(f"{tick} data", data[t])
            datainputs[t] = data[t]
            r, e, activated = eng.algo(data[t])

        for n in activated:
            if n not in refract_suppress.keys():
                refract_suppress[n] = refractory_period

        # print(activated)
        actives = eng.get_actives(pot_threshold)
        htraces[tick] = []
        tscores[tick] = {}

        for n in actives:

            ncounts = sum(neurones_levels(n[0]).values())
            scores = score(n[1], ncounts, eng.network.neurones[n[0]].heirarcs)
            entry = (n[1], n[0], scores)
            htraces[tick].append(entry)

            for s in scores:
                if s not in tscores[tick]:
                    tscores[tick][s] = 0
                tscores[tick][s] += scores[s]

        tscores[tick] = [(x, tscores[tick][x]) for x in tscores[tick] if x.split(
            '@')[0] not in refract_suppress.keys()]
        tscores[tick].sort(reverse=True, key=lambda x: x[1])
        htraces[tick].sort(reverse=True, key=lambda x: x[0])

        # pp.pprint(htraces[t])
        # pp.pprint(tscores[t])
        # input()

        tick += 1

    for t in range(p_ticks):

        # print('#####################################')

        r, e, a = eng.algo([])
        # print(f"{tick} ptick", [])

        actives = eng.get_actives(pot_threshold)
        htraces[tick] = []
        tscores[tick] = {}

        for n in actives:

            ncounts = sum(neurones_levels(n[0]).values())
            scores = score(n[1], ncounts, eng.network.neurones[n[0]].heirarcs)
            entry = (n[1], n[0], scores)
            htraces[tick].append(entry)

            for s in scores:
                if s not in tscores[tick]:
                    tscores[tick][s] = 0
                tscores[tick][s] += scores[s]

        tscores[tick] = [(x, tscores[tick][x]) for x in tscores[tick]]
        tscores[tick].sort(reverse=True, key=lambda x: x[1])
        htraces[tick].sort(reverse=True, key=lambda x: x[0])

        # pp.pprint(htraces[t])
        # pp.pprint(tscores[tick])
        # input()

        tick += 1

    # pp.pprint(htraces)

    return tscores, htraces, datainputs


def feed_trace(eng, tfirst, data, ticks=[], p_ticks=0, reset_potentials=True):

    binding_window, refractory_period, reverberating, reverb_window = eng.network.check_params(
        prompt_fix=False)

    tick = tfirst

    if ticks == []:
        ticks = [t for t in range(max(data.keys()))]

    pscores = {}
    datainputs = {}
    templist = []

    if reset_potentials:
        eng.reset_potentials()

    # propagate & capture
    for t in ticks:

        if t not in data.keys():
            r, e, a = eng.algo([])
            for i in a:
                templist.append((tick, i, eng.network.neurones[i].potential))
        else:
            datainputs[t] = data[t]
            r, e, a = eng.algo(data[t])
            for i in a:
                templist.append((tick, i, eng.network.neurones[i].potential))
        tick += 1

    for t in range(p_ticks):

        # curractives = [
        #     x for x in templist if 'CMP' not in x[1] and x[0] == tick-1]
        # curractives.sort(key=lambda x: x[2], reverse=True)
        # window = [x for x in templist if x[0] > tick - 3]
        # window.sort(key=lambda x: x[2], reverse=True)

        # print('\n\n', tick)
        # print('current actives')
        # pp.pprint(curractives)
        # print('window actives')
        # pp.pprint(window)
        # input()

        # included
        r, e, a = eng.algo([])
        for i in a:
            templist.append((tick, i, eng.network.neurones[i].potential))
        tick += 1

        # pass

    # pp.pprint(templist)
    # input()

    for i in templist:
        # print(i[0], ' prime for ', i[1],
        #       npredict.primeProductWeights(i[1], eng))
        # input()

        if 'CMP' not in i[1]:
            if i[0] not in pscores:
                pscores[i[0]] = {}
            if i[1] not in pscores[i[0]]:
                pscores[i[0]][i[1]] = round(i[2], 4)
            else:
                pscores[i[0]][i[1]] += round(i[2], 4)
        else:
            s = npredict.primeProductWeights(i[1], eng)
            stotal = i[2]
            # print('s is ', s, ' total is ', i[2])

            for j in s:
                # print('t is ', i[0], 'offset is ', s[j][1])
                # k = s[j][1]+i[0]
                try:
                    if s[j][1]+i[0] not in pscores:
                        pscores[s[j][1] + i[0]] = {}
                    if j not in pscores[s[j][1]+i[0]]:
                        pscores[s[j][1]+i[0]][j] = s[j][0] * stotal
                    else:
                        # print(pscores[s[j][1]+i[0]][j])
                        pscores[s[j][1]+i[0]][j] += s[j][0] * stotal
                except:
                    continue

        # pp.pprint(pscores)
        # input()

    vals = [x for x in pscores]
    lowest = min(vals)
    highest = max(vals)
    # input(f'{lowest} {highest}')

    for i in range(lowest, highest):
        if i not in pscores:
            pscores[i] = {}

    return pscores, datainputs


def feed_trace_unfold_old(eng, datainputs, pticks=0, reset_potentials=True):

    def feed_trace_once(eng, dinputs, timeat, reset_potentials, pscores, datainputs):

        pscores = {}
        datainputs = {}

        if reset_potentials:
            eng.reset_potentials()

        # propagate & capture
        # for t in datainputs:
        if len(dinputs) == 0:
            r, e, a = eng.algo([])
            for i in a:
                templist.append(
                    (tstart+t, i, eng.network.neurones[i].potential))
        else:
            r, e, a = eng.algo(datainputs[t])
            for i in a:
                templist.append(
                    (tstart+t, i, eng.network.neurones[i].potential))

        # for t in range(pticks):
        #     r, e, a = eng.algo([])
        #     for i in a:
        #         templist.append((tick, i, eng.network.neurones[i].potential))
        #     tick += 1

        templist = []

        for i in templist:

            if 'CMP' not in i[1]:
                if i[0] not in pscores:
                    pscores[i[0]] = {}
                if i[1] not in pscores[i[0]]:
                    pscores[i[0]][i[1]] = round(i[2], 4)
                else:
                    pscores[i[0]][i[1]] += round(i[2], 4)
            else:
                s = npredict.primeProductWeights(i[1], eng)
                stotal = i[2]

                for j in s:
                    try:
                        if s[j][1]+i[0] not in pscores:
                            pscores[s[j][1] + i[0]] = {}
                        if j not in pscores[s[j][1]+i[0]]:
                            pscores[s[j][1]+i[0]][j] = s[j][0] * stotal
                        else:
                            pscores[s[j][1]+i[0]][j] += s[j][0] * stotal
                    except:
                        continue

        vals = [x for x in pscores]
        lowest = min(vals)
        highest = max(vals)

        for i in range(lowest, highest):
            if i not in pscores:
                pscores[i] = {}

        return pscores, datainputs

    tstart = min(datainputs.keys())
    tfrom = max(datainputs.keys())
    tto = tfrom + pticks

    # current pscores, datainputs from feed_trace
    pscores, datainputs = feed_trace_once(
        eng, datainputs=datainputs, tstart=tstart, reset_potentials=reset_potentials)

    for t in range(tfrom, tto):
        window = [i for i in pscores if i >= t]
        scores = {}
        # scores = []

        for tt in window:
            for it in pscores[tt]:
                if tt == tfrom and it in datainputs[tt]:
                    continue
                elif pscores[tt][it] >= 1:
                    # scores.append((it, pscores[tt][it]))
                    if it not in scores:
                        scores[it] = pscores[tt][it]
                    else:
                        scores[it] += pscores[tt][it]

        datainputs[t+1] = [x for x in scores if scores[x] >= 2]

        print()
        pp.pprint(datainputs)
        # print()
        # pp.pprint(scores)
        print()
        pp.pprint(pscores)
        input(f"{t} {window}")

        print('feeding', datainputs)
        pscores, datainputs = feed_trace_once(
            eng, datainputs=datainputs, pticks=pticks, tstart=tstart, reset_potentials=reset_potentials)
        print('post', datainputs)

    return pscores, datainputs



######### main prog starts #########
args = sys.argv[1:]

# print(" ".join(args))

pp = pprint.PrettyPrinter(indent=4)

init = False
verbose = False

while True:
    # try:
    if init == False:
        print("\n\n########### NSCL (Python) ###########\n")
        print(f" version: experimental/non-optimised")
        print(f" os.name: {os.name}")
        print(f" os.pid: {os.getpid()}")
        print("")

        # subprocess.call(f'top -p {os.getpid()}', shell=True)
        # os.system(f"top -p {os.getpid()}")
        # Popen('bash')

    # try:
    if init == True:
        if os.name == "posix":
            command = input(
                f"\033[1m\033[96m {os.getpid()}: NSCL [{eng.tick}]> \u001b[0m\033[1m"
            ).split(" ")
        else:
            command = input(f" {os.getpid()}: NSCL [{eng.tick}]> ").split(" ")
    else:
        init = True
        command = args

    if len(command) == 0:
        continue

    if command[0] in ["clear", "cls", "clr"]:
        clear()

    if command[0] in ["param", "set"]:
        if command[1] in ["verb", "verbose"]:
            verbose = bool(command[2])
            print(f"verbose={verbose}")

    if command[0] in ["check", "checkparams"]:
        binding_window, refractory_period, reverberating, reverb_window = eng.network.check_params()
        print("Binding Window =", binding_window)
        print("Current Refractoryperiod =", refractory_period)
        print("Reverberating Firing =", reverberating)
        print("Suggested Refractoryperiod:", reverb_window)

    if command[0] == "params":
        for p in eng.network.params:
            print(f"{p} {eng.network.params[p]}")

    if command[0] == "stream":
        print(" streaming test dataset as input - %s" % command[1])
        stream(command[1])

    if command[0] == "csvstream_traced":
        print(" streaming csv dataset as input - %s" % command[1])
        csvstream(command[1], command[2], True, command[3])

    if command[0] == "csvstream":
        eng.network.check_params()
        epoch = int(input('Epoch: '))
        print(" streaming csv dataset as input - %s" % (command[1]))
        csvstream(command[1], command[2], False, command[3], epoch)

    if command[0] in ["tracepaths", "trace", "traces", "paths", "path"]:
        limits = float(command[1])
        inp = command[2].split(",")
        print(f" NSCL.trace(limits={limits})")
        print(inp)
        pp.pprint(npredict.trace_paths(eng, inp, limits, verbose=True))

    if command[0] == "active":
        active = eng.get_actives()
        print(active)

    if command[0] == "backtrace":
        propslvl = eng.network.params["PropagationLevels"]  # float(command[1])
        neurone = command[1]
        print(f" NSCL.back_trace()")
        print("propslvl", propslvl)
        print("composite", neurone)
        pp.pprint(npredict.back_trace(propslvl, neurone))

    if command[0] == "sinfer_old":
        print("sinfer")

        # data = open(command[1]).readlines()
        # meta = open(command[2]).readlines()

        # c1 = input()

        file_data = f"./dataset/dataset_sin_{command[1]}.csv"
        file_meta = f"./dataset/meta_sin_{command[1]}.csv"

        data = load_data(file_data, file_meta)

        reset_potentials = True if input(
            "reset potentials (y/n): ") == 'y' else False
        first = int(input("first row in file (stream from): "))
        last = int(input("last row in file (stream to): "))
        pticks = int(input("post ticks after stream: "))
        pthres = float(input("potential threshold: "))

        ticks = [x for x in range(first, last)]

        # for i in range(first, last):
        #     print(i, data[i])

        input(f"\nStream Length: {len(data)}  \nStream Range: {ticks}\n")

        tscores, htraces, datainputs = feed_trace(deepcopy(eng), first, data, ticks=ticks, p_ticks=pticks,
                                                  pot_threshold=pthres, reset_potentials=reset_potentials)

        print("Params")
        pp.pprint(eng.network.params)

        print("\nInputs")
        pp.pprint(datainputs)

        print("\nTScores")
        pp.pprint(tscores)

        print("\nHTraces")
        pp.pprint(htraces)

        save = input("save results in filename (nosave, leave empty): ")

        if save != "":
            jsondump("feedtraces", f"{save}.json", {
                     "params": eng.network.params, "meta": eng.meta, "datainputs": datainputs, "tscores_sum": tscores, "htraces": htraces})

    if command[0] == "sinfer_old2":
        print("sinfer2")

        # data = open(command[1]).readlines()
        # meta = open(command[2]).readlines()

        # c1 = input()

        file_data = f"./dataset/dataset_{command[1]}_{command[2]}.csv"
        file_meta = f"./dataset/meta_{command[1]}_{command[2]}.csv"

        data = load_data(file_data, file_meta)

        reset_potentials = True if input(
            "reset potentials (y/n): ") == 'y' else False
        first = int(input("first row in file (stream from): "))
        last = int(input("last row in file (stream to): "))
        pticks = int(input("post ticks after stream: "))

        ticks = [x for x in range(first, last)]

        # for i in range(first, last):
        #     print(i, data[i])

        input(f"\nStream Length: {len(data)}  \nStream Range: {ticks}\n")

        tscores, datainputs = feed_trace(deepcopy(
            eng), first, data, ticks=ticks, p_ticks=pticks, reset_potentials=reset_potentials)

        print("Params")
        pp.pprint(eng.network.params)

        print("\nInputs")
        pp.pprint(datainputs)

        print("\nPScores")
        pp.pprint(tscores)

        save = input("\nsave results in filename (nosave, leave empty): ")

        if save != "":
            jsondump("feedtraces2", f"{save}.json", {
                     "params": eng.network.params, "meta": eng.meta, "datainputs": datainputs, "tscores_prod": tscores})

    if command[0] == "sinfer":
        print("sinfer2")

        file_data = f"./dataset/dataset_{command[1]}_{command[2]}.csv"
        file_meta = f"./dataset/meta_{command[1]}_{command[2]}.csv"

        alldata = load_data(file_data, file_meta)

        reset_potentials = True if input(
            "reset potentials (y/n): ") == 'y' else False
        first = int(input("first row in file (stream from): "))
        last = int(input("last row in file (stream to): "))
        pticks = int(input("post ticks after stream: "))

        data = {}

        for t in range(first, last):
            data[t] = deepcopy(alldata[t])

        del alldata

        priories = deepcopy(data)

        pscores, datainputs, tick = npredict.feed_trace_unfold(deepcopy(eng), data, pticks, reset_potentials,
                          propagation_count=20, refractoryperiod=2, divergence=3)

        print("Params")
        pp.pprint(eng.network.params)

        print("\nInputs")
        pp.pprint(priories)

        print("\nPScores")
        pp.pprint(pscores)

        save = input("\nsave results in filename (nosave, leave empty): ")

        if save != "":
            jsondump("feedtraces2", f"{save}.json", {
                     "params": eng.network.params, "meta": eng.meta, "datainputs": datainputs, "pscores_prod": pscores})


    if command[0] == "sinferm":
        print("sinfer2m")

        for i in range(int(command[4]), int(command[5])):

            file_data = f"./dataset/dataset_{command[1]}_{command[2]}.csv"
            file_meta = f"./dataset/meta_{command[1]}_{command[2]}.csv"

            data = load_data(file_data, file_meta)

            reset_potentials = True
            # first = int(input("first row in file (stream from): "))
            # last = int(input("last row in file (stream to): "))
            # pticks = int(input("post ticks after stream: "))

            first = int(command[4])
            last = int(command[5])
            pticks = 10

            ticks = [x for x in range(first, i)]

            # for i in range(first, last):
            #     print(i, data[i])

            # input(f"\nStream Length: {len(data)}  \nStream Range: {ticks}\n")

            tscores, datainputs = feed_trace(deepcopy(
                eng), first, data, ticks=ticks, p_ticks=pticks, reset_potentials=reset_potentials)

            print("Params")
            pp.pprint(eng.network.params)

            print("\nInputs")
            pp.pprint(datainputs)

            print("\nPScores")
            pp.pprint(tscores)

            save = f"{command[3]}_{command[4]}_{i}"

            if save != "":
                jsondump("feedtraces2", f"{save}.json", {
                    "params": eng.network.params, "meta": eng.meta, "datainputs": datainputs, "tscores_prod": tscores})

            # data = data[first:last]
            print('done')

    if command[0] == "feed":
        feed = [x for x in command[1].split(",") if x != ""]
        eng.algo(feed)

    if command[0] == "prune":
        ncount = len(eng.network.neurones)
        scount = len(eng.network.synapses)
        npcount = len(eng.npruned)
        spcount = len(eng.spruned)
        print(f" NSCL.prune()")
        eng.prune_network()
        print(f" ncount {ncount} -> {len(eng.network.neurones)}")
        print(f" scount {scount} -> {len(eng.network.synapses)}")
        print(f" npcount {npcount} -> {len(eng.npruned)}")
        print(f" spcount {spcount} -> {len(eng.spruned)}")

    if command[0] in ["potsyn", "ptsyn", "struct", "network", "ls"]:
        eng.potsyns()
        print()

    if command[0] == "new":
        confirm = input
        if input(" new network? (y/n)") == "y":
            del eng
            eng = NSCL.Engine()
            print("new net")

    if command[0] == "graphout":
        print(" exporting graphs")
        graphout(eng)

    if command[0] == "save":
        # if len(command) > 1:
        print(f" Savestate({command[1]})")
        eng.save_state(command[1])
        # else:
        #     eng.save_state(eng.network.)

    if command[0] == "load":
        print(f" Loadstate({command[1]})")
        del eng
        eng = NSCL.Engine()
        print(f" memsize={eng.load_state(command[1])}")

    if command[0] == "memsize":
        print(f" memsize={eng.size_stat()}")
        # print(eng.size_stat())

    if command[0] == "avg_wgt_r":
        print(f" neurone {command[1]} = {eng.network.avg_wgt_r(command[1])}")

    if command[0] == "avg_wgt_f":
        print(f" neurone {command[1]} = {eng.network.avg_wgt_f(command[1])}")

    if command[0] == "info":
        clear()

        print()
        print(" ########################### ")
        print(f"     NSCL_python ")
        print()
        print(f"tick = {eng.tick}")
        print(f"hashid = {eng.network.hash_id}")
        # print(f"progress = {(eng.tick - start) / (end - start) * 100 : .1f}%")
        print(f"neurones = {len(eng.network.neurones)}")
        print(f"synapses = {len(eng.network.synapses)}")
        print(f"bindings = {eng.network.params['BindingCount']}")
        print(f"PropagationLevels = {eng.network.params['PropagationLevels']}")
        print(f"npruned = {len(eng.npruned)}")
        print(f"spruned = {len(eng.spruned)}")
        print(f"prune_ctick = {eng.prune}")
        print()
        eng.network.check_params()
        print()
        pp.pprint(eng.network.params)
        print()
        print(" ########################### ")
        print()

    if command[0] in ["tick", "pass", "next"]:
        r, e = eng.algo([], {})
        print(" reinf %s " % r)
        print(" errs %s " % e)

    if command[0] == "test":
        del eng
        eng = NSCL.Engine()
        print(f" memsize={eng.load_state('DSB2L2_S10F10_DW2')}")
        testcode = NSCL.Test.CompName()
        pp.pprint(NSCL.Test.primeProductWeights(testcode.name, eng))

    if command[0] == "exit":
        sys.exit(0)
    # except Exception as e:
    #     print(str(e))


# cd /mnt/Data/Dropbox/PhD Stuff/Najiy/sourcecodes/nscl-python
