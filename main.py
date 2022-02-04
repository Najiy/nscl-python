#!/usr/bin/python

from datetime import date, datetime
from inspect import trace

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
from jinja2.defaults import NEWLINE_SEQUENCE

from networkx.algorithms.planarity import Interval
from networkx.generators.geometric import random_geometric_graph
from pandas.core.algorithms import take
from nscl import NSCL
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import networkx as nx
from nscl_predict import NSCLPredict as npredict
from pyvis.network import Network

import itertools
from itertools import islice

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

    my_colors = [(0.2, 0.3, 0.3), (0.4, 0.5, 0.4), (0.1, 0.7, 0), (0.1, 0.7, 0)]

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
    df = pd.DataFrame({"time": eng.ntime, "ncounts": eng.ncounts, "ntype": eng.nmask})
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

    # npredict.forward_predict(eng, [])


def stream(streamfile, trace=True):

    # text = "Top Cat! The most effectual Top Cat! Who’s intellectual close friends get to call him T.C., providing it’s with dignity. Top Cat! The indisputable leader of the gang. He’s the boss, he’s a pip, he’s the championship. He’s the most tip top, Top Cat."
    # txt_arr = text.lower().split(' ')
    # for i,v in enumerate(txt_arr):
    #     inputs[i] = [v]
    # input(inputs)

    filecontent = json.loads(open(f"dataset{pathdiv}{streamfile}.json", "r").read())

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

            eng.algo(inputs[eng.tick], trace)

            if eng.tick == maxit:
                graphout(eng)

        except KeyboardInterrupt:
            running = False

    eng.tick += temp

    print("\n\n streaming done.")
    print()


def csvstream(streamfile, trace=False, fname="default"):
    def take(n, iterable):
        # "Return first n items of the iterable as a list"
        return list(islice(iterable, n))

    data = {}

    file = open(streamfile, "r")
    sensors = file.readline().split(",")[1:]
    rawdata = file.readlines()

    for line in rawdata:
        sline = line.replace("\n", "").split(",")
        for i in range(1, 1 + len(sensors)):
            if sline[i] != "":
                if (
                    float(sline[i]) != 0.0
                    and float(sline[i]) != 1.0
                    and float(sline[i]) != -1.0
                ):
                    sline[i] = ""
                else:
                    sline[i] = f"{i}~{sline[i]}"
        data[int(sline[0])] = [x for x in sline[1:] if x != ""]

    start = int(rawdata[0].split(",")[0])
    end = int(rawdata[-1].split(",")[0])

    del rawdata

    print(start, end)
    # print(take(3, data.items()))
    input("loaded, now processing [enter] ")

    temp = eng.tick
    eng.tick = start
    maxit = end

    saveit = 0
    saveat = 604800

    # maxit = min(len(inputs) - 1, interv)
    running = True

    starttime = datetime.now().isoformat(timespec="minutes")

    while running and eng.tick <= maxit:
        try:
            if eng.prune == 0:
                eng.prune = eng.params["PruneInterval"][0]
            eng.prune -= 1
            saveit += 1

            if eng.tick % 5000 == 0 or eng.tick == maxit or eng.prune == 0:
                clear()

                print()
                print(" ###########################")
                print(f"     NSCL_python \n time = {starttime} ")
                print(f"tick = {eng.tick}")
                print(f"progress = {(eng.tick - start) / (end - start) * 100 : .1f}%")
                print(f"neurones = {len(eng.network.neurones)}")
                print(f"synapses = {len(eng.network.synapses)}")
                print(f"bindings = {eng.defparams['Bindings']}")
                print(f"levels = {eng.defparams['Levels']}")
                print(f"npruned = {len(eng.npruned)}")
                print(f"spruned = {len(eng.spruned)}")
                print(f"prune_ctick = {eng.prune}")
                print(" ###########################")
                print()

            if eng.tick not in data:
                eng.algo([], trace, now=eng.tick, prune=eng.prune == 0)
            else:
                eng.algo(data[eng.tick], trace, now=eng.tick, prune=eng.prune == 0)

            if eng.tick == maxit and trace == True:
                graphout(eng)

            if saveit == saveat or eng.tick == maxit:
                saveit = 0
                # os.mkdir(f'state{pathdiv}{fname}')
                # eng.save_state(f'{fname}{pathdiv}{eng.tick}')
                eng.save_state(f"{fname}_{eng.tick}")

            if eng.tick == maxit:
                input("csv stream done! [enter]")

        except KeyboardInterrupt:
            running = False

    eng.tick += temp

    print("\n\n streaming done.")
    print()


# def loading():
#     print("Loading...")
#     for i in range(0, 100):
#         time.sleep(0.1)
#         width = (i + 1) / 4
#         bar = "[" + "#" * width + " " * (25 - width) + "]"
#         sys.stdout.write(u"\u001b[1000D" +  bar)
#         sys.stdout.flush()
#     print("")


args = sys.argv[1:]

# print(" ".join(args))

pp = pprint.PrettyPrinter(indent=4)

init = False
verbose = False
while True:
    if init == False:
        print("\n\n########### NSCL (Python) ###########\n")
        print(f" version: experimental/non-optimised")
        print(f" os.name: {os.name}")
        print(f" os.pid: {os.getpid()}")
        print("")

        # subprocess.call(f'top -p {os.getpid()}', shell=True)
        # os.system(f"top -p {os.getpid()}")
        # Popen('bash')

    try:
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

        if command[0] == "stream":
            print(" streaming test dataset as input - %s" % command[1])
            stream(command[1])

        if command[0] == "csvstream_traced":
            print(" streaming csv dataset as input - %s" % command[1])
            csvstream(command[1], True, command[2])

        if command[0] == "csvstream":
            print(" streaming csv dataset as input - %s" % command[1])
            csvstream(command[1], False, command[2])

        if command[0] in ["tracepaths", "trace", "traces", "paths", "path"]:
            limits = float(command[1])
            inp = command[2].split(",")
            print(f" NSCL.trace(limits={limits})")
            print(inp)
            pp.pprint(npredict.trace_paths(eng, inp, limits, verbose=True))

        if command[0] == "spredict":
            limits = float(command[1])
            inp = command[2].split(",")
            print(f" NSCL.static_predict(limits={limits})")
            print(inp)
            pp.pprint(npredict.static_prediction(eng, inp, limits, verbose=False))

        if command[0] == "tpredict":
            limits = float(command[1])
            inp = command[2].split(",")
            print(f" NSCL.temporal_predict(limits={limits})")
            pp.pprint(npredict.temporal_prediction(eng, inp, limits, verbose=False))

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
            print(f" Savestate({command[1]})")
            eng.save_state(command[1])

        if command[0] == "load":
            print(f" Loadstate({command[1]})")
            del eng
            eng = NSCL.Engine()
            print(f" memsize={eng.load_state(command[1])}")

        if command[0] == "memsize":
            print(f" memsize={eng.size_stat()}")
            # print(eng.size_stat())

        if command[0] == "info":
            clear()

            print()
            print(" ###########################")
            print(f"     NSCL_python ")
            print(f"tick = {eng.tick}")
            # print(f"progress = {(eng.tick - start) / (end - start) * 100 : .1f}%")
            print(f"neurones = {len(eng.network.neurones)}")
            print(f"synapses = {len(eng.network.synapses)}")
            print(f"bindings = {eng.defparams['Bindings']}")
            print(f"levels = {eng.defparams['Levels']}")
            print(" ###########################")
            print()

        if command[0] in ["tick", "pass", "next"]:
            r = eng.algo([], False)
            print(" reinf %s " % r["rsynapse"])

        if command[0] == "exit":
            sys.exit(0)

    except Exception as e:
        print(e)
