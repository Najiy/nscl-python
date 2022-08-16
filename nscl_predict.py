from copy import deepcopy
from datetime import date, datetime
import math

from numpy import SHIFT_OVERFLOW
from nscl_algo import NSCLAlgo
import nscl
from collections import Counter
import os
import json


class NSCLPredict:
    def save_predictions(fname, content):

        str(datetime.now().replace(microsecond=0)).replace(":", "_")
        # tt = str(datetime.now().replace(microsecond=0)).replace(":", "_")

        rpath = r"predictions\%s" % fname

        if not os.path.exists("predictions"):
            os.mkdir("predictions")

        if not os.path.exists(rpath):
            os.mkdir(rpath)

        outfile = open(rpath + "\\" + +".json", "w+")
        json.dump(content, outfile, indent=4)

        # outfile = open((rpath + "\\traces.json"), "w+")
        # json.dump(otraces, outfile, indent=4)

    def trace_paths(eng, inputs, limits=0, verbose=True):
        neurones = eng.network.neurones
        synapses = eng.network.synapses
        # inp_neurones = eng.ineurones()

        temp = {}

        def trace(n, level, limits_thr=0):
            # if verbose == True:
            #     input(neurones[n].rsynapses)

            for i, r in enumerate(neurones[n].rsynapses):
                syn_name = NSCLAlgo.sname(r, n)

                # synaptic threshold limiter
                if synapses[syn_name].wgt < limits_thr:
                    continue

                if verbose == True:
                    print(
                        " %s  %s <- %s (wgt %f, lvl %s)"
                        % (i, r, n, synapses[syn_name].wgt, level)
                    )

                if len(neurones[r].rsynapses) == 0:
                    rkey = f"iL{level}"
                    if rkey not in temp.keys():
                        temp[rkey] = []
                    else:
                        temp[rkey].append(r)

        def forward(n, level=0, limits_thr=0):
            for f in neurones[n].fsynapses:
                syn_name = NSCLAlgo.sname(n, f)

                # synaptic threshold limiter
                if synapses[syn_name].wgt < limits_thr:
                    continue

                rkey = f"cL{level}"

                if rkey not in temp.keys():
                    temp[rkey] = []
                else:
                    temp[rkey].append(n)

                if verbose == True:
                    print(
                        "%s -> %s (wgt %f, lvl %s)"
                        % (n, f, synapses[syn_name].wgt, level)
                    )

                trace(f, level, limits_thr=limits)
                forward(f, level=level + 1, limits_thr=limits)

        for n in inputs:
            forward(n)

        result = {"inputs": inputs}

        for e in temp:
            result[e] = dict(Counter(temp[e]))

        result["prediction_time"] = str(datetime.now().replace(microsecond=0)).replace(
            ":", "_"
        )

        return result

    # def static_prediction(eng, inputs, limits=0, verbose=True):
    #     result = NSCLPredict.trace_paths(eng, inputs, limits, verbose)
    #     return result

    # def temporal_prediction(eng, inputs, limits=0, verbose=True):
    #     result = NSCLPredict.trace_paths(eng, inputs, limits, verbose)
    #     return result

    def back_trace(propsLvl, neurone) -> object:
        struct = ""

        if type(neurone) is str:
            struct = neurone
        else:
            struct = neurone.name

        struct = struct.replace("CMP(", "#(#")
        struct = struct.replace(")", "#)#")
        struct = struct.replace(",", "#")
        struct = [x for x in struct.split("#") if x != ""]
        infers = [[] for x in range(propsLvl + 1)]

        lvl = 0
        for i, v in enumerate(struct):
            if struct[i] == "(":
                lvl += 1
            elif struct[i] == ")":
                lvl -= 1
            else:
                infers[lvl].append(struct[i])
        infers.reverse()

        while len(infers[0]) == 0:
            infers.pop(0)
            infers.append([])
        counts = []

        for i, v in enumerate(infers):
            coll = dict(Counter(infers[i]))
            coll = dict((k, v) for k,v in coll.items() if k is not None and k != '"')
            maxx = 0

            try:
                maxx = max(coll.values())
            except:
                maxx = 0

            ncoll = {}
            # ncoll['"'] = 5.0
            # del ncoll['"']

            for k, v in coll.items():
                ncoll[k] = v / maxx
            
            counts.append(ncoll)

        infers = counts

        return infers


    def feed_forward_infer(eng, inputs, use_current_ctx=False) -> object:

        cp_eng = deepcopy(eng)
        cp_eng.set_algo(NSCLAlgo.algo1)

        res = []

        if not use_current_ctx:
            for nkey in cp_eng.network.neurones.keys():
                cp_eng.network.neurones[nkey].potential = 0.0

        for i in range(len(inputs)):
            eng.algo([inputs[i]])
            cluster = [{k:v.potential} for k, v in cp_eng.network.neurones.items() if v.potential > cp_eng.network.params['FiringThreshold']]
            res.append(cluster)
        
        return res