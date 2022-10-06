from copy import deepcopy
from datetime import date, datetime
from decimal import DivisionByZero
import math
import pprint as pp

from numpy import SHIFT_OVERFLOW
from nscl_algo import NSCLAlgo
import nscl
from collections import Counter
import os
import json


def mergeDictionary(dict_1, dict_2):
    dict_3 = {**dict_1, **dict_2}
    for key, value in dict_3.items():
        if key in dict_1 and key in dict_2:
            dict_3[key] = [value, dict_1[key]]
    return dict_3


class NSCLPredict:
    def feed_trace_unfold(eng, datainputs, pticks=10, reset_potentials=True, propagation_count=20, divergence=3, refractoryperiod=2):

        tick = min(list(datainputs.keys()))

        def feed_all(datainputs, tick=tick, pscores={}, propagation_count=propagation_count):

            # propagate & capture
            times = list(datainputs.keys())
            times.sort()
            templist = []

            for t in datainputs:
                if len(datainputs[t]) == 0:
                    r, e, a = eng.algo([])
                    for i in a:
                        templist.append(
                            (t, i, eng.network.neurones[i].potential))
                else:
                    r, e, a = eng.algo(datainputs[t])
                    for i in a:
                        templist.append(
                            (t, i, eng.network.neurones[i].potential))
                print(f'processing all, at {t}')
                tick += 1

            for t in range(propagation_count):
                r, e, a = eng.algo([])
                for i in a:
                    templist.append((tick, i, eng.network.neurones[i].potential))

            # collect pscores
            for i in templist:
                if 'CMP' not in i[1]:
                    if i[0] not in pscores:
                        pscores[i[0]] = {}
                    if i[1] not in pscores[i[0]]:
                        pscores[i[0]][i[1]] = round(i[2], 4)
                    else:
                        pscores[i[0]][i[1]] += round(i[2], 4)
                else:
                    s = NSCLPredict.primeProductWeights(i[1], eng)
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

            return pscores, datainputs, tick

        def feed_once(datainputs, tick, pscores={}, propagation_count=propagation_count, divergence=1, refractoryperiod=2):

            times = list(datainputs.keys())
            times.sort()
            templist = []

            print('##############################')
            print('\npscores before')
            pp.pprint(pscores)
            print('datainput before', datainputs, '\n')
            # print(templist)

            if len(datainputs[tick-1]) == 0:
                r, e, a = eng.algo([])
                for i in a:
                    templist.append(
                        (tick, i, eng.network.neurones[i].potential))
            else:
                r, e, a = eng.algo(datainputs[tick-1])
                for i in a:
                    templist.append(
                        (tick, i, eng.network.neurones[i].potential))
            print(f'processing once, at {tick}')
            # tick += 1

            for t in range(propagation_count):
                r, e, a = eng.algo([])
                for i in a:
                    templist.append((tick, i, eng.network.neurones[i].potential))

            # collect pscores
            for i in templist:
                if 'CMP' not in i[1]:
                    if i[0] not in pscores:
                        pscores[i[0]] = {}
                    if i[1] not in pscores[i[0]]:
                        pscores[i[0]][i[1]] = round(i[2], 4)
                    else:
                        pscores[i[0]][i[1]] += round(i[2], 4)
                else:
                    s = NSCLPredict.primeProductWeights(i[1], eng)
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

            ptimes = [x for x in pscores.keys() if x >= tick-1]
            ptimes.sort(reverse=True)

            # pscorewindow = {}
            pscorewindowunfied = []
            refracorywindow = [x for x in range(
                max(ptimes)-1, max(ptimes)-1-refractoryperiod, -1)]
            exclusioninputs = []

            for rw in refracorywindow:
                if rw in datainputs:
                    for i in datainputs[rw]:
                        exclusioninputs.append(i)

            print('\nexclusions', exclusioninputs)
            print('\npscores after')
            pp.pprint(pscores)

            for t in ptimes:
                # pscorewindow[t] = []
                for s in pscores[t]:
                    if s not in exclusioninputs:
                        # pscorewindow[t].append((s, pscores[t][s]))
                        pscorewindowunfied.append((s, pscores[t][s]))

            pscorewindowunfied.sort(key=lambda x: x[1], reverse=True)
            print('divergence', divergence)
            input()
            pscorewindowunfied = pscorewindowunfied[:divergence]
            # print('pscorewindow', pscorewindow)
            print('pscoreunified', pscorewindowunfied)
            # print('ptimes', ptimes)
            datainputs[max(ptimes)] = [x[0] for x in pscorewindowunfied]

            print('datainput times after', datainputs)
            input()

            tick += 1

            return pscores, datainputs, tick

        if reset_potentials:
            eng.reset_potentials()

        pscores, datainputs, tick = feed_all(
            datainputs, pscores={}, propagation_count=propagation_count)
        # pp.pprint(datainputs)
        # pp.pprint(pscores)

        for i in range(len(datainputs), pticks):
            try:
                _pscores, _datainputs, _tick = feed_once(
                    datainputs, tick, pscores, propagation_count=propagation_count, divergence=divergence, refractoryperiod=refractoryperiod)
                pscores, datainputs, tick = (_pscores, _datainputs, _tick)
            except:
                break

        return pscores, datainputs, tick


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
            coll = dict((k, v)
                        for k, v in coll.items() if k is not None and k != '"')
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
            cluster = [{k: v.potential} for k, v in cp_eng.network.neurones.items(
            ) if v.potential > cp_eng.network.params['FiringThreshold']]
            res.append(cluster)

        return res

    def traceProductWeights(name, eng,  primes={}, acc=[], level=0):
        neurones = eng.network.neurones
        synapses = eng.network.synapses

        primes = {}
        levels = {}

        # for i in neurones.values():
        #     print(i.name)

        for rsyn in neurones[name].rsynapses:
            mapping = f'{rsyn}->{name}'
            wgt = eng.network.synapses[mapping].wgt
            tacc = deepcopy(acc)
            tacc.append(wgt)
            primes[mapping] = tacc
            levels[mapping] = level-1

            nprimes, nlevels = NSCLPredict.traceProductWeights(
                rsyn, eng, primes, tacc, level-1)

            primes = mergeDictionary(primes, nprimes)
            levels = mergeDictionary(levels, nlevels)

        return primes, levels


    def flatten(list_of_lists):
        if len(list_of_lists) == 0:
            return list_of_lists
        if isinstance(list_of_lists[0], list):
            return NSCLPredict.flatten(list_of_lists[0]) + NSCLPredict.flatten(list_of_lists[1:])
        return list_of_lists[:1] + NSCLPredict.flatten(list_of_lists[1:])


    def myprod(list1):
        try:
            return math.prod(list1)
        except:
            return math.prod(NSCLPredict.flatten(list1))

    def primeProductWeights(name, eng, ratio=True):
        traceWeights, traceLevels = NSCLPredict.traceProductWeights(name, eng)
        primeProducts = [(x.split('->')[0], NSCLPredict.myprod(traceWeights[x]), traceLevels[x])
                         for x in traceWeights if 'CMP' not in x.split('->')[0]]
        result = {}

        sumv = 0

        for p in primeProducts:
            result[p[0]] = (p[1], p[2])
            sumv += p[1]

        if ratio:
            for r in result:
                try:
                    result[r] = (result[r][0] / sumv, result[r][1])
                except ZeroDivisionError:
                    result[r] = (0, result[r][1])

        return result
