from datetime import date, datetime
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

            outfile = open(rpath + "\\"+  +".json", "w+")
            json.dump(content, outfile, indent=4)

            # outfile = open((rpath + "\\traces.json"), "w+")
            # json.dump(otraces, outfile, indent=4)


    def trace_synapses(eng, inputs, syn_limiter=0, verbose=True):
        neurones = eng.network.neurones
        synapses = eng.network.synapses
        # inp_neurones = eng.ineurones()

        temp = {}

        def trace(n, level, limits=0):
            if verbose == True:
                input(neurones[n].rsynapses)

            for i, r in enumerate(neurones[n].rsynapses):
                syn_name = NSCLAlgo.sname(r, n)

                if synapses[syn_name].wgt < limits:
                    continue

                if verbose == True:
                    print(
                        " %s  %s <- %s (wgt %f, lvl %s)"
                        % (i, r, n, synapses[syn_name].wgt, level)
                    )

                if len(neurones[r].rsynapses) == 0:
                    key = "%s" % level
                    if key not in temp.keys():
                        temp[key] = []
                    else:
                        temp[key].append(r)

        def forward(n, level=0, limits=0):
            for f in neurones[n].fsynapses:
                syn_name = NSCLAlgo.sname(n, f)

                if synapses[syn_name].wgt < limits:
                    continue

                if verbose == True:
                    print(
                        "%s -> %s (wgt %f, lvl %s)"
                        % (n, f, synapses[syn_name].wgt, level)
                    )

                # conditional limiter
                trace(f, level, limits=syn_limiter)
                # conditional limiter
                forward(f, level=level + 1, limits=syn_limiter)

        for n in inputs:
            forward(n)

        result = {}

        for e in temp:
            result[e] = dict(Counter(temp[e]))

        result['prediction_time'] =  str(datetime.now().replace(microsecond=0)).replace(":", "_")

        return result


    def static_prediction(eng, inputs, syn_limiter=0, verbose=True):
        result = NSCLPredict.trace_synapses(eng,inputs, syn_limiter, verbose)
        return result


    def temporal_prediction(eng, inputs, integrate=False):
        pass