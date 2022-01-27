from numpy import string_
import nscl
import math
import random
import secrets
import copy
from datetime import datetime


class NSCLAlgo:
    def expo_decay(t) -> float:
        return math.exp(-t / 2)

    # def chunks(lst, n):
    #     for i in range(0, len(lst), n):
    #         yield lst[i : i + n]

    def chunks(lst, n):
        return [sorted(lst[i : i + n]) for i in range(0, len(lst), n)]

    # def expo_decay_2(t):
    #     return math.exp(-t / 4)

    # def expo_decay_3(t):
    #     return math.exp(-t / 4)

    def sname(pre, post):
        return "%s->%s" % (pre, post)

    def new_NSymbol(eng, name, lastspike="", potential=1.0) -> object:
        n = nscl.NSCL.NSymbol(name)
        eng.network.neurones[name] = n
        n.lastspike = lastspike
        n.potential = potential
        return n

    # returns 1 for success, 0 for error, -1 for existed (should be reinforced)
    def new_ssynapse(
        eng, pre_NSymbol, post_NSymbol, wgt=0.01, counter=0, lastspike=""
    ) -> str:
        neurones = eng.network.neurones
        synapses = eng.network.synapses

        pre = neurones[pre_NSymbol] if pre_NSymbol in neurones.keys() else None
        post = neurones[post_NSymbol] if post_NSymbol in neurones.keys() else None

        if not pre or not post or pre_NSymbol == post_NSymbol:
            print("new synapse error (%s->%s) " % (pre_NSymbol, post_NSymbol))
            return "error"

        sname = NSCLAlgo.sname(pre_NSymbol, post_NSymbol)

        if sname in synapses:
            return "reinforce"

        syn = nscl.NSCL.SSynapse(pre_NSymbol, post_NSymbol, wgt, counter, lastspike)
        eng.network.synapses[syn.name()] = syn
        if post_NSymbol not in pre.fsynapses:
            pre.fsynapses.append(post_NSymbol)
        if pre_NSymbol not in post.rsynapses:
            post.rsynapses.append(pre_NSymbol)
        return "created"

    def relevel(eng) -> None:
        # print(" Relevel()")

        neurones = eng.network.neurones
        synapses = eng.network.synapses

        # initialised all to level -1
        for n in neurones:
            neurones[n].level = -1
            neurones[n].heirarcs = []

        # inputs to level 0
        for n in neurones:
            if neurones[n].rsynapses == []:
                neurones[n].level = 0
            for s in neurones[n].fsynapses:
                neurones[s].heirarcs.append(1)

        # set other levels
        for i in range(1, eng.network.params["Levels"]):
            for n in neurones:
                if len(neurones[n].heirarcs) > 0 and neurones[n].level == -1:
                    neurones[n].level = max(neurones[n].heirarcs)
                    for s in neurones[n].fsynapses:
                        neurones[s].heirarcs.append(neurones[n].level + 1)

        # eng.network.neurones.sort(lambda n: n.level, reverse = True)

    def structural_plasticity(eng, time) -> list:
        # print(" StructuralPlasticity()")

        synapses = eng.network.synapses
        neurones = eng.network.neurones
        inp_neurones = eng.ineurones()

        active = NSCLAlgo.chunks(
            [
                n
                for n in neurones
                if neurones[n].potential > 0.5
                and neurones[n].level < eng.network.params["Levels"]  # G2
            ],
            eng.network.params["Bindings"],  # G4
        )

        reinforce_synapse = []  # enchancement list for if neurones already exists
        neurones_down_potentials = []

        if len(active) > 1:
            for a_set in active:
                if len(a_set) > 1:
                    # t = secrets.token_hex(nbytes=16)
                    post_new = f"CMP({','.join(a_set)})"
                    if post_new not in neurones.keys():
                        n = NSCLAlgo.new_NSymbol(eng, name=post_new, lastspike=time)
                    for pre_active in a_set:
                        if neurones[pre_active].level == eng.network.params["Levels"]:
                            continue
                        r = NSCLAlgo.new_ssynapse(eng, pre_active, post_new)
                        if r == "reinforce":
                            reinforce_synapse.append((pre_active, post_new))
                        # neurones_down_potentials.append(pre_active)
                        # neurones[pre_active].potential *= 0.3  # G3

        return (reinforce_synapse, neurones_down_potentials)

    def functional_plasticity(eng, reinforces, time) -> None:
        # print(" FunctionalPlasticity()")

        synapses = eng.network.synapses
        reinforce_synapse = []

        for s in reinforces:
            sname = NSCLAlgo.sname(s[0], s[1])
            wgt = synapses[sname].wgt
            wgt += (1 - wgt) * 0.3  # G5
            synapses[sname].wgt = wgt
            reinforce_synapse.append(f"reinforcing  {sname} {wgt: .4f}")

        return reinforce_synapse

    def algo1(eng, inputs, verbose=True, now=None, prune=False) -> list:
        # print("Algo {")
        # if now == None:
        #     now = datetime.now().isoformat()
        synapses = eng.network.synapses
        neurones = eng.network.neurones
        inp_neurones = eng.ineurones()

        gen_nsymbol = []

        # generate new input neurones
        # # GenerateInputNeurones() & FeedInputs()
        # print(" GenerateInputNeurones() & FeedInputs()")

        for i in inputs:
            if i not in neurones.keys():
                n = NSCLAlgo.new_NSymbol(eng, i)
                n.potential = 0.2
                # neurones[i].occurs += 1
                n.lastspike = eng.tick  # now
                gen_nsymbol.append(gen_nsymbol)
            elif i in inp_neurones:
                neurones[i].potential = 1.0
                # neurones[i].counter = 31
                neurones[i].occurs += 1
                neurones[i].lastspike = eng.tick  # now

        NSCLAlgo.relevel(eng)

        # Propagate() - calculates action potentials
        ns = [n for n in neurones.values()]
        ns.sort(reverse=True, key=lambda n: n.level)
        # print(" Propagate()")

        for n in ns:
            for l in range(eng.network.params["Levels"]):  # G6
                if n.level == l:
                    if n.potential > 0.8:
                        n.potential = 1.0
                        n.occurs += 1
                        for s in n.fsynapses:
                            ## forwards potentials
                            neurones[s].potential += (
                                n.potential * synapses[NSCLAlgo.sname(n.name, s)].wgt
                            )
                            # ADD TO RE-INFORCE LIST if necessary
                            ## potential decay (leaky for existing neurones)

                    if n.potential < 0.01:
                        n.potential = 0.0
                    else:
                        n.potential *= (  # G1
                            0.7
                            # if len(n.fsynapses) == 0
                            # else 0.5 + 0.4 / len(n.fsynapses)
                        )

        # generate neurones and synapses based on tau (spike-time differences)
        # GenerateNeurones() & GenerateSynapses()
        NSCLAlgo.relevel(eng)
        (rsynapse, downpotentials) = NSCLAlgo.structural_plasticity(
            eng, time=eng.tick
        )
        NSCLAlgo.relevel(eng)

        # for d in downpotentials:
        #     neurones[d].potential *= 0.9  # G3 (leaky for new neurones)

        # ReinforceSynapses()
        (reinforce_synapse) = NSCLAlgo.functional_plasticity(eng, rsynapse, eng.tick)
        # NSCLAlgo.relevel(eng)

        # print("}")

        if prune:
            for n in neurones:
                if n.occurs == 1:
                    eng.remove_neurone(n.name)

        return {
            "trace1": [neurones[n].potential for n in neurones],
            "rsynapse": reinforce_synapse,
            # "new_nsymbol": ,
            # "new_syn": ,
        }
