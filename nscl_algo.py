import nscl
import math
import random
import secrets
import copy
from datetime import datetime


class NSCLAlgo:
    def expo_decay(t) -> float:
        return math.exp(-t / 2)

    # def expo_decay_2(t):
    #     return math.exp(-t / 4)

    # def expo_decay_3(t):
    #     return math.exp(-t / 4)

    def sname(pre, post):
        return "%s->%s" % (pre, post)

    def new_sneurone(eng, name, lastspike="") -> object:
        n = nscl.NSCL.SNeurone(name)
        eng.network.neurones[name] = n
        n.lastspike = lastspike
        return n

    # returns 1 for success, 0 for error, -1 for existed (should be reinforced)
    def new_ssynapse(eng, pre_sneurone, post_sneurone, wgt = 0.01, counter = 0, lastspike='') -> str:
        neurones = eng.network.neurones
        synapses = eng.network.synapses

        pre = neurones[pre_sneurone] if pre_sneurone in neurones.keys() else None
        post = neurones[post_sneurone] if post_sneurone in neurones.keys() else None

        if not pre or not post or pre_sneurone == post_sneurone:
            print("new synapse error (%s->%s) " % (pre_sneurone, post_sneurone))
            return "error"

        sname = NSCLAlgo.sname(pre_sneurone, post_sneurone)

        if sname in synapses:
            return "reinforce"

        syn = nscl.NSCL.SSynapse(pre_sneurone, post_sneurone, wgt, counter, lastspike)
        eng.network.synapses[syn.name()] = syn
        if post_sneurone not in pre.fsynapses:
            pre.fsynapses.append(post_sneurone)
        if pre_sneurone not in post.rsynapses:
            post.rsynapses.append(pre_sneurone)
        return "created"

    def relevel(eng, maxi=6) -> None:
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
        for i in range(1, maxi):
            for n in neurones:
                if len(neurones[n].heirarcs) > 0 and neurones[n].level == -1:
                    neurones[n].level = max(neurones[n].heirarcs)
                    for s in neurones[n].fsynapses:
                        neurones[s].heirarcs.append(neurones[n].level + 1)

        # eng.network.neurones.sort(lambda n: n.level, reverse = True)

    def structural_plasticity(eng, time, maxi=6) -> list:
        synapses = eng.network.synapses
        neurones = eng.network.neurones
        inp_neurones = eng.ineurones()

        active = [n for n in neurones if neurones[n].potential > 0.5 and neurones[n].level < maxi]  # G2

        reinforce_synapse = []  # enchancement list for if neurones already exists
        neurones_down_potentials = []

        if len(active) > 1:
            # t = secrets.token_hex(nbytes=16)
            post_new = f"CMP({','.join(active)})"
            if post_new not in neurones.keys():
                n = NSCLAlgo.new_sneurone(eng, name=post_new, lastspike=time)
            for pre_active in active:
                if neurones[pre_active].level == maxi:
                    continue
                r = NSCLAlgo.new_ssynapse(eng, pre_active, post_new)
                if r == "reinforce":
                    reinforce_synapse.append((pre_active, post_new))
                neurones_down_potentials.append(pre_active)
                # neurones[pre_active].potential *= 0.3  # G3

        return (reinforce_synapse, neurones_down_potentials)

    def functional_plasticity(eng, reinforces, time) -> None:
        synapses = eng.network.synapses

        for s in reinforces:
            sname = NSCLAlgo.sname(s[0], s[1])
            wgt = synapses[sname].wgt
            wgt += (1 - wgt) * 0.3  # G5
            synapses[sname].wgt = wgt
            print(f"  reinforcing  {wgt: .4f} {sname}")

    def algo1(eng, inputs, maxi=6) -> list:
        now = datetime.now().isoformat()
        synapses = eng.network.synapses
        neurones = eng.network.neurones
        inp_neurones = eng.ineurones()

        # generate new input neurones
        # # GenerateInputNeurones() & FeedInputs()
        print("GenerateInputNeurones() & FeedInputs()")
        for i in inputs:
            if i not in neurones.keys():
                n = NSCLAlgo.new_sneurone(eng, i)
                n.potential += 1.0
                # neurones[i].occurs += 1
                n.lastspike = now
            elif i in inp_neurones:
                neurones[i].potential += 1.0
                # neurones[i].counter = 31
                neurones[i].occurs += 1
                neurones[i].lastspike = now

        NSCLAlgo.relevel(eng)

        # Propagate() - calculates action potentials
        ns = [n for n in neurones.values()]
        ns.sort(reverse=True, key=lambda n: n.level)
        print("Propagate()\n")

        for n in ns:
            print(
                f"  pot: {n.potential: .4f}  lvl: {n.level: ^4}  occ: {n.occurs}  fsyn: {len(n.fsynapses)}  rsyn: {len(n.rsynapses)}  {n.name:.20s}"
            )
            for l in range(maxi):  # G6
                if n.level == l:
                    for s in n.fsynapses:
                        neurones[s].potential += (
                            n.potential * synapses[NSCLAlgo.sname(n.name, s)].wgt
                        )
                        # ADD TO RE-INFORCE LIST
                    n.potential *= (
                        0.8
                        if len(n.fsynapses) == 0
                        else 0.5 + 1 / len(n.fsynapses) * 0.4  # G1
                    )

        # generate neurones and synapses based on tau (spike-time differences)
        # GenerateNeurones() & GenerateSynapses()
        NSCLAlgo.relevel(eng)
        (rsynapse, downpotentials) = NSCLAlgo.structural_plasticity(eng, time=now, maxi=maxi)
        NSCLAlgo.relevel(eng)

        for d in downpotentials:
            neurones[d].potential *= 0.8  # G3

        print()

        # ReinforceSynapses()
        NSCLAlgo.functional_plasticity(eng, rsynapse, now)
        # NSCLAlgo.relevel(eng)




        return {
            "trace1": [neurones[n].potential for n in neurones],
            "rsyanpse": rsynapse,
        }
