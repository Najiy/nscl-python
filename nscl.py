# from main import networkx
# from asyncio.windows_events import NULL
# from curses.ascii import NUL
# from email import header
# from genericpath import getsize
# from inspect import trace

from pickle import NONE
from nscl_algo import NSCLAlgo
from datetime import date, datetime
import copy
import json
import os
import sys
import psutil
import hashlib

pathdiv = ""

if os.name == "posix":
    pathdiv = "/"
elif os.name == "nt":
    pathdiv = "\\"


class NSCL:
    class NSymbol:
        def __neuroneLvl(self, name):
            arr = list(filter(lambda item: item != "", name.replace(
                "CMP(", "#(#").replace(")", "#)#").replace(",", "#").split("#")))
            lvl = 0
            maxlvl = 0

            for i in arr:
                if i == "(":
                    lvl += 1
                    if maxlvl < lvl:
                        maxlvl = lvl
                elif i == ")":
                    lvl -= 1

            return maxlvl

        def heirarchies(self, name):
            arr = list(filter(lambda item: item != "", name.replace(
                "CMP(", "#(#").replace(")", "#)#").replace(",", "#").split("#")))
            lvl = 0
            maxlvl = 0
            heirarcs = {}

            for i in arr:
                if i == "(":
                    lvl -= 1
                    if maxlvl > lvl:
                        maxlvl = lvl
                elif i == ")":
                    lvl += 1
                else:
                    if lvl not in heirarcs.keys():
                        heirarcs[lvl] = []
                    heirarcs[lvl].append(i)

            return maxlvl, heirarcs

        def __init__(
            self, name, occurs=1, potential=0, refractory=0, lastspike="", probationary= -1
        ) -> None:
            self.name = name
            self.meta = {}
            # self.pot = [0 for i in range(0,NSCL.defparams['TrainLength'])]
            self.potential = potential
            # self.pot2 = 0
            # self.pot3 = 0
            self.fsynapses = []
            self.rsynapses = []
            self.lastspike = lastspike
            self.refractory = refractory
            self.occurs = occurs
            self.level, self.heirarcs = self.heirarchies(name)
            self.level = self.__neuroneLvl(name)
            self.probationary = probationary
            # print(f'generated {name}', end=' ')

        # def level(self):
        #     self.level = max(self.heirarcs)
        #     return self.level

    class SSynapse:
        def __init__(
            self, r_neurone, f_neurone, wgt=0.01, occurs=1, lastspike=""
        ) -> None:
            
            self.fref = f_neurone
            self.rref = r_neurone
            self.wgt = wgt
            self.occurs = occurs
            self.lastspike = lastspike
            
            # self.f1 = 0
            # self.f2 = 0
            # self.aug = 0
            # self.ptp = 0
            # self.ltp = 0

        def name(self):
            return "%s->%s" % (self.rref, self.fref)

    class Network:
        def __init__(self, params={}) -> None:
            self.ititialised = datetime.now().isoformat()
            self.hash_id = hashlib.md5(self.ititialised.encode()).hexdigest()
            self.neurones = {}
            self.synapses = {}
            infile = open(f"defparams.json", "r")
            cont = infile.read()
            cont = cont.replace("\t", "")
            cont = cont.replace("\n", "")
            defparams = json.loads(cont)
            self.params = {**defparams, **params}

        def net_struct(self) -> object:
            return {
                "params": self.params,
                "sneurones": self.neurones,
                "ssynapses": self.synapses,
            }

        def avg_wgt_r(self, neurone):
            synaptic_weights = [
                self.synapses[NSCLAlgo.sname(x, neurone)].wgt
                for x in self.neurones[neurone].rsynapses
            ]
            return sum(synaptic_weights) / len(synaptic_weights)

        def avg_wgt_f(self, neurone):
            synaptic_weights = [
                self.synapses[NSCLAlgo.sname(x, neurone)].wgt
                for x in self.neurones[neurone].fsynapses
            ]
            return sum(synaptic_weights) / len(synaptic_weights)

        def __repr__(self) -> str:
            return str(self.net_struct())

        def check_params(self, prompt_fix=  True) -> str:
            firing_threshold = self.params["FiringThreshold"]
            zeroing_threshold = self.params["ZeroingThreshold"]
            binding_threshold = self.params["BindingThreshold"]
            decay_factor = self.params["DecayFactor"]
            refractory_period = self.params["RefractoryPeriod"]

            potential = 1
            pots = []
            reverberating = True

            while (potential != 0):
                pots.append(potential)
                potential *= decay_factor
                if potential < zeroing_threshold:
                    potential = 0

            binding_window = len([x for x in pots if x >= binding_threshold])
            reverb_window = len([x for x in pots if x >= firing_threshold])

            # try:
            if pots[refractory_period] < firing_threshold:
                reverberating = False
            # except IndexError:
            #     reverberating = False

            if prompt_fix and reverberating:
                print("Binding Window =", binding_window)
                print("Current Refractoryperiod =", refractory_period)
                print("Reverberating Firing =", reverberating)
                print("Suggested Refractoryperiod:", reverb_window)
                print(
                    " WARNING: consider changing refractory period to avoid reverberating firing")
                if input(f"     change RefractoryPeriod to {reverb_window}? y/n") == 'y':
                    self.params['RefractoryPeriod'] = reverb_window

            return binding_window, refractory_period, reverberating, reverb_window

    class Engine:
        def __init__(self, network=None, meta={}, algorithm=None) -> None:
            self.network = NSCL.Network() if network == None else network
            self.npruned = {}
            self.spruned = {}
            self._algo = algorithm if algorithm else NSCLAlgo.algo1
            self.traces = []
            self.ntime = []
            self.ncounts = []
            self.nmask = []
            self.tick = 0
            self.meta = {**meta}
            self.prune = self.network.params["PruneInterval"]

        def set_algo(self, algo) -> None:
            self._algo = algo

        def params(self) -> object:
            return self.network.params

        def clear_traces(self) -> None:
            self.traces = []
            self.ntime = []
            self.ncounts = []
            self.nmask = []

        def params(self) -> dict:
            return self.network.params

        # def prune_network(self) -> None:
        #     nlist = [k for k in self.network.neurones.keys()]  # useless lol
        #     for n in nlist:
        #         if (
        #             self.network.neurones[n].occurs == 1
        #             and len(self.network.neurones[n].rsynapses) > 0
        #             and self.tick - self.network.neurones[n].lastspike
        #             > self.network.params["PruneInterval"]
        #             and self.network.avg_wgt_r(n)
        #             < self.network.params["PruningThreshold"]
        #         ):
        #             self.remove_neurone(n)

        def algo(self, inputs, meta={}) -> tuple:
            r,errors,activated = self._algo(self, inputs, meta)
            it = self.tick

            # if self.trace:
            #     self.traces.append(r["trace1"])
            #     self.ntime.append(it)
            #     self.ncounts.append(len(r["trace1"]))
            #     self.nmask.append("reinforced")

            #     self.ntime.append(it)
            #     self.ncounts.append(len(self.ineurones()))
            #     self.nmask.append("input")

            #     self.ntime.append(it)
            #     self.ncounts.append(len(self.gneurones()))
            #     self.nmask.append("composite")

            #     self.ntime.append(it)
            #     self.ncounts.append(len(self.neurones()))
            #     self.nmask.append("total")

            #     self.ntime.append(it)
            #     self.ncounts.append(len(self.synapses()))
            #     self.nmask.append("synapses")

            #     while len(self.traces) > self.network.params["TraceLength"]:
            #         self.traces.pop(0)
            #     while len(self.ntime) * 5 > self.network.params["TraceLength"] * 5:
            #         self.ntime.pop(0)
            #     while len(self.ncounts) * 5 > self.network.params["TraceLength"] * 5:
            #         self.ncounts.pop(0)
            #     while len(self.nmask) * 5 > self.network.params["TraceLength"] * 5:
            #         self.nmask.pop(0)

            self.tick += 1
            return r, errors, activated

        def neurones(self) -> dict:
            return self.network.neurones

        def gneurones(self) -> dict:
            return [n for n in self.network.neurones if n not in self.ineurones()]

        def ineurones(self) -> dict:
            neurones = self.neurones()
            return [n for n in neurones.keys() if neurones[n].rsynapses == []]

        def synapses(self) -> dict:
            return self.network.synapses

        def net_struct(self) -> object:
            return self.network.net_struct()

        def new_sneurone(self, name) -> object:
            return NSCLAlgo.new_NSymbol(self, name)

        def new_pruned_sneurone(self, name) -> object:
            return NSCLAlgo.new_pruned_NSymbol(self, name)

        def remove_neurone(self, name) -> object:
            # try:
            n = self.network.neurones[name]
            for f in n.fsynapses:
                nf = self.network.neurones[f]
                if name in nf.rsynapses:
                    sn = NSCLAlgo.sname(name, f)
                    self.spruned[sn] = copy.deepcopy(self.network.synapses[sn])
                    del self.network.synapses[sn]
                    nf.rsynapses.remove(name)
            for r in n.rsynapses:
                nr = self.network.neurones[r]
                if name in nr.fsynapses:
                    sn = NSCLAlgo.sname(r, name)
                    self.spruned[sn] = copy.deepcopy(self.network.synapses[sn])
                    del self.network.synapses[sn]
                    nr.fsynapses.remove(name)
            delneurone = copy.deepcopy(self.network.neurones[name])
            self.npruned[name] = delneurone
            del self.network.neurones[name]
            return delneurone
            
        def reset_potentials(self):
            for n in self.network.neurones:
                self.network.neurones[n].potential = 0

        def new_ssynapse(
            self, pre_sneurone, post_sneurone, wgt=0.01,  lastspike=""
        ) -> bool:
            NSCLAlgo.new_ssynapse(
                self, pre_sneurone, post_sneurone, wgt,  lastspike
            )

        def new_pruned_ssynapse(
            self, pre_sneurone, post_sneurone, wgt=0.01, lastspike=""
        ) -> bool:
            NSCLAlgo.new_pruned_ssynapse(
                self, pre_sneurone, post_sneurone, wgt, lastspike
            )

        def save_state(self, fname) -> None:
            network = self.network
            neurones = network.neurones
            synapses = network.synapses
            params = network.params
            npruned = self.npruned
            spruned = self.spruned

            content = {
                "neurones": [],
                "synapses": [],
                "npruned": [],
                "spruned": [],
                "params": params,
                "tick": self.tick,
            }
            otraces = {
                "traces": self.traces,
                "ntime": self.ntime,
                "ncounts": self.ncounts,
                "nmask": self.nmask,
            }

            for k in neurones:
                n = neurones[k]
                content["neurones"].append(
                    {
                        "name": n.name,
                        "meta": n.meta,
                        "potential": n.potential,
                        "fsynapses": n.fsynapses,
                        "rsynapses": n.rsynapses,
                        "lastspike": n.lastspike,
                        "refractory": n.refractory,
                        "occurs": n.occurs,
                        "level": n.level,
                        "probationary": n.probationary
                    }
                )

            for k in synapses:
                s = synapses[k]
                content["synapses"].append(
                    {
                        "fref": s.fref,
                        "rref": s.rref,
                        "wgt": s.wgt,
                        "occurs": s.occurs,
                        "lastspike": s.lastspike,
                    }
                )

            for k in npruned:
                n = npruned[k]
                content["npruned"].append(
                    {
                        "name": n.name,
                        "meta": n.meta,
                        "potential": n.potential,
                        "fsynapses": n.fsynapses,
                        "rsynapses": n.rsynapses,
                        "lastspike": n.lastspike,
                        "refractory": n.refractory,
                        "occurs": n.occurs,
                        "level": n.level,
                        "probationary": n.probationary
                    }
                )

            for k in spruned:
                s = spruned[k]
                content["spruned"].append(
                    {
                        "fref": s.fref,
                        "rref": s.rref,
                        "wgt": s.wgt,
                        "occurs": n.occurs,
                        "lastspike": s.lastspike,
                    }
                )

            # tt = str(datetime.now().replace(microsecond=0)).replace(":", "_")
            rpath = f"states{pathdiv}{fname}"

            if not os.path.exists("states"):
                os.mkdir("states")

            if not os.path.exists(rpath):
                os.mkdir(rpath)

            outfile = open(rpath + f"{pathdiv}state.json", "w+")
            json.dump(content, outfile, indent=4)

            outfile = open(rpath + f"{pathdiv}traces.json", "w+")
            json.dump(otraces, outfile, indent=4)

            timestamp = datetime.now().isoformat(timespec="minutes").replace("T", " ")
            # netmeta = open(rpath + f"{pathdiv}networks.meta", "a")

            netmetapath = f"states{pathdiv}networks.meta"
            headers = True
            if os.path.exists(netmetapath):
                headers = False
            netmeta = open(netmetapath, "a")

            if headers:
                netmeta.write(
                    "name,hashid,time,tick,neurones,synapses,npruned,spruned,inputs,composites\n")

            inputs = len(
                [
                    x
                    for x in self.network.neurones
                    if len(self.network.neurones[x].rsynapses) == 0
                ]
            )
            composites = len(
                [
                    x
                    for x in self.network.neurones
                    if len(self.network.neurones[x].rsynapses) > 0
                ]
            )
            netmeta.write(
                f"{fname},{self.network.hash_id},{timestamp},{self.tick},{len(self.network.neurones)},{len(self.network.synapses)},{len(self.npruned)},{len(self.spruned)},{inputs},{composites}\n"
            )
            netmeta.close()

        def load_state(self, fname) -> None:

            rpath = f"states{pathdiv}%s" % fname
            # input(rpath + f"{pathdiv}state.json")
            infile = open(rpath + f"{pathdiv}state.json", "r")
            cont = infile.read()
            cont = cont.replace("\t", "")
            cont = cont.replace("\n", "")
            load_state = json.loads(cont)

            infile = open(rpath + f"{pathdiv}traces.json", "r")
            cont = infile.read()
            cont = cont.replace("\t", "")
            cont = cont.replace("\n", "")
            load_traces = json.loads(cont)

            # self = NSCL.Engine()

            for nprop in load_state["neurones"]:
                self.new_sneurone(nprop["name"])

                n = self.network.neurones[nprop["name"]]
                try:
                    n.meta = nprop["meta"]
                except:
                    pass
                n.potential = nprop["potential"]
                n.fsynapses = nprop["fsynapses"]
                n.rsynapses = nprop["rsynapses"]
                n.lastspike = nprop["lastspike"]
                n.refractory = nprop["refractory"]
                n.occurs = nprop["occurs"]
                n.level = nprop["level"]
                try:
                    n.probationary = nprop["probationary"]
                except:
                    pass
                # n.heirarcs = nprop["heirarcs"]

            for sprop in load_state["synapses"]:
                pre = sprop["rref"]
                post = sprop["fref"]
                self.new_ssynapse(
                    pre, post, sprop["wgt"], sprop["lastspike"]
                )

            for nprop in load_state["npruned"]:
                self.new_pruned_sneurone(nprop["name"])
                n = self.npruned[nprop["name"]]
                try:
                    n.meta = nprop["meta"]
                except:
                    pass
                n.potential = nprop["potential"]
                n.fsynapses = nprop["fsynapses"]
                n.rsynapses = nprop["rsynapses"]
                n.lastspike = nprop["lastspike"]
                try:
                    n.refractory = nprop["refractory"]
                except:
                    pass
                n.occurs = nprop["occurs"]
                n.level = nprop["level"]
                try:
                    n.probationary = nprop["probationary"]
                except:
                    pass
                # n.heirarcs = nprop["heirarcs"]

            for sprop in load_state["spruned"]:
                pre = sprop["rref"]
                post = sprop["fref"]
                self.new_pruned_ssynapse(
                    pre, post, sprop["wgt"], sprop["lastspike"]
                )

            # self.npruned = load_state["npruned"]
            # self.spruned = load_state["spruned"]

            # s = eng.network.synapses[NSCLAlgo.sname(pre, post)]
            # self.network.neurones = content["neurones"]
            # self.network.synapses = content["synapses"]
            # self.network.params = content["params"]

            self.network.params = load_state["params"]

            self.network

            self.tick = load_state["tick"]
            self.traces = load_traces["traces"]
            self.ncounts = load_traces["ncounts"]
            self.nmask = load_traces["nmask"]
            self.ntime = load_traces["ntime"]

            # NSCLAlgo.relevel(self)

            return self.size_stat()

        def size_stat(self):
            def get_size(obj, seen=None):
                """Recursively finds size of objects"""
                size = sys.getsizeof(obj)
                if seen is None:
                    seen = set()
                obj_id = id(obj)
                if obj_id in seen:
                    return 0
                # Important mark as seen *before* entering recursion to gracefully handle
                # self-referential objects
                seen.add(obj_id)
                if isinstance(obj, dict):
                    size += sum([get_size(v, seen) for v in obj.values()])
                    size += sum([get_size(k, seen) for k in obj.keys()])
                elif hasattr(obj, "__dict__"):
                    size += get_size(obj.__dict__, seen)
                elif hasattr(obj, "__iter__") and not isinstance(
                    obj, (str, bytes, bytearray)
                ):
                    size += sum([get_size(i, seen) for i in obj])
                return size

            # r = "%d" % sys.getsizeof(self.network.neurones)
            n_size = "neurones=%db|%db" % (
                sys.getsizeof(self.network.neurones),
                get_size(self.network.neurones),
            )
            s_size = "synapses=%db|%db" % (
                sys.getsizeof(self.network.synapses),
                get_size(self.network.synapses),
            )
            np_size = "p_neurones=%db|%db" % (
                sys.getsizeof(self.npruned),
                get_size(self.npruned),
            )
            sp_size = "p_synapses=%db|%db" % (
                sys.getsizeof(self.spruned),
                get_size(self.spruned),
            )
            p_size = "params=%db|%db" % (
                sys.getsizeof(self.network.params),
                get_size(self.network.params),
            )
            traces = "traces=%db|%db" % (
                sys.getsizeof(self.traces),
                get_size(self.traces),
            )
            process_rss = f"process_rss={ int(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)}MB|{psutil.Process(os.getpid()).memory_info().rss}b"
            process_vms = f"process_vms={int(psutil.Process(os.getpid()).memory_info().vms / 1024 ** 2)}MB|{psutil.Process(os.getpid()).memory_info().vms}b"
            all = "all=%db|%db" % (sys.getsizeof(self), get_size(self))
            return (n_size, s_size, np_size, sp_size, p_size, traces, process_rss, process_vms, all)

        def set_tick(self, tick):
            self.tick = tick

        def potsyns(eng):
            now = str(datetime.now().replace(microsecond=0)).replace(":", "_")
            synapses = eng.network.synapses
            neurones = eng.network.neurones
            # inp_neurones = eng.ineurones()

            ns = [n for n in neurones.values()]
            ns.sort(reverse=True, key=lambda n: n.level)

            print(f"\nPotSyn() at {now}")
            print("Nsymbols Potentials:")

            for n in ns:
                print(
                    f"  {n.name:^20}   pot: {n.potential: .4f}   lvl: {n.level: ^4}   occ: {n.occurs}   fsyn: {len(n.fsynapses)}   rsyn: {len(n.rsynapses)}"
                )

            syn = [s for s in synapses.values()]
            syn.sort(reverse=True, key=lambda s: s.wgt)

            print("Synapses Weightings:")

            for s in syn:
                print(
                    f"  {s.name():^20}   wgt: {s.wgt}"
                )

        def get_actives(self, threshold = -1):
            if threshold == -1:
                threshold = self.params()["BindingThreshold"]
            neurones = [(x.name, x.potential) for x in self.network.neurones.values() if x.potential >= threshold]
            return neurones