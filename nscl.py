# from main import networkx
from genericpath import getsize
from inspect import trace
from pickle import NONE
from nscl_algo import NSCLAlgo
from datetime import date, datetime
import json
import os, sys, psutil

pathdiv = ""

if os.name == "posix":
    pathdiv = "/"
elif os.name == "nt":
    pathdiv = "\\"


class NSCL:

    defparams = {
        "F1": [0.8, 1.0],
        "F2": [0.12, 6.0],
        "A": [0.01, 140],
        "PTP": [0.01, 600],
        "LTP": [0.01, 30000],
        "Bindings": 2,
        "Levels": 2,
        "TraceLength": 60,
        "TimeConstant": [3],
        "FiringThreshold": [0.5],
        "MaxPropagation": [6],
        "DecayCoefficient": [0.2],
        "PruneInterval": [60*60*24*30],
    }

    class NSymbol:
        def __init__(self, name, counter=0, potential=0, lastspike="") -> None:
            self.name = name
            self.counter = counter
            # self.pot = [0 for i in range(0,NSCL.defparams['TrainLength'])]
            self.potential = potential
            # self.pot2 = 0
            # self.pot3 = 0
            self.fsynapses = []
            self.rsynapses = []
            self.lastspike = lastspike
            self.occurs = 1
            self.heirarcs = []
            # self.level = -1

        def level(self):
            self.level = max(self.heirarcs)
            return self.level

    class SSynapse:
        def __init__(
            self, r_neurone, f_neurone, wgt=0.01, counter=0, lastspike=""
        ) -> None:
            self.fref = f_neurone
            self.rref = r_neurone
            self.wgt = wgt
            self.counter = counter
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
            self.neurones = {}
            self.synapses = {}
            self.params = {**NSCL.defparams, **params}

        def net_struct(self) -> object:
            return {
                "params": self.params,
                "sneurones": self.neurones,
                "ssynapses": self.synapses,
            }

        def __repr__(self) -> str:
            return str(self.structure())

    class Engine:
        def __init__(self, network=None, algorithm=None) -> None:
            self.network = NSCL.Network() if network == None else network
            self.npruned = {}
            self.spruned = {}
            self._algo = algorithm if algorithm else NSCLAlgo.algo1
            self.defparams = NSCL.defparams
            self.traces = []
            self.ntime = []
            self.ncounts = []
            self.nmask = []
            self.tick = 0
            self.prune = NSCL.defparams["PruneInterval"][0]

        def clear_traces(self) -> None:
            self.traces = []
            self.ntime = []
            self.ncounts = []
            self.nmask = []

        def params(self) -> dict:
            return self.network.params

        def algo(self, input, trace, now=None, prune=False) -> list:
            r = self._algo(self, input, now=now, prune=prune)
            it = self.tick

            if trace:
                self.traces.append(r["trace1"])
                self.ntime.append(it)
                self.ncounts.append(len(r["trace1"]))
                self.nmask.append("reinforced")

                self.ntime.append(it)
                self.ncounts.append(len(self.ineurones()))
                self.nmask.append("input")

                self.ntime.append(it)
                self.ncounts.append(len(self.gneurones()))
                self.nmask.append("composite")

                self.ntime.append(it)
                self.ncounts.append(len(self.neurones()))
                self.nmask.append("total")

                self.ntime.append(it)
                self.ncounts.append(len(self.synapses()))
                self.nmask.append("synapses")

                while len(self.traces) > NSCL.defparams["TraceLength"]:
                    self.traces.pop(0)
                while len(self.ntime) * 5 > NSCL.defparams["TraceLength"] * 5:
                    self.ntime.pop(0)
                while len(self.ncounts) * 5 > NSCL.defparams["TraceLength"] * 5:
                    self.ncounts.pop(0)
                while len(self.nmask) * 5 > NSCL.defparams["TraceLength"] * 5:
                    self.nmask.pop(0)

            self.tick += 1
            return r

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

        def remove_neurone(self, name) -> None:
            try:
                n = self.neurones[name]
                for f in n.fsynapses:
                    nf = self.neurones[f.fref]
                    if name in nf.rsynapses:
                        sn = NSCLAlgo.sname(name, f.fref)
                        self.spruned[sn] = self.synapses.pop(sn)
                for r in n.rsynapses:
                    nr = self.neurones[r.rref]
                    if name in nr.rsynapses:
                        sn = NSCLAlgo.sname(r.rref, name)
                        self.spruned[sn] = self.synapses.pop(sn)
                delneurone = self.neurones.pop(name)
                self.npruned[delneurone.name] = delneurone
            except:
                input("remove failed")

        def new_ssynapse(
            self, pre_sneurone, post_sneurone, wgt=0.01, counter=0, lastspike=""
        ) -> bool:
            NSCLAlgo.new_ssynapse(
                self, pre_sneurone, post_sneurone, wgt, counter, lastspike
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
                        "counter": n.counter,
                        "potential": n.potential,
                        "fsynapses": n.fsynapses,
                        "rsynapses": n.rsynapses,
                        "lastspike": n.lastspike,
                        "occurs": n.occurs,
                        "heirarcs": n.heirarcs,
                    }
                )

            for k in synapses:
                s = synapses[k]
                content["synapses"].append(
                    {
                        "fref": s.fref,
                        "rref": s.rref,
                        "wgt": s.wgt,
                        "counter": s.counter,
                        "lastspike": s.lastspike,
                    }
                )

            for k in npruned:
                n = npruned[k]
                content["npruned"].append(
                    {
                        "name": n.name,
                        "counter": n.counter,
                        "potential": n.potential,
                        "fsynapses": n.fsynapses,
                        "rsynapses": n.rsynapses,
                        "lastspike": n.lastspike,
                        "occurs": n.occurs,
                        "heirarcs": n.heirarcs,
                    }
                )

            for k in spruned:
                s = spruned[k]
                content["spruned"].append(
                    {
                        "fref": s.fref,
                        "rref": s.rref,
                        "wgt": s.wgt,
                        "counter": s.counter,
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

        def load_state(self, fname) -> None:

            rpath = f"states{pathdiv}%s" % fname

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
                n.counter = nprop["counter"]
                n.potential = nprop["potential"]
                n.fsynapses = nprop["fsynapses"]
                n.rsynapses = nprop["rsynapses"]
                n.lastspike = nprop["lastspike"]
                n.occurs = nprop["occurs"]
                n.heirarcs = nprop["heirarcs"]

            for sprop in load_state["synapses"]:
                pre = sprop["rref"]
                post = sprop["fref"]
                self.new_ssynapse(
                    pre, post, sprop["wgt"], sprop["counter"], sprop["lastspike"]
                )

            self.npruned = load_state["npruned"]
            self.spruned = load_state["spruned"]            

                # s = eng.network.synapses[NSCLAlgo.sname(pre, post)]
                # self.network.neurones = content["neurones"]
                # self.network.synapses = content["synapses"]
                # self.network.params = content["params"]

            self.defparams = load_state["params"]

            self.tick = load_state["tick"]
            self.traces = load_traces["traces"]
            self.ncounts = load_traces["ncounts"]
            self.nmask = load_traces["nmask"]
            self.ntime = load_traces["ntime"]

            NSCLAlgo.relevel(self)

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
            return (n_size, s_size, p_size, traces, process_rss, process_vms, all)

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
                    f"  {s.name():^20}   wgt: {s.wgt}   cnt: {s.counter}   lspk: {s.lastspike}"
                )
