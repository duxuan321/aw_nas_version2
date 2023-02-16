import numpy as np
import json

with open("mAP.json", "r") as fr:
    mAP = json.load(fr)

with open("results/latency.json", "r") as fr:
    lat = json.load(fr)

keys = [k for k in lat.keys()]
ret = {}
for k,v in mAP.items():
    if k in lat:
        ret[keys[int(k)]] = (mAP[k], -lat[keys[int(k)]])


pareto = {}
def find_pareto(pop):
    pop_keys = list(pop.keys())
    pop_size = len(pop_keys)
    pop = {k: v for k, v in pop.items()}
    for ind1 in range(pop_size):
        k1 = pop_keys[ind1]
        if k1 not in pop:
            continue
        for ind2 in range(ind1, pop_size):
            k2 = pop_keys[ind2]
            if k2 not in pop:
                continue
            if all(np.array(pop[k1]) > np.array(pop[k2])):
                pop.pop(k2)
            elif all(np.array(pop[k1]) < np.array(pop[k2])):
                pop.pop(k1)
                break
    return pop

pareto = find_pareto(ret)
print(pareto)
print([keys.index(i) for i in pareto.keys()])
