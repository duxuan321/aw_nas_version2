import os
import re
from turtle import pd
dirname = "/data/duxuan/aw_nas_private/results"
from icecream import ic

latency = {}
for ir_name in os.listdir(dirname):
    if ir_name.endswith("ir"):
        txt = os.path.join(dirname, ir_name, "inst_output/inst.info.txt")
        if not os.path.exists(txt):
            continue
        with open(txt, "r") as fr:
            a = fr.read().split("\n")[-2]
        if "TOTAL" in a:
            a = a.split("\t")[-2]
            latency[ir_name] = float(a.replace(",", ""))
lat = sorted([(k.split(".")[0], v) for k, v in latency.items()], key=lambda x: x[1])
#! lat = dict([(k, v) for k, v in lat if v < 32000])
lat = dict([(k, v) for k, v in lat if v < 50000])

import json
with open("latency.json", "w") as fw:
    json.dump(lat, fw)
