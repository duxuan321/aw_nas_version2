import os
import re
from icecream import ic

dirname = "/data/duxuan/aw_nas_private/results/genotypes/lat_logs"

mAP = {}
for name in os.listdir(dirname):
    ic(name)
    if not name.endswith("log"):
        continue
    with open(os.path.join(dirname, name), "r") as fr:
        a = fr.read()
    ic(a)
    try:
        a = re.findall("INFO: mAP: ([0-9.]*),", a)[0]
        name = name.split(".")[0]
        mAP[name] = float(a)
    except:
        pass
    print(name, a)

import json
with open("mAP.json", "w") as fw:
    json.dump(mAP, fw)
