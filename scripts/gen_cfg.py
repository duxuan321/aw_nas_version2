# -- coding: utf-8 -*-

import os
import copy
import time
import yaml
import subprocess
import multiprocessing
import argparse
from icecream import ic

parser = argparse.ArgumentParser()
parser.add_argument("config_dir")
parser.add_argument("--result-dir", required=True)
args = parser.parse_args()

# Generate derive config file, add "avoid_repeat: true"
ic(args.config_dir)
with open(os.path.join(args.config_dir, "config.yaml"), "r") as fr:
    cfg = yaml.load(fr)

#! with open(os.path.join(args.config_dir, "sample.yaml"), "r") as fr:
with open(os.path.join("/data/duxuan/aw_nas_private/", "sample_epo20.yaml"), "r") as fr:
    samples = yaml.load(fr)

os.makedirs(args.result_dir, exist_ok=True)
for i, geno in enumerate(samples):
    cfg["final_model_cfg"]["genotypes"] = geno
    cfg_file = os.path.join(args.result_dir, f"{i}.yaml")
    with open(cfg_file, "w") as fw:
        yaml.dump(cfg, fw)

