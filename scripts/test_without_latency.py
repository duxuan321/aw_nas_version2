# -- coding: utf-8 -*-

import os
import copy
import time
import yaml
import subprocess
import multiprocessing
import argparse
from icecream import ic

GPUs = [1, 2, 3]
parser = argparse.ArgumentParser()
parser.add_argument("supernet")
parser.add_argument("--result-dir", required=True)
args = parser.parse_args()

# Generate derive config file, add "avoid_repeat: true"

num_processes = len(GPUs)
queue = multiprocessing.Queue(maxsize=num_processes)
log_dir = os.path.join(args.result_dir, "lat_logs")
os.makedirs(log_dir, exist_ok=True)

def _worker(p_id, gpu_id, queue):
    for idx in range(2000):
        print("*" * 10, idx)
        cfg_file = os.path.join(args.result_dir, f"{idx}.yaml")
        derive_log = os.path.join(log_dir, "{}.log".format(idx))
        cmd = f"awnas test {cfg_file} --load-supernet {args.supernet} --gpus {gpu_id} --seed 20 -s test >{derive_log} 2>&1"
        print("Process #{}: GPU {} Get: {}; CMD: {}".format(p_id, gpu_id, idx, cmd))
        try:
            subprocess.check_call(cmd, shell=True)
        except Exception as e:
            import traceback
            traceback.print_exc()
    print("Process #{} end".format(p_id))

for p_id in range(num_processes):
    p = multiprocessing.Process(target=_worker, args=(p_id, GPUs[p_id], queue))
    p.start()

