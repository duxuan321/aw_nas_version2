
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("cfg_dir")
parser.add_argument("--result-dir", required=True)
args = parser.parse_args()

def to_ir(model, data_loader, dirname):

    import importlib
    importlib.invalidate_caches()
    model_wrapper = ModelWrapper(model, (3, 480, 800))
    model_wrapper.init_fuse_model()
    model_wrapper.init_quan_model()
    model_wrapper.calibrate_quan_model(data_loader, num_batchs=1)
    model_wrapper.parse_to_IR(dirname)


for cfg_name in os.listdir(args.cfg_dir):
    import sys
    modules = list(sys.modules.keys())
    for m in modules:
        if hasattr(sys.modules[m], '__file__') and sys.modules[m].__file__:
            if "tmp_file" in sys.modules[m].__file__:
                del sys.modules[m]
    import yaml
    import torch

    from dpt.build.build_model import ModelWrapper
    from aw_nas.main import _init_component

    if cfg_name.endswith("yaml"):
        cfg = os.path.join(args.cfg_dir, cfg_name)
        with open(cfg, "r") as fr:
            cfg = yaml.load(fr)
        print("**" * 10, cfg_name)
        search_space = _init_component(cfg, "search_space")
        model = _init_component(cfg, "final_model", search_space=search_space, device=0)
        data_loader = [(torch.rand(1, 3, 480, 800).cuda(), None)]
        to_ir(model, data_loader, os.path.join(args.result_dir, cfg_name.split(".")[0] + ".ir"))
        
        import shutil
        shutil.rmtree("tmp_file")

