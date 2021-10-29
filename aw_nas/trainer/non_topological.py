# -*- coding: utf-8 -*-
"""
Trainer definition, this is the orchestration of all the components.
"""

from __future__ import print_function
from __future__ import division

import os

import torch

from aw_nas import utils
from aw_nas.trainer.simple import SimpleTrainer
from aw_nas.utils.exception import expect, ConfigException


__all__ = ["NonTopologicalTrainer"]

class NonTopologicalTrainer(SimpleTrainer):
    """
    Trainer for non-topological search space, 
    load weight from final training result for continuous supernet-training.

    Schedulable Attributes:
        mepa_surrogate_steps:
        mepa_samples:
        controller_steps:
        controller_surrogate_steps:
        controller_samples:
    """

    NAME = "non-topological"

    def setup(self, load=None, save_every=None, save_controller_every=None,train_dir=None, writer=None, load_components=None,
              interleave_report_every=None):
        """
        Setup the scaffold: saving/loading/visualization settings.
        """
        if load is not None:
            if os.path.isdir(load):
                all_components = ("controller", "evaluator", "trainer", "model_state.pt")
                load_components = all_components\
                                if load_components is None else load_components
                expect(set(load_components).issubset(all_components), "Invalid `load_components`")

                if "controller" in load_components:
                    path = os.path.join(load, "controller")
                    self.logger.info("Load controller from %s", path)
                    try:
                        self.controller.load(path)
                    except Exception as e:
                        self.logger.error("Controller not loaded! %s", e)
                if "evaluator" in load_components:
                    path = os.path.join(load, "evaluator")
                    # if os.path.exists(path):
                    self.logger.info("Load evaluator from %s", path)
                    try:
                        self.evaluator.load(path)
                    except Exception as e:
                        self.logger.error("Evaluator not loaded: %s", e)
                if "trainer" in load_components:
                    path = os.path.join(load, "trainer")
                    # if os.path.exists(path):
                    self.logger.info("Load trainer from %s", path)
                    try:
                        self.load(path)
                    except Exception as e:
                        self.logger.error("Trainer not loaded: %s", e)
            else:
                self.logger.info("Load evaluator from final training result %s", load)
                state_dict = torch.load(load, "cpu")
                tmp_path = os.path.join(os.path.dirname(load), "fake_evaluator_{}".format(os.environ.get("LOCAL_RANK", 0)))
                torch.save({"weights_manager": state_dict}, tmp_path)
                try:
                    self.evaluator.load(tmp_path)
                except Exception as e:
                        self.logger.error("Evaluator not loaded: %s", e)
                os.remove(tmp_path)

        self.save_every = save_every
        self.save_controller_every = save_controller_every
        self.train_dir = utils.makedir(train_dir) if train_dir is not None else train_dir
        if writer is not None:
            self.setup_writer(writer.get_sub_writer("trainer"))
            self.controller.setup_writer(writer.get_sub_writer("controller"))
            self.evaluator.setup_writer(writer.get_sub_writer("evaluator"))
        self.interleave_report_every = interleave_report_every
        self.is_setup = True
