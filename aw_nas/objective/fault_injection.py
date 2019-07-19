# -*- coding: utf-8 -*-
"""
Fault injection objective.
* Clean accuracy and fault-injected accuracy weighted for reward (for discrete controller search)
* Clean loss and fault-injected loss weighted for loss
  (for differentiable controller search or fault-injection training).

Copyright (c) 2019 Wenshuo Li
Copyright (c) 2019 Xuefei Ning
"""

from collections import defaultdict
import six

import numpy as np
import torch
from torch import nn

from aw_nas import utils
from aw_nas.utils.torch_utils import accuracy
from aw_nas.objective.base import BaseObjective
from aw_nas.utils.exception import expect, ConfigException

class FaultInjector(object):
    def __init__(self, gaussian_std=1., mode="fixed"):
        self.random_inject = 0.001
        self.gaussian_std = gaussian_std
        self.mode = mode
        self.max_value_mode = True
        self.fault_bit_list = np.array([2**x for x in range(8)] + [-2**x for x in range(8)],
                                       dtype=np.float32)

    def set_random_inject(self, value, max_value=True):
        self.random_inject = value
        self.max_value_mode = max_value

    def set_gaussian_std(self, value):
        self.gaussian_std = value

    def inject_gaussian(self, out):
        gaussian = torch.randn(out.shape, dtype=out.dtype, device=out.device) * self.gaussian_std
        out = out + gaussian
        return out

    def inject_saltandpepper(self, out):
        random_tensor = out.new(out.size()).random_(0, 2*int(1./self.random_inject))
        salt_ind = (random_tensor == 0)
        pepper_ind = (random_tensor == 1)
        max_ = torch.max(torch.abs(out)).cpu().data
        out[salt_ind] = 0
        out[pepper_ind] = max_
        return out

    def inject_fixed(self, out):
        # set the fault tensor
        random_tensor = out.new(out.size()).random_(0, int(1. / self.random_inject))
        scale = torch.ceil(torch.log(
            torch.max(torch.max(torch.abs(out)),
                      torch.tensor(1e-5).float().to(out.device))) / np.log(2.))
        step = torch.pow(torch.autograd.Variable(torch.FloatTensor([2.]).to(out.device),
                                                 requires_grad=False),
                         (scale.float() - 7.))
        fault_ind = (random_tensor < 1)
        random_tensor.zero_()
        if self.max_value_mode:
            fault_bias = step * 128.
            random_tensor[fault_ind] = fault_bias
        else:
            random_tensor[fault_ind] = step * \
                torch.tensor(self.fault_bit_list[np.random.randint(
                    0, 16, size=fault_ind.sum().cpu().data)]).to(out.device)
        max_ = torch.max(torch.abs(out)).cpu().data
        out = out + random_tensor

        # clip
        out.clamp_(min=-max_, max=max_)

        # # for masked bp
        # normal_mask = torch.ones_like(out)
        # normal_mask[fault_ind] = 0
        # masked = normal_mask * out
        # out = (out - masked).detach() + masked
        return out

    def inject(self, out):
        return eval("self.inject_" + self.mode)(out) #pylint: disable=eval-used

class FaultInjectionObjective(BaseObjective):
    NAME = "fault_injection"
    SCHEDULABLE_ATTRS = ["fault_reward_coeff", "fault_loss_coeff", "inject_prob", "gaussian_std"]

    def __init__(self, search_space,
                 fault_modes="gaussian", gaussian_std=1., inject_prob=0.001, max_value_mode=True,
                 inject_propto_flops=False,
                 # loss
                 fault_loss_coeff=0.,
                 as_controller_regularization=False,
                 as_evaluator_regularization=False,
                 # reward
                 fault_reward_coeff=0.2,
                 schedule_cfg=None):
        super(FaultInjectionObjective, self).__init__(search_space, schedule_cfg)
        assert 0. <= fault_reward_coeff <= 1.
        self.injector = FaultInjector(gaussian_std, fault_modes)
        self.injector.set_random_inject(inject_prob, max_value_mode)
        self.fault_loss_coeff = fault_loss_coeff
        self.as_controller_regularization = as_controller_regularization
        self.as_evaluator_regularization = as_evaluator_regularization
        if self.fault_loss_coeff > 0:
            expect(self.as_controller_regularization or self.as_evaluator_regularization,
                   "When `fault_loss_coeff` > 0, you should either use this fault-injected loss"
                   " as controller regularization or as evaluator regularization, or both. "
                   "By setting `as_controller_regularization` and `as_evaluator_regularization`.",
                   ConfigException)
        self.fault_reward_coeff = fault_reward_coeff
        self.inject_propto_flops = inject_propto_flops
        if self.inject_propto_flops:
            expect(fault_modes == "fixed",
                   "When `inject_propto_flops` is True, must use the bit-flip fault mode `fixed`",
                   ConfigException)
        self.inject_prob_avg_meters = defaultdict(utils.AverageMeter)

    @classmethod
    def supported_data_types(cls):
        return ["image"]

    def perf_names(cls):
        return ["acc_clean", "acc_fault"]

    def get_reward(self, inputs, outputs, targets, cand_net):
        perfs = self.get_perfs(inputs, outputs, targets, cand_net)
        return perfs[0] * (1 - self.fault_reward_coeff) + perfs[1] * self.fault_reward_coeff

    def get_perfs(self, inputs, outputs, targets, cand_net):
        """
        Get top-1 acc.
        """
        outputs_f = cand_net.forward_one_step_callback(inputs, callback=self.inject)
        return float(accuracy(outputs, targets)[0]) / 100, \
            float(accuracy(outputs_f, targets)[0]) / 100

    def get_loss(self, inputs, outputs, targets, cand_net,
                 add_controller_regularization=True, add_evaluator_regularization=True):
        """
        Get the cross entropy loss *tensor*, optionally add regluarization loss.

        Args:
            inputs: data inputs
            outputs: logits
            targets: labels
        """
        loss = nn.CrossEntropyLoss()(outputs, targets)
        if self.fault_loss_coeff > 0 and \
           ((add_controller_regularization and self.as_controller_regularization) or \
            (add_evaluator_regularization and self.as_evaluator_regularization)):
            # only forward and random inject once, this might not be of high variance
            # for differentiable controller training?
            outputs_f = cand_net.forward_one_step_callback(inputs, callback=self.inject)
            ce_loss_f = nn.CrossEntropyLoss()(outputs_f, targets)
            loss = (1 - self.fault_loss_coeff) * loss + self.fault_loss_coeff * ce_loss_f
        return loss

    def inject(self, state, context):
        if context.is_last_concat_op or not context.is_last_inject:
            return
        assert state is context.last_state
        if self.inject_propto_flops:
            mod = context.last_conv_module
            if mod is None:
                return # last op is not conv op
            mul_per_loc = mod.in_channels / mod.groups * mod.kernel_size[0] * mod.kernel_size[1]
            backup_inject_prob = self.inject_prob
            inject_prob = 1 - (1 - backup_inject_prob) ** mul_per_loc
            self.inject_prob = inject_prob
            self.inject_prob_avg_meters[context.index].update(inject_prob)
        context.last_state = self.injector.inject(state)
        if self.inject_propto_flops:
            self.inject_prob = backup_inject_prob

    def on_epoch_end(self, epoch):
        super(FaultInjectionObjective, self).on_epoch_end(epoch)
        if self.inject_prob_avg_meters:
            # in final trianing, if the base inject prob do not vary, the inject prob of the same
            # position/feature map should always be the same.
            stats = [(ind, meter.avg) for ind, meter in six.iteritems(self.inject_prob_avg_meters)]
            num_pos = len(stats) # number of inject position
            stats = sorted(stats, key=lambda stat: stat[1])
            mean_prob = np.mean([stat[1] for stat in stats])
            geomean_prob = np.prod([stat[1] for stat in stats])**(1.0/num_pos)
            self.logger.info("[NOTE: not meaningful in search, as every pass the same index "
                             "corresponds to different op] Num feature map injected: %3d; "
                             "Inject prob range: [%.4f (%s), %.4f (%s)]; "
                             "Mean: %.4f ; Geometric mean: %.4f",
                             num_pos, stats[0][1], stats[0][0], stats[-1][1], stats[-1][0],
                             mean_prob, geomean_prob)
            self.inject_prob_avg_meters = defaultdict(utils.AverageMeter) # reset

    @property
    def inject_prob(self):
        return self.injector.random_inject

    @inject_prob.setter
    def inject_prob(self, value):
        self.injector.set_random_inject(value)

    @property
    def gaussian_std(self):
        return self.injector.gaussian_std

    @gaussian_std.setter
    def gaussian_std(self, value):
        self.injector.set_gaussian_std(value)
