import functools
from typing import List
from aw_nas.germ.decisions import apply_post_fn

import torch
import torch.nn.functional as F
from torch import nn

from aw_nas import germ
from aw_nas.ops import get_op, MobileNetV2Block
from aw_nas.utils import make_divisible, feature_level_to_stage_index
from icecream import ic

def schedule_choice_callback(
    choices: germ.Choices, epoch: int, schedule: List[dict]
) -> None:
    """
    Args:
        choices: instances of Choices
        epoch: int
        schedule: list
            [
                {
                    "epoch": int,
                    "choices": list,
                },
                ...
            ]
    """
    if schedule is None:
        return
    for sch in schedule:
        assert "epoch" in sch and "choices" in sch
        if epoch >= sch["epoch"]:
            choices.choices = sch["choices"]
    print(
        "Epoch: {:>4d}, decision id: {}, choices: {}".format(
            epoch, choices.decision_id, choices.choices
        )
    )


class MobileNetV2(germ.SearchableBlock):
    NAME = "mbv2"

    def __init__(
        self,
        ctx,
        num_classes=10,
        depth_choices=[2, 3, 4],
        strides=[2, 2, 2, 1, 2, 1],
        channels=[32, 16, 24, 32, 64, 96, 160, 320, 1280],
        mult_ratio_choices=(1.0,),
        kernel_sizes=[3, 5, 7],
        expansion_choices=[2, 3, 4, 6],
        activation="relu",
        stem_stride=2,
        first_stride=1,
        pretrained_path=None,
        schedule_cfg={},
    ):
        super().__init__(ctx)

        self.num_classes = num_classes
        self.stem_stride = stem_stride
        self.first_stride = first_stride
        self.strides = strides

        self.depth_choices = depth_choices
        self.kernel_sizes = kernel_sizes
        self.expansion_choices = expansion_choices
        self.channels = channels
        self.mult_ratio_choices = mult_ratio_choices

        self.stem = nn.Sequential(
            nn.Conv2d(
                3,
                self.channels[0],
                kernel_size=3,
                stride=self.stem_stride,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(self.channels[0]),
            get_op(activation)(),
        )

        kernel_choices_cb = functools.partial(
            schedule_choice_callback, schedule=schedule_cfg.get("kernel_sizes")
        )
        exp_choices_cb = functools.partial(
            schedule_choice_callback, schedule=schedule_cfg.get("expansion_choices")
        )
        width_choices_cb = functools.partial(
            schedule_choice_callback, schedule=schedule_cfg.get("mult_ratio_choices")
        )
        depth_choices_cb = functools.partial(
            schedule_choice_callback, schedule=schedule_cfg.get("depth_choices")
        )

        divisor_fn = functools.partial(make_divisible, divisor=8)

        prev_channels = (germ.Choices(
            mult_ratio_choices,
            epoch_callback=width_choices_cb
        ) * self.channels[1]).apply(divisor_fn)
        ic(prev_channels, prev_channels.choices)
        
        self.first_block = germ.SearchableMBV2Block(
            ctx,
            self.channels[0],
            prev_channels,
            1,
            3,
            stride=first_stride,
        )

        self.cells = nn.ModuleList([])
        self.depth_decs = germ.DecisionDict()
        self.stage_out_channel_decs = germ.DecisionDict()

        for i, stride in enumerate(self.strides):
            stage = nn.ModuleList([])
            cur_channels = (
                germ.Choices(
                    mult_ratio_choices,
                    epoch_callback=width_choices_cb
                )
                * self.channels[i + 2]
            ).apply(divisor_fn)
            if stride == 1 and isinstance(prev_channels, germ.Choices) and \
                cur_channels.choices == prev_channels.choices:
                # accidently, we found the first block in some stage that has the same channels
                # coincidentlly, then add shortcut to it.
                cur_channels = prev_channels

            if i < len(self.strides) - 1:
                self.depth_decs[str(i)] = germ.Choices(
                    depth_choices, epoch_callback=depth_choices_cb
                )
            for j in range(max(depth_choices)):
                exp_ratio = germ.Choices(
                    expansion_choices,
                    epoch_callback=exp_choices_cb
                )
                kernel_choice = germ.Choices(
                    self.kernel_sizes, epoch_callback=kernel_choices_cb
                )

                block = germ.SearchableMBV2Block(
                    ctx,
                    prev_channels,
                    cur_channels,
                    exp_ratio,
                    kernel_choice,
                    stride=stride if j == 0 else 1,
                    short_cut=j > 0
                )
                prev_channels = cur_channels
                stage.append(block)

                if i == len(self.strides) - 1:
                    # last cell has one block only
                    break

            self.cells.append(stage)

        self.conv_final = germ.SearchableConvBNBlock(
            ctx, prev_channels, self.channels[-1], 1
        )

        self.classifier = nn.Conv2d(self.channels[-1], num_classes, 1, 1, 0)

        if pretrained_path:
            state_dict = torch.load(pretrained_path, "cpu")
            if (
                "classifier.weight" in state_dict
                and state_dict["classifier.weight"].shape[0] != self.num_classes
            ):
                del state_dict["classifier.weight"]
                del state_dict["classifier.bias"]
            self.logger.info(self.load_state_dict(state_dict, strict=False))

    def extract_features_rollout(self, rollout, inputs, p_levels=None):
        self.ctx.rollout = rollout
        return self.extract_features(inputs, p_levels)

    def extract_features(self, inputs, p_levels=None):
        stemed = self.stem(inputs)
        out = self.first_block(stemed)
        features = [inputs, out]
        for i, cell in enumerate(self.cells):
            if self.ctx.rollout is not None and str(i) in self.depth_decs:
                depth = self._get_decision(self.depth_decs[str(i)], self.ctx.rollout)
            else:
                depth = len(cell)
            for j, block in enumerate(cell):
                if j >= depth:
                    break
                out = block(out)
            if i == len(self.strides) - 1 or self.strides[i + 1] == 2:
                features.append(out)
        if p_levels is not None:
            features = [features[p] for p in p_levels]
        return features

    def forward(self, inputs):
        features = self.extract_features(inputs)
        out = features[-1]
        out = F.adaptive_avg_pool2d(out, 1)
        out = self.conv_final.forward(out)
        return self.classifier(out).flatten(1)

    def finalize_rollout(self, rollout):
        with self.finalize_context(rollout):
            self.first_block = self.first_block.finalize_rollout(rollout)
            cells = nn.ModuleList()
            for i, cell in enumerate(self.cells):
                if str(i) in self.depth_decs:
                    depth = self._get_decision(self.depth_decs[str(i)], rollout)
                else:
                    depth = len(cell)
                cells.append(
                    nn.ModuleList([c.finalize_rollout(rollout) for c in cell[:depth]])
                )
            self.cells = cells
            self.conv_final.finalize_rollout(rollout)
        return self

    def get_level_indexes(self):
        level_indexes = feature_level_to_stage_index(self.strides,
                int(self.stem_stride == 2) + int(self.first_stride == 2))
        return level_indexes

    def get_feature_channel_num(self, p_levels):
        level_indexes = self.get_level_indexes()
        return [
            self.cells[level_indexes[p]][-1].out_channels for p in p_levels
        ]


class MBV2SuperNet(germ.GermSuperNet):
    NAME = "mbv2"

    def __init__(self, search_space, *args, **kwargs):
        super().__init__(search_space)
        with self.begin_searchable() as ctx:
            self.backbone = MobileNetV2(ctx, *args, **kwargs)

    def forward(self, inputs):
        return self.backbone(inputs)

    def extract_features(self, inputs, p_levels):
        return self.backbone.extract_features(inputs, p_levels)

    def extract_features_rollout(self, rollout, inputs, p_levels):
        self.ctx.rollout = rollout
        return self.extract_features(inputs, p_levels)

    def get_level_indexes(self):
        return self.backbone.get_level_indexes()

    def get_feature_channel_num(self, p_levels):
        return self.backbone.get_feature_channel_num(p_levels)
