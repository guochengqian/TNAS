#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2021.03 #
#####################################################
import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from typing import Optional, Callable

import spaces
from .super_module import SuperModule
from .super_module import IntSpaceType
from .super_module import BoolSpaceType


class SuperReLU(SuperModule):
    """Applies a the rectified linear unit function element-wise."""

    def __init__(self, inplace=False) -> None:
        super(SuperReLU, self).__init__()
        self._inplace = inplace

    @property
    def abstract_search_space(self):
        return spaces.VirtualNode(id(self))

    def forward_candidate(self, input: torch.Tensor) -> torch.Tensor:
        return self.forward_raw(input)

    def forward_raw(self, input: torch.Tensor) -> torch.Tensor:
        return F.relu(input, inplace=self._inplace)

    def extra_repr(self) -> str:
        return "inplace=True" if self._inplace else ""
