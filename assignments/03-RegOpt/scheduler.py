from typing import List

from torch.optim.lr_scheduler import _LRScheduler

import numpy as np


class CustomLRScheduler(_LRScheduler):
    """
    Create a custom cyclical learning rate scheduler from scratch
    """

    def __init__(
        self,
        optimizer,
        last_epoch=-1,
        step_size=1000,
        gamma=0.998,
        mode="triangular2",
        max_lr=0.01,
    ):
        """
        Create a cyclical learning rate scheduler.

        base_lr: initial learning rate which is the
            lower boundary in the cycle.

        max_lr: upper boundary in the cycle. Functionally,
            it defines the cycle amplitude (max_lr - base_lr).
            The lr at any cycle is the sum of base_lr
            and some scaling of the amplitude; therefore
            max_lr may not actually be reached depending on
            scaling function.

        step_size: number of training iterations per
            half cycle.

        mode: one of {triangular, triangular2, exp_range}.

        gamma: constant in 'exp_range' scaling function:
            gamma**(cycle iterations)

        scale_fn: Custom scaling policy defined by a single
            argument lambda function, where
            0 <= scale_fn(x) <= 1 for all x >= 0.
            mode paramater is ignored

        scale_mode: {'cycle', 'iterations'}.
            Defines whether scale_fn is evaluated on
            cycle number or cycle iterations (training
            iterations since start of cycle). Default is 'cycle'.

        """
        self.step_size = step_size
        self.gamma = gamma
        self.mode = mode
        self.max_lr = max_lr

        if self.mode == "triangular":
            self.scale_fn = lambda x: 1.0
            self.scale_mode = "cycle"
        elif self.mode == "triangular2":
            self.scale_fn = lambda x: 1 / (2.0 ** (x - 1))
            self.scale_mode = "cycle"
        elif self.mode == "exp_range":
            self.scale_fn = lambda x: gamma ** (x)
            self.scale_mode = "iterations"

        super(CustomLRScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """
        return the learning rate

        Note to students: You CANNOT change the arguments or return type of
        this function (because it is called internally by Torch)

        """

        cycle = np.floor(1 + self.last_epoch / (2 * self.step_size))

        x = np.abs(self.last_epoch / self.step_size - 2 * cycle + 1)

        if self.scale_mode == "cycle":
            lrs = [
                self.base_lrs[0]
                + (self.max_lr - self.base_lrs[0])
                * np.maximum(0, (1 - x))
                * self.scale_fn(cycle)
            ]
        else:
            lrs = [
                self.base_lrs[0]
                + (self.max_lr - self.base_lrs[0])
                * np.maximum(0, (1 - x))
                * self.scale_fn(self.last_epoch)
            ]

        return lrs
