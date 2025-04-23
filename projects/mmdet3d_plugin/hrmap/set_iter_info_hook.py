# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.parallel import is_module_wrapper
from mmcv.runner import HOOKS, Hook

@HOOKS.register_module()
class SetIterInfoHook(Hook):
    """Set runner's epoch information to the model."""

    def __init__(self,
                 start=None,
                 interval=1,
                 by_epoch=True):

        self.start = start
        self.interval = interval
        self.by_epoch = by_epoch

    def after_train_iter(self, runner):
        # print("here iter")
        iteration = runner.iter
        model = runner.model
        if is_module_wrapper(model):
            model = model.module
        model.set_iter(iteration)
