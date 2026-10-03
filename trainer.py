import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau


class Optim(object):

    def _makeOptimizer(self):
        if self.method == 'sgd':
            self.optimizer = optim.SGD(self.params, lr=self.lr, weight_decay=self.lr_decay)
        elif self.method == 'adagrad':
            self.optimizer = optim.Adagrad(self.params, lr=self.lr, weight_decay=self.lr_decay)
        elif self.method == 'adadelta':
            self.optimizer = optim.Adadelta(self.params, lr=self.lr, weight_decay=self.lr_decay)
        elif self.method == 'adam':
            self.optimizer = optim.Adam(self.params, lr=self.lr, weight_decay=self.lr_decay)
        elif self.method == 'Nadam':
            self.optimizer = optim.NAdam(self.params, lr=self.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        elif self.method == 'adamw':
            self.optimizer = optim.AdamW(self.params, lr=self.lr, weight_decay=self.lr_decay)
        else:
            raise RuntimeError("Invalid optim method: " + self.method)

    def __init__(
        self,
        params,
        method,
        lr,
        clip,
        mode,
        factor,
        patience,
        steps_per_epoch=None,
        epochs=None,
        lr_decay=1,
        start_decay_at=None,
    ):
        self.params = list(params)
        self.step_count = 0
        self.lr = lr
        self.clip = clip
        self.method = method
        self.lr_decay = lr_decay
        self._makeOptimizer()
        try:
            self.scheduler = ReduceLROnPlateau(self.optimizer, mode=mode, factor=factor, patience=patience, verbose=True)
        except TypeError:
            self.scheduler = ReduceLROnPlateau(self.optimizer, mode=mode, factor=factor, patience=patience)

    def step(self):
        grad_norm = 0
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.params, self.clip)
        self.optimizer.step()
        self.step_count += 1
        return grad_norm

    def lronplateau(self, loss):
        self.scheduler.step(loss)


class MultiGroupPlateauOptim(object):
    def __init__(self, param_groups, clip, mode, factor, patience):
        self.params = []
        materialized_groups = []
        for group in param_groups:
            params = list(group["params"])
            self.params.extend(params)
            next_group = dict(group)
            next_group["params"] = params
            materialized_groups.append(next_group)

        self.clip = clip
        self.step_count = 0
        self.optimizer = optim.AdamW(materialized_groups)
        try:
            self.scheduler = ReduceLROnPlateau(self.optimizer, mode=mode, factor=factor, patience=patience, verbose=True)
        except TypeError:
            self.scheduler = ReduceLROnPlateau(self.optimizer, mode=mode, factor=factor, patience=patience)

    def step(self):
        trainable = [p for p in self.params if p.requires_grad]
        if self.clip is not None and trainable:
            torch.nn.utils.clip_grad_norm_(trainable, self.clip)
        self.optimizer.step()
        self.step_count += 1
        return 0

    def lronplateau(self, loss):
        self.scheduler.step(loss)
