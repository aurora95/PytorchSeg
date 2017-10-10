from __future__ import absolute_import

import torch

class MetricBase(object):
    def __init__(self, name, **kwargs):
        self.name = str(name)
        self._kwargs = kwargs
        self.reset()

    def __str__(self):
        return "MetricBase: {}".format(dict(self.get_name_value()))

    def get_config(self):
        config = self._kwargs.copy()
        config.update({
            'metric': self.__class__.__name__,
            'name': self.name})
        return config

    def update(self, preds, labels):
        raise NotImplementedError()

    def reset(self):
        self.num_inst = 0
        self.sum_metric = 0.0

    def get(self):
        if self.num_inst == 0:
            return (self.name, float('nan'))
        else:
            return (self.name, self.sum_metric / self.num_inst)

    def get_name_value(self):
        name, value = self.get()
        if not isinstance(name, list):
            name = [name]
        if not isinstance(value, list):
            value = [value]
        return list(zip(name, value))

class SparseAccuracy(MetricBase):
    def __init__(self, axis=1, ignore_label=255, name='accuracy'):
        super(SparseAccuracy, self).__init__(name, axis=axis)
        self.axis = axis
        self.ignore_label = ignore_label

    def update(self, preds, labels):
        for pred, label in zip(preds, labels):
            pred_value, pred_label = torch.max(pred, dim=1)

            #pred_label = pred_label.view(-1)
            #label = label.view(-1)
            pred_label = pred_label[label != self.ignore_label].contiguous()
            label = label[label != self.ignore_label].contiguous()
            correct = (pred_label == label).float().sum().cpu().data.numpy()[0]
            total = len(label)

            self.sum_metric += correct
            self.num_inst += total
