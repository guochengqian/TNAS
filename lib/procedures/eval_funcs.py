#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2020.04 #
#####################################################
import abc

def obtain_accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class EvaluationMetric(abc.ABC):
    
    def __init__(self):
        self._total_metrics = 0

    def __len__(self):
        return self._total_metrics