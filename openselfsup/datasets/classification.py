import torch

from openselfsup.utils import print_log

from .registry import DATASETS
from .base import BaseDataset

from sklearn.metrics import classification_report
from sklearn.utils.multiclass import type_of_target


@DATASETS.register_module
class ClassificationDataset(BaseDataset):
    """Dataset for classification
    """

    def __init__(self, data_source, pipeline):
        super(ClassificationDataset, self).__init__(data_source, pipeline)

    def __getitem__(self, idx):
        img, target = self.data_source.get_sample(idx)
        img = self.pipeline(img)
        return dict(img=img, gt_label=target)

    def evaluate(self, scores, keyword, logger=None, topk=(1, 5)):
        '''results: Tensor (NxC)
        '''
        eval_res = {}

        target = torch.LongTensor(self.data_source.labels)
        assert scores.size(0) == target.size(0), \
            "Inconsistent length for results and labels, {} vs {}".format(
            scores.size(0), target.size(0))
        num = scores.size(0)
        _, pred = scores.topk(max(topk), dim=1, largest=True, sorted=True)
        pred = pred.t()
        print("target:", target.size(), target)
        print("pred:", pred.squeeze().size(), pred.squeeze())
        print("type of target:", type_of_target(target))
        print("type of pred:", type_of_target(pred.squeeze()))
        correct = pred.eq(target.view(1, -1).expand_as(pred))  # KxN
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0).item()
            acc = correct_k * 100.0 / num
            eval_res["{}_top{}".format(keyword, k)] = acc
            if logger is not None and logger != 'silent':
                print_log(
                    "{}_top{}: {:.03f}".format(keyword, k, acc),
                    logger=logger)

        # target_names = self.data_source.classes
        # print(classification_report(target, pred.squeeze(), target_names=target_names))

        return eval_res
