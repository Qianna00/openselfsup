import torch
from PIL import Image
from .registry import DATASETS, PIPELINES
from .base import BaseDataset
from openselfsup.utils import print_log, build_from_cfg
from torchvision.transforms import Compose


@DATASETS.register_module
class ContrastiveDataset(BaseDataset):
    """Dataset for rotation prediction 
    """

    def __init__(self, data_source, pipeline):
        super(ContrastiveDataset, self).__init__(data_source, pipeline)

    def __getitem__(self, idx):
        img = self.data_source.get_sample(idx)
        assert isinstance(img, Image.Image), \
            'The output from the data source must be an Image, got: {}. \
            Please ensure that the list file does not contain labels.'.format(
            type(img))
        img1 = self.pipeline(img)
        img2 = self.pipeline(img)
        img_cat = torch.cat((img1.unsqueeze(0), img2.unsqueeze(0)), dim=0)
        return dict(img=img_cat)

    def evaluate(self, scores, keyword, logger=None):
        raise NotImplemented


# @DATASETS.register_module
"""class MultiScaleContrastiveDataset(BaseDataset):
    def __init__(self, data_source, pipelines):
        pipeline = pipelines[0]
        patch_pipeline = pipelines[1]
        super(MultiScaleContrastiveDataset, self).__init__(data_source, pipeline)
        patch_pipeline = [build_from_cfg(p, PIPELINES) for p in patch_pipeline]
        self.patch_pipeline = Compose(patch_pipeline)



    def __getitem__(self, idx):
        img = self.data_source.get_sample(idx)
        assert isinstance(img, Image.Image), \
            'The output from the data source must be an Image, got: {}. \
            Please ensure that the list file does not contain labels.'.format(
            type(img))
        img1 = self.pipeline(img)
        img2 = self.pipeline(img)
        img3 = self.patch_pipeline(img)
        img4 = self.patch_pipeline(img)
        patch1 = torch.chunk(img3, 3, dim=)

        img_cat = torch.cat((img1.unsqueeze(0), img2.unsqueeze(0)), dim=0)
        return dict(img=img_cat)

    def evaluate(self, scores, keyword, logger=None):
        raise NotImplemented"""
