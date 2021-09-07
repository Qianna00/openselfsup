import torch
from PIL import Image
from .registry import DATASETS, PIPELINES
from .base import BaseDataset
from openselfsup.utils import print_log, build_from_cfg
from torchvision.transforms import Compose
import random


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


@DATASETS.register_module
class MultiScaleContrastiveDataset(BaseDataset):
    def __init__(self, data_source, pipelines):
        pipeline = pipelines[0]
        patch_pipeline = pipelines[1]
        super(MultiScaleContrastiveDataset, self).__init__(data_source, pipeline)
        patch_pipeline = [build_from_cfg(p, PIPELINES) for p in patch_pipeline]
        self.patch_pipeline = Compose(patch_pipeline)
        post_pipeline = [dict(type='RandomCrop', size=64),
                         dict(type='Normalize',
                              img_norm_cfg=dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))]
        self.post_pipeline = Compose(post_pipeline)

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
        patches1 = []
        patches2 = []
        patch1_1 = img3.chunk(3, 1)
        patch2_1 = img4.chunk(3, 1)
        for i in range(3):
            patch1_2 = list(patch1_1[i].chunk(3, 2))
            patches1.extend(patch1_2)
            patch2_2 = list(patch2_1[i].chunk(3, 2))
            patches2.extend(patch2_2)
        print(img3)
        print(patch1_1)
        patches1 = torch.cat([self.post_pipeline(patch).unsqueeze(0) for patch in random.shuffle(patches1)], dim=0)
        patches2 = torch.cat([self.post_pipeline(patch).unsqueeze(0) for patch in random.shuffle(patches2)], dim=0)

        img_cat = torch.cat((img1.unsqueeze(0), img2.unsqueeze(0)), dim=0)
        patches_cat = torch.cat((patches1.unsqueeze(0), patches2.unsqueeze(0)), dim=0)
        return dict(img=img_cat, patches=patches_cat)

    def evaluate(self, scores, keyword, logger=None):
        raise NotImplemented
