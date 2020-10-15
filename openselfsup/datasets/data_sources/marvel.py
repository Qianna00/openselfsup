from ..registry import DATASOURCES
from .image_list import ImageList
from PIL import Image


@DATASOURCES.register_module
class Marvel(object):

    def __init__(self, list_file):
        with open(list_file, 'r') as f:
            lines = f.readlines()
        self.fns = [l.split(',')[-1][:-1] if l.split(',')[-1] != "/root/data/zq/data/marvel/140k/W9_5/925860.jpg"
                    else l.split(',')[-1] for l in lines]
        self.labels = [int(l.split(',')[2]) for l in lines]

    def get_length(self):
        return len(self.fns)

    def get_sample(self, idx):
        img = Image.open(self.fns[idx])
        img = img.convert('RGB')
        label = self.labels[idx]
        return img, label
