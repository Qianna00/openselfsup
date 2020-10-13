from ..registry import DATASOURCES
from .image_list import ImageList
from PIL import Image


@DATASOURCES.register_module
class Marvel(object):

    def __init__(self, list_file):
        with open(list_file, 'r') as f:
            lines = f.readlines()
        self.fns = []
        self.labels = []
        for l in lines:
            fn = l.split(',')[-1][:-1]
            label = l.split(',')[2]
            self.fns.append(fn)
            self.labels.append(label)

    def get_length(self):
        return len(self.fns)

    def get_sample(self, idx):
        img = Image.open(self.fns[idx])
        img = img.convert('RGB')
        label = self.labels[idx]
        return img, label
