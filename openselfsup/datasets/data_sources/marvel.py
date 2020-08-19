from ..registry import DATASOURCES
from .image_list import ImageList
from PIL import Image


@DATASOURCES.register_module
class Marvel(object):

    def __init__(self, list_file):
        with open(list_file, 'r') as f:
            lines = f.readlines()
        self.fns = [l.split(',')[-1][:-1] for l in lines]

    def get_length(self):
        return len(self.fns)

    def get_sample(self, idx):
        img = Image.open(self.fns[idx])
        img = img.convert('RGB')
        return img
