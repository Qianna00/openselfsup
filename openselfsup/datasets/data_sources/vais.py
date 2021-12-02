from ..registry import DATASOURCES
from .image_list import ImageList
from PIL import Image


@DATASOURCES.register_module
class VAIS(object):

    def __init__(self, list_file):
        with open(list_file, 'r') as f:
            lines = f.readlines()
        self.fns = []
        self.labels = []
        self.cls2label = {'passenger': 0, 'sailing': 1, 'cargo': 2, 'tug': 3, 'small': 4, 'medium-other': 5}
        for l in lines:
            fn = '/root/data/zq/data/VAIS/' + l.split(' ')[0]
            if fn != 'null':
                cls = l.split(' ')[3]
                label = self.cls2label[cls]
                self.fns.append(fn)
                self.labels.append(label)

        self.classes = ['passenger', 'sailing', 'cargo', 'tug', 'small', 'medium-other']


    def get_length(self):
        return len(self.fns)

    def get_sample(self, idx):
        img = Image.open(self.fns[idx])
        img = img.convert('RGB')
        # label = self.labels[idx]
        return img
