from ..registry import DATASOURCES
from .image_list import ImageList
from PIL import Image


@DATASOURCES.register_module
class Marvel(object):

    def __init__(self, list_file):
        with open(list_file, 'r') as f:
            lines = f.readlines()
        self.fns = [l.split(',')[-1][:-1] if l.split(',')[-1] != "/root/data/zq/data/marvel/400k/W9_8/609684.jpg"
                    else l.split(',')[-1] for l in lines]
        self.labels = [int(l.split(',')[2])-1 for l in lines]

        self.classes = ['Container Ship', 'Bulk Carrier', 'Passengers Ship', 'Ro-ro/passenger Ship',
                        'Ro-ro Cargo', 'Tug', 'Vehicles Carrier', 'Reefer', 'Yacht', 'Sailing Vessel',
                        'Heavy Load Carrier', 'Wood Chips Carrier', 'Livestock Carrier', 'Fire Fighting Vessel',
                        'Patrol Vessel', 'Platform', 'Standby Safety Vessel', 'Combat Vessel',
                        'Training Ship', 'Icebreaker', 'Replenishment Vessel', 'Tankers', 'Fishing Vessels',
                        'Supply Vessels', 'Carrier/Floating', 'Dredgers']

    def get_length(self):
        return len(self.fns)

    def get_sample(self, idx):
        img = Image.open(self.fns[idx])
        img = img.convert('RGB')
        label = self.labels[idx]
        return img, label
