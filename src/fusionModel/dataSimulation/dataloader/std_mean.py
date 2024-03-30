from PIL import ImageStat, Image
import os
from operator import add
import yaml

class Stats(ImageStat.Stat):
    def __add__(self, other):
        return Stats(list(map(add, self.h, other.h)))
    
split = 'val_images'
imageDir = '/home/anirudhan/Documents/project/fusion/Fusion/data/val/images'
images = os.listdir(imageDir)

statistics = None
for image in images:
    if statistics is None:
        statistics = Stats(Image.open(f'{imageDir}/{image}'))

    else:
        statistics +=  Stats(Image.open(f'{imageDir}/{image}'))
mean = [round(val, 2) for val in statistics.mean]
std = [round(val, 2) for val in statistics.stddev]

if os.path.exists('dataset.yml'):
    with open('dataset.yml', 'r') as file:
        data = yaml.safe_load(file)
else:
    data = {}


data['mean'] = mean
data['std'] = std
variable = {split: data}
print(f'mean:{[round(val, 2) for val in statistics.mean]}, std:{[round(val, 2) for val in statistics.stddev]}')
# mean:[199.59, 156.30, 170.59], std:[31.30, 31.28, 35.95]

with open('dataset.yml', 'w') as file:
    yaml.dump(variable, file)