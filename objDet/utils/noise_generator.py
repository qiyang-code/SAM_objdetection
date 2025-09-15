import json, random, copy, argparse

def add_noise(anns, ratio=0.6):
    anns = copy.deepcopy(anns)
    for ann in anns['annotations']:
        x,y,w,h = ann['bbox']
        dx = random.uniform(-ratio, ratio)*w
        dy = random.uniform(-ratio, ratio)*h
        dw = random.uniform(-ratio, ratio)*w
        dh = random.uniform(-ratio, ratio)*h
        ann['bbox'] = [x+dx, y+dy, w+dw, h+dh]
    return anns