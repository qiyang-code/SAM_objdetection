#!/usr/bin/env python
import json, argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from fmgdet.utils import add_noise_to_coco

parser = argparse.ArgumentParser()
parser.add_argument('coco', help='original coco json')
parser.add_argument('-o', '--out', required=True, help='output noisy json')
parser.add_argument('-r', '--ratio', type=float, default=0.6)
args = parser.parse_args()

data = json.load(open(args.coco))
noisy = add_noise_to_coco(data, args.ratio)
os.makedirs(os.path.dirname(args.out), exist_ok=True)
json.dump(noisy, open(args.out, 'w'), indent=2)
print('saved noisy COCO ->', args.out)