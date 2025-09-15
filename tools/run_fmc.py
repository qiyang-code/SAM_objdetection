#!/usr/bin/env python
import os, json, cv2, pickle, tqdm, argparse
from fmgdet.utils import fmc_one_image

parser = argparse.ArgumentParser()
parser.add_argument('--img-dir', required=True)
parser.add_argument('--ann', required=True)
parser.add_argument('--out-dir', required=True)
args = parser.parse_args()
os.makedirs(args.out_dir, exist_ok=True)

coco = json.load(open(args.ann))
img_map = {i['id']: i for i in coco['images']}
anns  = {}
for ann in coco['annotations']:
    anns.setdefault(ann['image_id'], []).append(ann)

for img_id, img_info in tqdm.tqdm(img_map.items()):
    img_path = os.path.join(args.img_dir, img_info['file_name'])
    bbox_list = [a['bbox'] for a in anns[img_id]]
    label_list = [coco['categories'][a['category_id']-1]['name'] for a in anns[img_id]]
    corrected = fmc_one_image(img_path, bbox_list, label_list)
    pickle.dump(corrected, open(os.path.join(args.out_dir, f'{img_id}.pkl'), 'wb'))