#!/usr/bin/env python
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))+'/..')
from mmdet.apis import train_detector
from fmgdet.datasets import NoisyCocoDataset
from fmgdet.models import InterpRoIHead
import mmcv

cfg = mmcv.Config.fromfile('configs/fmgdet_r50_1x.py')
cfg.work_dir = 'work_dirs/fmgdet_r50_1x'
model = mmcv.build_from_cfg(cfg.model, mmcv.Registry('model'))
train_detector(model, cfg.datasets, cfg, distributed=True)