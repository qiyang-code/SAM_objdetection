#!/usr/bin/env python
import os, sys, torch, mmcv
from mmdet.apis import single_gpu_test, init_detector
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))+'/..')

cfg = mmcv.Config.fromfile('configs/fmgdet_r50_1x.py')
checkpoint = 'work_dirs/fmgdet_r50_1x/latest.pth'
model = init_detector(cfg, checkpoint, device='cuda:0')
single_gpu_test(model, cfg.data.test)