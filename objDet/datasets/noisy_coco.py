from mmdet.datasets import CocoDataset
import numpy as np

class NoisyCocoDataset(CocoDataset):
    CLASSES = CocoDataset.CLASSES   # 80 类