import torch, torch.nn as nn
from mmdet.models.roi_heads import StandardRoIHead
from mmdet.core import bbox2roi, bbox2result

class InterpRoIHead(StandardRoIHead):
    def __init__(self, **cfg):
        super().__init__(**cfg)
        self.interp = nn.Sequential(
            nn.Linear(256*7*7*2, 256), nn.ReLU(),
            nn.Linear(256, 64), nn.ReLU(),
            nn.Linear(64, 1), nn.Sigmoid())

    def _bbox_forward_train(self, x, sampling_results, gt_bboxes, gt_labels, img_metas):
        rois = bbox2roi([res.bboxes for res in sampling_results])
        bbox_feats = self.bbox_roi_extractor(x[:self.bbox_roi_extractor.num_inputs], rois)
        # 载入离线 FMC 框
        clean_bboxes = self.load_clean_bboxes(img_metas, sampling_results)
        rois_c = bbox2roi(clean_bboxes)
        feats_c = self.bbox_roi_extractor(x[:self.bbox_roi_extractor.num_inputs], rois_c)
        gamma = self.interp(torch.cat([bbox_feats, feats_c], dim=1).view(bbox_feats.size(0), -1))
        # 插值框
        inter_bboxes = (gamma * torch.stack(clean_bboxes) + (1 - gamma) * torch.stack([res.bboxes for res in sampling_results]))
        # 继续父类逻辑
        bbox_results = self._bbox_forward(x, bbox2roi([inter_bboxes[i] for i in range(inter_bboxes.size(0))]))
        bbox_targets = self.get_targets(sampling_results, gt_bboxes, gt_labels, self.train_cfg.rcnn)
        loss_bbox = self.bbox_head.loss(bbox_results['cls_score'], bbox_results['bbox_pred'], rois, *bbox_targets)
        return loss_bbox

    def load_clean_bboxes(self, img_metas, sampling_results):
        # 离线 FMC 结果已存为 pkl，这里按 img_id 索引返回 list[Tensor]
        import pickle, os
        pkl_dir = 'data/coco/fmc_correction'
        clean = []
        for res in sampling_results:
            img_id = res.img_id.item()
            pkl_path = os.path.join(pkl_dir, f'{img_id}.pkl')
            clean.append(pickle.load(open(pkl_path, 'rb')))
        return clean