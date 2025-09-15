_base_ = [
    '../fmgdet/models/roi_head.py',
    '../fmgdet/datasets/noisy_coco.py'
]
model = dict(
    type='FasterRCNN',
    backbone=dict(type='ResNet', depth=50, num_stages=4, out_indices=(0,1,2,3), frozen_stages=1, norm_cfg=dict(type='BN', requires_grad=True), norm_eval=True, style='pytorch'),
    neck=dict(type='FPN', in_channels=[256,512,1024,2048], out_channels=256, num_outs=5),
    roi_head=dict(type='InterpRoIHead', bbox_roi_extractor=dict(type='SingleRoIExtractor', roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0), out_channels=256, featmap_strides=[4,8,16,32]), bbox_head=dict(type='Shared2FCBBoxHead', in_channels=256, fc_out_channels=1024, roi_feat_size=7, num_classes=80)),
    train_cfg=dict(rcnn=dict(assigner=dict(type='MaxIoUAssigner', pos_iou_thr=0.5, neg_iou_thr=0.5, min_pos_iou=0.5), sampler=dict(type='RandomSampler', num=512, pos_fraction=0.25), pos_weight=-1, debug=False)),
    test_cfg=dict(rcnn=dict(score_thr=0.05, nms=dict(type='nms', iou_thr=0.5), max_per_img=100))
)