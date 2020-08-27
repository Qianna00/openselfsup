_base_ = '../../base.py'
# model settings
model = dict(
    type='MOCO',
    pretrained=None,
    queue_len=65536,
    feat_dim=128,
    momentum=0.999,
    backbone=dict(
        type='ResNet',
        depth=50,
        in_channels=3,
        out_indices=[4],  # 0: conv-1, x: stage-x
        norm_cfg=dict(type='BN')),
    neck=dict(
        type='LinearNeck',
        in_channels=2048,
        out_channels=128,
        with_avg_pool=True),
    head=dict(type='ContrastiveHead', temperature=0.07))
# dataset settings
data_source_cfg = dict(
    type='Marvel')
data_train_list = '/root/data/zq/data/marvel/data_400k/final_400k_new.txt'
# data_train_root = 'data/imagenet/train'
dataset_type = 'ContrastiveDataset'
img_norm_cfg = dict(mean=[0.524, 0.553, 0.580], std=[0.242, 0.239, 0.250])
train_pipeline = [
    dict(type='RandomResizedCrop', size=224, scale=(0.2, 1.)),
    dict(type='RandomGrayscale', p=0.2),
    dict(
        type='ColorJitter',
        brightness=0.4,
        contrast=0.4,
        saturation=0.4,
        hue=0.4),
    dict(type='RandomHorizontalFlip'),
    dict(type='ToTensor'),
    dict(type='Normalize', **img_norm_cfg),
]
data = dict(
    imgs_per_gpu=128,  # total 32*8=256
    workers_per_gpu=8,
    drop_last=True,
    train=dict(
        type=dataset_type,
        data_source=dict(
            list_file=data_train_list, **data_source_cfg),
        pipeline=train_pipeline))
# optimizer
optimizer = dict(type='SGD', lr=0.03, weight_decay=0.0001, momentum=0.9)
# learning policy
lr_config = dict(policy='step', step=[60, 75, 90])
checkpoint_config = dict(interval=2)
# runtime settings
total_epochs = 100
