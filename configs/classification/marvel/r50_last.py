_base_ = '../../base.py'
# model settings
model = dict(
    type='Classification',
    pretrained=None,
    backbone=dict(
        type='ResNet',
        depth=50,
        out_indices=[4],  # 4: stage-4
        norm_cfg=dict(type='BN'),
        frozen_stages=4),
    head=dict(
        type='ClsHead', with_avg_pool=True, in_channels=2048, num_classes=26))
# dataset settings
data_source_cfg = dict(type='Marvel')
data_train_list = '/root/data/zq/data/marvel/140k/train.txt'
data_val_list = '/root/data/zq/data/marvel/140k/val.txt'
dataset_type = 'ClassificationDataset'
img_norm_cfg = dict(mean=[0.524, 0.553, 0.580], std=[0.242, 0.239, 0.250])
train_pipeline = [
    # dict(type='RandomCrop', size=32, padding=4),
    # dict(type='RandomResizedCrop', size=224, scale=(0.2, 1.)),
    dict(type='RandomHorizontalFlip'),
    dict(type='ToTensor'),
    dict(type='Normalize', **img_norm_cfg),
]
test_pipeline = [
    dict(type='ToTensor'),
    dict(type='Normalize', **img_norm_cfg),
]
data = dict(
    imgs_per_gpu=64,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        data_source=dict(list_file=data_train_list, **data_source_cfg),
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_source=dict(list_file=data_val_list, **data_source_cfg),
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_source=dict(list_file=data_val_list, **data_source_cfg),
        pipeline=test_pipeline))
# additional hooks
custom_hooks = [
    dict(
        type='ValidateHook',
        dataset=data['val'],
        initial=True,
        interval=2,
        imgs_per_gpu=64,
        workers_per_gpu=8,
        eval_param=dict(topk=(1,)))
]
# optimizer
optimizer = dict(type='SGD', lr=0.05, momentum=0.9, weight_decay=0.)
# learning policy
lr_config = dict(policy='step', step=[25, 30])
checkpoint_config = dict(interval=2)
# runtime settings
total_epochs = 35
