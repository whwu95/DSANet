import os

import datetime
import numpy as np

MODE = 'TRAIN'
DATASET = 'kinetics400/'
experiment_class = 'DSA/'
base_config_name = 'R50_E100'
other_config_name = '/base'

BASE_PATH = os.environ['HOME']
DATA_BASE_PATH = '/data/'

data_root = DATA_BASE_PATH + 'Kinetics400/train'
data_root_val = DATA_BASE_PATH + 'Kinetics400/val'
ANN_BASE_DIR = BASE_PATH + '/workspace/files/'
EXPERIMENT_WORK_DIR = BASE_PATH + '/workspace/experiment/'

load_from = None

T = datetime.datetime.now().strftime('%Y%m%d%H%M%S')


# model settings
model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='ResNet_I3D',
        pretrained='pretrained/resnet50.pth',
        pretrained2d=True,
        depth=50,
        num_stages=4,
        out_indices=[3],
        frozen_stages=-1,
        inflate_freq=(0, 0, 1, 1),
        inflate_style='3x1x1',
        conv1_kernel=(1, 7, 7),
        conv1_stride_t=1,
        pool1_kernel_t=1,
        pool1_stride_t=1,
        no_pool2=True,
        norm_eval=False,
        partial_norm=False,
        norm_cfg=dict(type='BN3d', requires_grad=True),
        style='pytorch',
        zero_init_residual=False,
        with_cp=False),
    cls_head=dict(
        type='I3DClsHead',
        spatial_type='avg',
        spatial_size=-1,
        temporal_size=-1,
        dropout_ratio=0.5,
        in_channels=2048,
        num_classes=400,
        init_std=0.01),
    module_cfg=dict(
        build_function='block_maker',
        type='Make_DSA',
        insert_freq=(0, (0, 1, 0, 1), (0, 1, 0, 1, 0, 1), 0),
        depth=50,
        insert_place='res_2',
        U_segment=4,
        alpha=2,
        split_ratio=1/8,
    )
)
train_cfg = None
test_cfg = None

if MODE == 'DEBUG':
    ann_file_train = ANN_BASE_DIR + DATASET + 'kinetics_debug_train_list.txt'
    ann_file_val = ANN_BASE_DIR + DATASET + 'kinetics_debug_val_list.txt'
    ann_file_test = ann_file_val
elif MODE == 'TRAIN':
    ann_file_train = ANN_BASE_DIR + DATASET + 'train_ffmpeg_fps30.txt'
    ann_file_val = ANN_BASE_DIR + DATASET + 'val_ffmpeg_fps30.txt'
    ann_file_test = ann_file_val
else:
    raise NotImplemented

dataset_type = 'RawFramesDataset'

train_pipeline = [
    dict(type='SampleFrames',
         clip_len=8,
         frame_interval=8,
         num_clips=4),
    dict(type='FrameSelector'),
    dict(
        type='RandomResizedCrop',
        input_size=224),
    dict(type='Flip', flip_ratio=0.5),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        div_255=False,
        to_rgb=True),
    dict(type='FormatShape',
         input_format='NCTHW'),
    dict(type='Collect',
         keys=['img_group', 'label', 'num_clips'],
         meta_keys=[]),
    dict(type='ToTensor',
         keys=['img_group', 'label', 'num_clips']),
]

val_pipeline = [
    dict(type='SampleFrames',
         clip_len=8,
         frame_interval=8,
         num_clips=4),
    dict(type='FrameSelector'),
    dict(type='Resize',
         scale=(np.Inf, 256),
         keep_ratio=True),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Flip', flip_ratio=0),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        div_255=False,
        to_rgb=True),
    dict(type='FormatShape',
         input_format='NCTHW'),
    dict(type='Collect',
         keys=['img_group', 'label', 'num_clips', 'num_crops'],
         meta_keys=[]),
    dict(type='ToTensor',
         keys=['img_group', 'num_clips', 'num_crops']),
]

test_pipeline = [
    dict(type='SampleFrames',
         clip_len=8,
         frame_interval=8,
         num_clips=8),
    dict(type='FrameSelector'),
    dict(type='Resize', scale=(np.Inf, 256), keep_ratio=True),
    dict(type='ThreeCrop', crop_size=256),
    dict(type='Flip', flip_ratio=0),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        div_255=False,
        to_rgb=True),
    dict(type='FormatShape',
         input_format='NCTHW'),
    dict(
        type='Collect',
        keys=['img_group', 'label', 'num_clips', 'num_crops'],
        meta_keys=[]),
    dict(type='ToTensor',
         keys=['img_group', 'num_clips', 'num_crops']),
]

data = dict(
    videos_per_gpu=12,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_root=data_root,
        pipeline=train_pipeline,
        test_mode=False,
        modality='RGB',
        filename_tmpl='img_{:05}.jpg'
    ),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_root=data_root_val,
        pipeline=val_pipeline,
        test_mode=True,
        modality='RGB',
        filename_tmpl='img_{:05}.jpg'
    ),
    test=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_root=data_root_val,
        pipeline=test_pipeline,
        test_mode=True,
        modality='RGB',
        filename_tmpl='img_{:05}.jpg'
    ))

# optimizer
optimizer = dict(type='SGD', lr=0.015, momentum=0.9,
                 weight_decay=0.0001, nesterov=True)
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    step=[50, 75, 90],
    warmup_ratio=0.01,
    warmup='linear',
    warmup_iters=2507*10)
checkpoint_config = dict(interval=1)
workflow = [('train', 1)]
log_config = dict(
    interval=20,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])

# runtime settings
total_epochs = 100
dist_params = dict(backend='nccl')
log_level = 'INFO'

if MODE == 'DEBUG':
    work_dir = BASE_PATH + '/debug'
else:
    work_dir = EXPERIMENT_WORK_DIR + DATASET + experiment_class + base_config_name + other_config_name

resume_from = None

if MODE == 'DEBUG':
    eval_interval = 1
else:
    eval_interval = 10

cudnn_benchmark = True
fp16_cfg = None
