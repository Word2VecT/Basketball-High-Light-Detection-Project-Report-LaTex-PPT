auto_scale_lr = dict(base_batch_size=1024)
backend_args = dict(backend='local')
codec = dict(
    num_keypoints=17,
    remove_root=False,
    root_index=0,
    type='VideoPoseLifting',
    zero_center=True)
custom_hooks = [
    dict(type='SyncBuffersHook'),
]
data_root = 'data/h36m/'
dataset_type = 'Human36mDataset'
default_hooks = dict(
    badcase=dict(
        badcase_thr=5,
        enable=False,
        metric_type='loss',
        out_dir='badcase',
        type='BadCaseAnalysisHook'),
    checkpoint=dict(
        interval=10,
        max_keep_ckpts=1,
        rule='less',
        save_best='MPJPE',
        type='CheckpointHook'),
    logger=dict(interval=20, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(enable=False, type='PoseVisualizationHook'))
default_scope = 'mmpose'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
load_from = None
log_level = 'INFO'
log_processor = dict(
    by_epoch=True, num_digits=6, type='LogProcessor', window_size=50)
model = dict(
    backbone=dict(
        dropout=0.25,
        in_channels=34,
        kernel_sizes=(
            3,
            3,
            3,
        ),
        num_blocks=2,
        stem_channels=1024,
        type='TCN',
        use_stride_conv=True),
    head=dict(
        decoder=dict(
            num_keypoints=17,
            remove_root=False,
            root_index=0,
            type='VideoPoseLifting',
            zero_center=True),
        in_channels=1024,
        loss=dict(type='MPJPELoss'),
        num_joints=17,
        type='TemporalRegressionHead'),
    type='PoseLifter')
optim_wrapper = dict(optimizer=dict(lr=0.001, type='Adam'))
param_scheduler = [
    dict(by_epoch=True, end=80, gamma=0.975, type='ExponentialLR'),
]
resume = False
test_cfg = dict()
test_dataloader = dict(
    batch_size=128,
    dataset=dict(
        ann_file='annotation_body3d/fps50/h36m_test.npz',
        camera_param_file='annotation_body3d/cameras.pkl',
        causal=False,
        data_prefix=dict(img='images/'),
        data_root='data/h36m/',
        pad_video_seq=True,
        pipeline=[
            dict(
                encoder=dict(
                    num_keypoints=17,
                    remove_root=False,
                    root_index=0,
                    type='VideoPoseLifting',
                    zero_center=True),
                type='GenerateTarget'),
            dict(
                meta_keys=(
                    'id',
                    'category_id',
                    'target_img_path',
                    'flip_indices',
                    'target_root',
                ),
                type='PackPoseInputs'),
        ],
        seq_len=27,
        test_mode=True,
        type='Human36mDataset'),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(round_up=False, shuffle=False, type='DefaultSampler'))
test_evaluator = [
    dict(mode='mpjpe', type='MPJPE'),
    dict(mode='p-mpjpe', type='MPJPE'),
]
train_cfg = dict(by_epoch=True, max_epochs=160, val_interval=10)
train_dataloader = dict(
    batch_size=128,
    dataset=dict(
        ann_file='annotation_body3d/fps50/h36m_train.npz',
        camera_param_file='annotation_body3d/cameras.pkl',
        causal=False,
        data_prefix=dict(img='images/'),
        data_root='data/h36m/',
        pad_video_seq=True,
        pipeline=[
            dict(
                keypoints_flip_cfg=dict(),
                target_flip_cfg=dict(),
                type='RandomFlipAroundRoot'),
            dict(
                encoder=dict(
                    num_keypoints=17,
                    remove_root=False,
                    root_index=0,
                    type='VideoPoseLifting',
                    zero_center=True),
                type='GenerateTarget'),
            dict(
                meta_keys=(
                    'id',
                    'category_id',
                    'target_img_path',
                    'flip_indices',
                    'target_root',
                ),
                type='PackPoseInputs'),
        ],
        seq_len=27,
        type='Human36mDataset'),
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(
        keypoints_flip_cfg=dict(),
        target_flip_cfg=dict(),
        type='RandomFlipAroundRoot'),
    dict(
        encoder=dict(
            num_keypoints=17,
            remove_root=False,
            root_index=0,
            type='VideoPoseLifting',
            zero_center=True),
        type='GenerateTarget'),
    dict(
        meta_keys=(
            'id',
            'category_id',
            'target_img_path',
            'flip_indices',
            'target_root',
        ),
        type='PackPoseInputs'),
]
val_cfg = dict()
val_dataloader = dict(
    batch_size=128,
    dataset=dict(
        ann_file='annotation_body3d/fps50/h36m_test.npz',
        camera_param_file='annotation_body3d/cameras.pkl',
        causal=False,
        data_prefix=dict(img='images/'),
        data_root='data/h36m/',
        pad_video_seq=True,
        pipeline=[
            dict(
                encoder=dict(
                    num_keypoints=17,
                    remove_root=False,
                    root_index=0,
                    type='VideoPoseLifting',
                    zero_center=True),
                type='GenerateTarget'),
            dict(
                meta_keys=(
                    'id',
                    'category_id',
                    'target_img_path',
                    'flip_indices',
                    'target_root',
                ),
                type='PackPoseInputs'),
        ],
        seq_len=27,
        test_mode=True,
        type='Human36mDataset'),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(round_up=False, shuffle=False, type='DefaultSampler'))
val_evaluator = [
    dict(mode='mpjpe', type='MPJPE'),
    dict(mode='p-mpjpe', type='MPJPE'),
]
val_pipeline = [
    dict(
        encoder=dict(
            num_keypoints=17,
            remove_root=False,
            root_index=0,
            type='VideoPoseLifting',
            zero_center=True),
        type='GenerateTarget'),
    dict(
        meta_keys=(
            'id',
            'category_id',
            'target_img_path',
            'flip_indices',
            'target_root',
        ),
        type='PackPoseInputs'),
]
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='Pose3dLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
