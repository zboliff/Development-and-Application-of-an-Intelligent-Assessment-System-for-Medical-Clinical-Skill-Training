_base_ = ['../../mmaction2/configs/recognition/tsm/tsm_imagenet-pretrained-r50_8xb16-1x1x8-50e_sthv2-rgb.py']

model = dict(
    backbone=dict(num_segments=16),
)

file_client_args = dict(io_backend='disk')

sthv2_flip_label_map = {86: 87, 87: 86, 93: 94, 94: 93, 166: 167, 167: 166}
