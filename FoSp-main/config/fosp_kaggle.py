"""Single GPU, full two-stage MAT; paths can be set in notebook os.environ."""
import os

_base_ = ['./fosp_config.py']

data_root = os.environ.get('SMOKE_DATA_ROOT', '/kaggle/input/smokeseg/SmokeSeg')
split_dir = os.environ.get('SMOKE_SPLIT_DIR', 'ImageSets/Segmentation')
mat_checkpoint = os.environ.get('MAT_CHECKPOINT', 'pretrained/Places_512_FullData.pkl')
backbone_checkpoint = os.environ.get(
    'MIT_CHECKPOINT',
    'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b3_20220624-13b1141c.pth')

model = dict(pretrained=backbone_checkpoint,
             decode_head=dict(mat_checkpoint=mat_checkpoint, mat_impl='ref'))
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
    train=dict(data_root=data_root, split=split_dir + '/train.txt'),
    val=dict(data_root=data_root, split=split_dir + '/val.txt'),
    test=dict(data_root=data_root, split=split_dir + '/test.txt'))
work_dir = '/kaggle/working/fosp_runs'
checkpoint_config = dict(by_epoch=False, interval=1000, max_keep_ckpts=2,
                         create_symlink=False)
evaluation = dict(interval=1000, metric=['mIoU', 'mFscore'], pre_eval=True,
                  save_best='IoU.smoke', rule='greater')
cudnn_benchmark = False
