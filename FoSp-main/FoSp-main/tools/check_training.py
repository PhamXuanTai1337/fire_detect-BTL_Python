"""Exercise real training, validation, checkpoint save and resume on one image.

Uses the supplied config and pretrained weights. This verifies execution only,
not model quality. Output checkpoints contain full MAT and may occupy several GB.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mmcv import Config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', nargs='?', default='config/fosp_kaggle.py')
    parser.add_argument('--work-dir', default='work_dirs/train_check')
    parser.add_argument('--skip-backbone-pretrain', action='store_true',
                        help='Wiring test only: avoid downloading MiT; MAT is still required.')
    args = parser.parse_args()
    cfg = Config.fromfile(args.config)
    work = Path(args.work_dir).resolve()
    work.mkdir(parents=True, exist_ok=True)
    if (work / 'report.json').exists():
        raise SystemExit('Choose a fresh --work-dir for this check.')
    for split in ('train', 'val'):
        spec = cfg.data[split]
        source = Path(spec.data_root) / spec.split
        names = [line.strip() for line in source.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
        if not names:
            raise ValueError(f'Empty split: {source}')
        target = work / (split + '.txt')
        target.write_text(names[0] + '\n', encoding='utf-8')
        spec.split = target.as_posix()
    cfg.work_dir = work.as_posix()
    cfg.data.samples_per_gpu = 1
    cfg.data.workers_per_gpu = 0
    cfg.runner.max_iters = 1
    cfg.lr_config.warmup = None
    cfg.log_config.interval = 1
    cfg.checkpoint_config.update(interval=1, max_keep_ckpts=1, create_symlink=False)
    cfg.evaluation.update(interval=1, pre_eval=True, metric=['mIoU', 'mFscore'],
                          save_best='IoU.smoke', rule='greater')
    cfg.model.decode_head.mat_impl = 'ref'
    cfg.load_from = cfg.resume_from = None
    if args.skip_backbone_pretrain:
        cfg.model.pretrained = None
    config_path = work / 'check_config.py'
    cfg.dump(str(config_path))
    command = [sys.executable, str(ROOT / 'tools/train.py'), str(config_path)]
    subprocess.run(command, cwd=ROOT, check=True)
    first = work / 'iter_1.pth'
    if not first.is_file() or not list(work.glob('best_IoU.smoke_*.pth')):
        raise RuntimeError('Expected iteration and best validation checkpoints were not saved')
    subprocess.run(command + ['--resume-from', first.as_posix(), '--cfg-options',
                              'runner.max_iters=2'], cwd=ROOT, check=True)
    second = work / 'iter_2.pth'
    import torch
    state = torch.load(second, map_location='cpu', weights_only=False)
    assert state['meta']['iter'] == 2
    optimizer_states = state['optimizer']['state'].values()
    steps = [float(item['step']) for item in optimizer_states if 'step' in item]
    assert steps and min(steps) == max(steps) == 2, 'Optimizer did not resume from iteration 1'
    report = dict(passed=True, iterations=2, optimizer_resumed=True,
                  validation_and_best_checkpoint=True,
                  backbone_pretrained=not args.skip_backbone_pretrain,
                  mat_pretrained=True, mat_impl='ref',
                  torch=str(torch.__version__), cuda=torch.version.cuda,
                  gpu=torch.cuda.get_device_name(0))
    (work / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
