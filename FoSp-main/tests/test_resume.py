import logging

import torch

from mmseg.core.runner import FoSpIterBasedRunner
from mmseg.utils.misc import find_latest_checkpoint


def test_auto_resume_ignores_older_best_checkpoint(tmp_path):
    for name in ('best_IoU.smoke_iter_1000.pth', 'iter_1000.pth', 'iter_2000.pth'):
        (tmp_path / name).touch()
    assert find_latest_checkpoint(str(tmp_path)) == str(tmp_path / 'iter_2000.pth')


class TinyModel(torch.nn.Linear):
    def train_step(self, *args, **kwargs):
        raise NotImplementedError


def test_checkpoint_records_completed_step_and_restores_optimizer(tmp_path):
    model = TinyModel(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    runner = FoSpIterBasedRunner(model=model, optimizer=optimizer,
                                work_dir=str(tmp_path), max_iters=2,
                                logger=logging.getLogger('resume-test'),
                                meta={'hook_msgs': {'best_score': 0.75}})
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    runner.save_checkpoint(str(tmp_path))
    checkpoint = tmp_path / 'iter_1.pth'
    state = torch.load(checkpoint, weights_only=False)
    assert state['meta']['iter'] == 1

    restored = TinyModel(2, 1)
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=0.01)
    resumed = FoSpIterBasedRunner(model=restored, optimizer=restored_optimizer,
                                 work_dir=str(tmp_path), max_iters=2,
                                 logger=logging.getLogger('resume-test'), meta={})
    resumed.resume(str(checkpoint))
    assert resumed.iter == 1
    assert resumed.meta['hook_msgs']['best_score'] == 0.75
    torch.testing.assert_close(model.weight, restored.weight)
    restored(torch.ones(1, 2)).sum().backward()
    restored_optimizer.step()
    resumed.save_checkpoint(str(tmp_path))
    final = torch.load(tmp_path / 'iter_2.pth', weights_only=False)
    assert final['meta']['iter'] == 2
    assert all(float(item['step']) == 2 for item in final['optimizer']['state'].values())
