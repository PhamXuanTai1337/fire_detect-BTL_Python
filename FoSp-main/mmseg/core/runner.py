"""Iteration checkpoints with an unambiguous completed-step count."""
import os
import os.path as osp
import shutil

import mmcv
from mmcv.runner import RUNNERS, IterBasedRunner, save_checkpoint


@RUNNERS.register_module()
class FoSpIterBasedRunner(IterBasedRunner):
    def save_checkpoint(self, out_dir, filename_tmpl='iter_{}.pth', meta=None,
                        save_optimizer=True, create_symlink=False):
        # after_train_iter runs AFTER optimizer.step but BEFORE _iter += 1.
        # MMCV 1.6.2 saves _iter, which repeats a completed step on resume.
        completed = self.iter + 1
        metadata = dict(meta or {})
        metadata.update(self.meta or {})
        metadata.update(epoch=self.epoch + 1, iter=completed,
                        fosp_checkpoint_version=1)
        filename = filename_tmpl.format(completed)
        filepath = osp.join(out_dir, filename)
        save_checkpoint(self.model, filepath,
                        optimizer=self.optimizer if save_optimizer else None,
                        meta=metadata)
        if create_symlink:
            # Keep compatibility with callers requesting latest.pth.
            if os.name == 'nt':
                shutil.copyfile(filepath, osp.join(out_dir, 'latest.pth'))
            else:
                mmcv.symlink(filename, osp.join(out_dir, 'latest.pth'))

    def resume(self, checkpoint, resume_optimizer=True, map_location='cpu'):
        state = self.load_checkpoint(checkpoint, map_location=map_location)
        metadata = state['meta']
        self._epoch = metadata['epoch']
        self._iter = metadata['iter']
        if metadata.get('fosp_checkpoint_version') != 1:
            # Checkpoints written by upstream MMCV 1.6.2 record zero-based step.
            self._iter += 1
        self._inner_iter = self._iter
        if self.meta is None:
            self.meta = {}
        self.meta.update(metadata)
        if resume_optimizer:
            if 'optimizer' not in state:
                raise ValueError('Checkpoint has no optimizer; use --load-from for weights only.')
            self.optimizer.load_state_dict(state['optimizer'])
        self.logger.info(f'Resumed after {self.iter} completed iterations')
