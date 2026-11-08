"""Install the legacy FoSp stack without changing Kaggle's torch/CUDA pair.

Run inside a venv made with --system-site-packages. No CUDA compilation needed.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]


def pip(*args):
    subprocess.run([sys.executable, '-m', 'pip', '--disable-pip-version-check',
                    'install', '--no-cache-dir', *args], check=True)


def main():
    if sys.prefix == sys.base_prefix:
        raise SystemExit('Create/activate a --system-site-packages venv first; see KAGGLE.md.')
    import torch
    import torchvision
    print(f'Keeping torch={torch.__version__}, torchvision={torchvision.__version__}', flush=True)
    if not torch.cuda.is_available():
        raise SystemExit('Enable a GPU accelerator in Kaggle before setup.')
    pip('-r', str(ROOT / 'requirements/kaggle.txt'))
    # MMCV 1.6.2 sdist uses exec(...); locals() to read its version. That fails
    # under Python 3.13. Build the pure-Python wheel with an explicit namespace.
    with urlopen('https://pypi.org/pypi/mmcv/1.6.2/json', timeout=60) as response:
        metadata = json.load(response)
    source = next(item for item in metadata['urls'] if item['packagetype'] == 'sdist')
    with tempfile.TemporaryDirectory(prefix='fosp-mmcv-') as tmp:
        archive = Path(tmp) / 'mmcv.tar.gz'
        with urlopen(source['url'], timeout=60) as response:
            payload = response.read()
        if hashlib.sha256(payload).hexdigest() != source['digests']['sha256']:
            raise RuntimeError('MMCV source checksum mismatch')
        archive.write_bytes(payload)
        with tarfile.open(archive) as bundle:
            for member in bundle.getmembers():
                target = (Path(tmp) / member.name).resolve()
                if not target.is_relative_to(Path(tmp).resolve()) or member.issym() or member.islnk():
                    raise RuntimeError('Unsafe MMCV archive member')
            if hasattr(tarfile, 'data_filter'):
                bundle.extractall(tmp, filter='data')
            else:
                bundle.extractall(tmp)
        source_dir = Path(tmp) / 'mmcv-1.6.2'
        setup = source_dir / 'setup.py'
        text = setup.read_text(encoding='utf-8')
        start, end = text.index('def get_version():'), text.index('def parse_requirements(')
        text = text[:start] + '''def get_version():
    namespace = {}
    with open('mmcv/version.py', encoding='utf-8') as stream:
        exec(compile(stream.read(), 'mmcv/version.py', 'exec'), namespace)
    return namespace['__version__']


''' + text[end:]
        setup.write_text(text, encoding='utf-8')
        os.environ['MMCV_WITH_OPS'] = '0'
        pip('--use-pep517', '--no-build-isolation', '--no-deps', '--force-reinstall', str(source_dir))
    pip('--use-pep517', '--no-build-isolation', '--no-deps', '-e', str(ROOT))
    subprocess.run([sys.executable, '-c',
                    'import mmcv, mmseg, torch; from mmseg.models import build_segmentor; '
                    'print("Ready:", mmcv.__version__, torch.__version__, torch.cuda.get_device_name(0))'],
                   cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
