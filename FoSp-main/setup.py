# Copyright (c) OpenMMLab. All rights reserved.
from pathlib import Path
from setuptools import find_packages, setup

ROOT = Path(__file__).resolve().parent
namespace = {}
exec((ROOT / 'mmseg/version.py').read_text(encoding='utf-8'), namespace)
requirements = [line.strip() for line in
                (ROOT / 'requirements/runtime.txt').read_text().splitlines()
                if line.strip() and not line.startswith('#')]
setup(
    name='fosp-full-mat',
    version=namespace['__version__'],
    description='FoSp with the full pretrained two-stage MAT inpainter',
    packages=find_packages(include=['mmseg', 'mmseg.*', 'mat_inpainter', 'mat_inpainter.*']),
    package_data={'mat_inpainter': ['LICENSE', 'torch_utils/ops/*.cpp',
                                  'torch_utils/ops/*.cu', 'torch_utils/ops/*.h']},
    install_requires=requirements,
    extras_require={'tests': ['pytest']},
    python_requires='>=3.10',
    zip_safe=False,
)
