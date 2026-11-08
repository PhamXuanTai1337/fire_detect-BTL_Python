"""Create a source-only upload archive; never include data or checkpoints."""
import argparse
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DIRECTORIES = ('config', 'mmseg', 'mat_inpainter', 'tools', 'requirements', 'tests', 'imgs')
FILES = ('.gitignore', 'setup.py', 'pyproject.toml', 'requirements.txt', 'README.md', 'FULL_MAT.md',
         'KAGGLE.md', 'LICENSE', 'kaggle_train.ipynb')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT.parent / 'fosp_mat_github.zip')
    args = parser.parse_args()
    paths = [ROOT / name for name in FILES]
    for name in DIRECTORIES:
        paths.extend(path for path in (ROOT / name).rglob('*') if path.is_file()
                     and not {'__pycache__', '.pytest_cache', '.mim'}.intersection(path.parts)
                     and path.suffix not in {'.pyc', '.pkl', '.pth', '.pt', '.ckpt', '.log'})
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size >= 50 * 1024 * 1024:
            raise ValueError(f'Unexpected large source file: {path}')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(set(paths)):
            archive.write(path, path.relative_to(ROOT).as_posix())
    print(f'{args.output.resolve()}: {len(set(paths))} files, '
          f'{args.output.stat().st_size / 1024**2:.2f} MiB')


if __name__ == '__main__':
    main()
