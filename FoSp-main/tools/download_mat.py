"""Download the public Places-512 MAT checkpoint and verify its pinned SHA256."""

import argparse
import hashlib
from pathlib import Path
import time
import urllib.request


URL = ('https://huggingface.co/Icar/mat_places512_full/resolve/'
       '476b8ea5b3f82f1784cde1fa0578c710e4a323cd/Places_512_FullData.pkl')
SHA256 = 'd960c4e6b3266b6b9fa74ee4458a9482160d54c06d7738696bc9a9e2b34c66dc'
SIZE = 661420475
DEFAULT = Path(__file__).resolve().parents[1] / 'pretrained' / 'Places_512_FullData.pkl'


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT)
    args = parser.parse_args()
    target = args.output.resolve()
    if target.exists():
        if digest(target) == SHA256:
            print('Checkpoint already verified:', target)
            return
        raise RuntimeError('Existing checkpoint has a different checksum: ' + str(target))
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + '.part')
    offset = partial.stat().st_size if partial.exists() else 0
    request = urllib.request.Request(URL, headers={'User-Agent': 'FoSp-MAT-setup'})
    if offset:
        request.add_header('Range', 'bytes={}-'.format(offset))
    print('Downloading MAT (661 MB) from the pinned Hugging Face mirror...', flush=True)
    if offset < SIZE:
        with urllib.request.urlopen(request, timeout=90) as response:
            if response.status != 206:
                offset = 0
            elif not response.headers.get('Content-Range', '').startswith('bytes {}-'.format(offset)):
                raise RuntimeError('Unexpected download byte range.')
            last_report = time.monotonic()
            with partial.open('ab' if offset else 'wb') as stream:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    stream.write(chunk)
                    offset += len(chunk)
                    if time.monotonic() - last_report >= 15:
                        print('{:.1f}% ({:.0f} MB)'.format(offset / SIZE * 100, offset / 1e6), flush=True)
                        last_report = time.monotonic()
    if partial.stat().st_size != SIZE or digest(partial) != SHA256:
        raise RuntimeError('Download incomplete or SHA256 mismatch: ' + str(partial))
    partial.replace(target)
    print('SHA256 verified. Saved:', target)


if __name__ == '__main__':
    main()
