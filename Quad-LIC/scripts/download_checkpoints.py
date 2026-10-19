import argparse
import json
import tempfile
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / 'checkpoints/downloads.json').read_text())
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', nargs='+', choices=[f['file'] for f in manifest['files']])
    parser.add_argument('--root', type=Path, default=root / 'checkpoints')
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    for entry in manifest['files']:
        if args.file and entry['file'] not in args.file:
            continue
        target = args.root / entry['file']
        if target.is_file() and target.stat().st_size == entry['bytes']:
            print('Found complete file', entry['file'], flush=True)
            continue
        if not entry.get('url'):
            raise RuntimeError(f"{entry['file']}: Drive upload or public sharing is pending")
        import gdown

        with tempfile.TemporaryDirectory(dir=args.root) as tmp:
            downloaded = Path(tmp) / entry['file']
            result = gdown.download(
                url=entry['url'], output=str(downloaded), quiet=False, fuzzy=True
            )
            if result is None or not downloaded.is_file() or downloaded.stat().st_size != entry['bytes']:
                raise RuntimeError('Invalid download: ' + entry['file'])
            downloaded.replace(target)
        print('Downloaded and size checked', entry['file'], flush=True)


if __name__ == '__main__':
    main()
