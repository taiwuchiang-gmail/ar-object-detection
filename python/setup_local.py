"""Configure a fresh clone using existing assets, without modifying source folders."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--matlab-root', type=Path, required=True)
    parser.add_argument('--python-root', type=Path, required=True)
    parser.add_argument('--dataset-root', type=Path)
    parser.add_argument('--rtsp-url', default='')
    args = parser.parse_args(argv)
    for folder in [args.matlab_root, args.python_root]:
        if not folder.is_dir():
            parser.error(f'Source directory does not exist: {folder}')
    config_path = ROOT/'configs/project.json'
    if config_path.exists():
        parser.error('configs/project.json exists; edit it directly rather than overwrite it')
    cfg = json.loads((ROOT/'configs/project.example.json').read_text(encoding='utf-8'))
    cfg.update(original_matlab_root=args.matlab_root.resolve().as_posix(),
               original_python_root=args.python_root.resolve().as_posix(),
               external_datasets=args.dataset_root.resolve().as_posix() if args.dataset_root else '',
               rtsp_url=args.rtsp_url)
    assets = []
    for obj in cfg['objects'].values():
        for key in ['ground_truth', 'matlab_model']:
            relative = obj[key]
            assets.append((args.matlab_root/Path(relative).name, ROOT/relative))
    for source, destination in [
        ('runs/detect/train2/weights/best.pt', 'duck_train2.pt'),
        ('runs_bike/detect/train5/weights/best.pt', 'bike_train5.pt'),
        ('yolov8n.pt', 'yolov8n.pt'), ('yolov8s.pt', 'yolov8s.pt')]:
        assets.append((args.python_root/source, ROOT/'models/python'/destination))
    # Refuse conflicting destinations before copying anything.
    for source, destination in assets:
        if source.is_file() and destination.exists():
            if hashlib.sha256(source.read_bytes()).digest() != hashlib.sha256(destination.read_bytes()).digest():
                parser.error(f'Destination differs from source: {destination}')
    copied, missing = [], []
    for source, destination in assets:
        if not source.is_file():
            missing.append(str(source))
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            shutil.copy2(source, destination)
        copied.append({'source':str(source.resolve()), 'destination':str(destination.relative_to(ROOT)),
                       'sha256':hashlib.sha256(destination.read_bytes()).hexdigest()})
    config_path.write_text(json.dumps(cfg, indent=2)+'\n', encoding='utf-8')
    local = ROOT/'docs/local'
    local.mkdir(parents=True, exist_ok=True)
    (local/'asset-import.json').write_text(json.dumps({'assets':copied,'missing':missing}, indent=2)+'\n')
    print(f'Created {config_path}; {len(copied)} assets available')
    for source in missing:
        print(f'Missing optional asset: {source}')

if __name__ == '__main__':
    main()
