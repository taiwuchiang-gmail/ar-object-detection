"""Versioned YOLO training, inference, and candidate-label generation.
Relative paths resolve from the project root, not the working directory.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]

def resolve(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path

def load_config():
    config_path = ROOT / 'configs/project.json'
    if not config_path.is_file():
        config_path = ROOT / 'configs/project.example.json'
    return json.loads(config_path.read_text(encoding='utf-8'))

def split_dataset(manifest_path, output, object_name):
    """Assign entire recordings to splits; reject exact image leakage."""
    items = json.loads(resolve(manifest_path).read_text(encoding='utf-8'))
    groups = {s: [] for s in ('train', 'val', 'test')}
    seen, hashes, planned = set(), {}, []
    for item in items:
        split = item['split']
        if split not in groups:
            raise ValueError('split must be train, val, or test')
        folder = resolve(item['directory']).resolve()
        if folder in seen:
            raise ValueError(f'Duplicate recording directory: {folder}')
        seen.add(folder)
        images = sorted(p for p in (folder/'images').glob('*') if p.suffix.lower() in {'.png','.jpg','.jpeg'})
        if not images:
            raise ValueError(f'No images in {folder}')
        for image in images:
            label = folder/'labels'/(image.stem+'.txt')
            if not label.is_file():
                raise FileNotFoundError(label)
            digest = hashlib.sha256(image.read_bytes()).hexdigest()
            if digest in hashes and hashes[digest] != split:
                raise ValueError(f'Identical image crosses splits: {image}')
            hashes[digest] = split
            for line in label.read_text().splitlines():
                parts=line.split()
                if len(parts)!=5 or parts[0]!='0':
                    raise ValueError(f'Expected single-class YOLO label: {label}')
                x,y,w,h=map(float,parts[1:])
                if not (0<=x<=1 and 0<=y<=1 and 0<w<=1 and 0<h<=1):
                    raise ValueError(f'Invalid coordinates: {label}')
            groups[split].append(image)
            planned.append((split,folder,image,label))
    if not groups['train'] or not groups['val']:
        raise ValueError('Separate train and validation recordings are required')
    out=resolve(output)
    out.mkdir(parents=True,exist_ok=False)
    for split,folder,image,label in planned:
        prefix=hashlib.sha256(str(folder).encode()).hexdigest()[:12]
        for kind,src in [('images',image),('labels',label)]:
            dest=out/kind/split/(prefix+'_'+src.name)
            dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(src,dest)
    data={'path':out.resolve().as_posix(),'train':'images/train','val':'images/val','names':{0:object_name}}
    if groups['test']: data['test']='images/test'
    (out/'dataset.yaml').write_text(json.dumps(data,indent=2))
    (out/'split-manifest.json').write_text(json.dumps(items,indent=2))
    # Preserve the source candidate provenance alongside the assembled dataset.
    for item in items:
        folder=resolve(item['directory']).resolve()
        provenance=folder/'provenance.jsonl'
        if provenance.is_file():
            dest=out/'provenance'
            dest.mkdir(exist_ok=True)
            shutil.copy2(provenance,dest/(hashlib.sha256(str(folder).encode()).hexdigest()[:12]+'.jsonl'))
    print(out/'dataset.yaml')
    return out

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    split=sub.add_parser('split')
    split.add_argument('--manifest',required=True)
    split.add_argument('--output',required=True)
    split.add_argument('--object',required=True,choices=['bike','duck','laser'])
    for command in ['train','predict','pseudo-label']:
        cmd=sub.add_parser(command)
        cmd.add_argument('--object',required=True,choices=['bike','duck','laser'])
        cmd.add_argument('--model',help='Checkpoint; defaults to configured object model')
        cmd.add_argument('--name',required=True,help='New run or iteration name')
        cmd.add_argument('--imgsz',type=int,default=640)
        if command=='train':
            cmd.add_argument('--data',required=True)
            cmd.add_argument('--epochs',type=int,default=50)
        else:
            cmd.add_argument('--source',required=True,help='File, RTSP URL, camera number, or glasses')
            cmd.add_argument('--confidence',type=float,default=.7)
            cmd.add_argument('--show',action='store_true')
    args=parser.parse_args(argv)
    if args.command=='split':
        return split_dataset(args.manifest,args.output,args.object)
    if not args.name or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in args.name):
        parser.error('Use letters, digits, underscores and hyphens for --name')
    cfg=load_config()
    model_value=args.model or cfg['objects'][args.object]['python_model']
    if not model_value: parser.error('No configured checkpoint; supply --model')
    model_path=resolve(model_value)
    if not model_path.is_file(): parser.error(f'Model missing: {model_path}; supply --model')
    out=(ROOT/'datasets'/args.object/args.name if args.command=='pseudo-label' else ROOT/'runs'/'python'/args.name)
    if out.exists(): raise FileExistsError(out)
    if args.command=='train':
        data=resolve(args.data)
        if not data.is_file(): parser.error(f'Dataset YAML missing: {data}')
    else:
        if not 0<=args.confidence<=1: parser.error('Confidence must be between zero and one')
        source=cfg['rtsp_url'] if args.source=='glasses' else args.source
        if not source:
            parser.error('Set rtsp_url in configs/project.json or supply --source')
        if source.isdigit(): source=int(source)
        elif '://' not in source:
            source=str(resolve(source))
            if not Path(source).exists(): parser.error(f'Source missing: {source}')
    from ultralytics import YOLO
    model=YOLO(str(model_path))
    if args.command=='train':
        model.train(data=str(data),epochs=args.epochs,imgsz=args.imgsz,single_cls=True,
                    project=str(out.parent),name=out.name,exist_ok=False)
    elif args.command=='predict':
        for _ in model.predict(source=source,stream=True,conf=args.confidence,
                imgsz=args.imgsz,show=args.show,save=True,save_txt=True,
                project=str(out.parent),name=out.name,exist_ok=False,max_det=1):
            pass
    else:
        import cv2
        out.mkdir(parents=True,exist_ok=False)
        (out/'images').mkdir(); (out/'labels').mkdir()
        with (out/'provenance.jsonl').open('w',encoding='utf-8') as log:
            for index,result in enumerate(model.predict(source=source,stream=True,
                    conf=args.confidence,imgsz=args.imgsz,show=args.show,max_det=1),1):
                if result.boxes is None or len(result.boxes)==0: continue
                best=int(result.boxes.conf.argmax().item())
                box=result.boxes[best]
                coords=box.xywhn[0].cpu().tolist()
                name=f'frame_{index:08d}'
                if not cv2.imwrite(str(out/'images'/(name+'.jpg')),result.orig_img):
                    raise OSError('Failed writing image')
                (out/'labels'/(name+'.txt')).write_text('0 '+' '.join(f'{x:.8f}' for x in coords)+'\n')
                log.write(json.dumps({'image':name+'.jpg','source':str(source),
                    'result_index':index,'model':str(model_path),'confidence':float(box.conf.item()),
                    'threshold':args.confidence,'object':args.object,'iteration':args.name,
                    'label_origin':'yolov8_prediction'})+'\n')
    if out.is_dir():
        (out/'run.json').write_text(json.dumps(vars(args)|{'resolved_model':str(model_path)},indent=2))

if __name__=='__main__':
    main()
