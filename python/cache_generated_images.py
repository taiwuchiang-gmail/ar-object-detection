"""Create Ultralytics-compatible decoded-image caches for a generated run only."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import argparse,os,json,time

def cache_dataset(folder):
    import cv2,numpy as np
    images=sorted((folder/'images').rglob('*.png'))
    def cache(image_path):
        dest=image_path.with_suffix('.npy')
        if dest.exists(): return 0
        image=cv2.imread(str(image_path))
        if image is None: raise ValueError(f'Cannot read {image_path}')
        temp=dest.with_suffix(f'.tmp-{os.getpid()}')
        with temp.open('wb') as handle: np.save(handle,image)
        os.replace(temp,dest)
        return 1
    with ThreadPoolExecutor(max_workers=4) as pool:
        created=sum(pool.map(cache,images))
    if images:
        assert np.array_equal(cv2.imread(str(images[0])),np.load(images[0].with_suffix('.npy')))
    return len(images),created

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id',default='scratch_20261026_01')
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    run=root/'runs/duck'/args.run_id
    for stage in [1,2]:
        folder=root/'datasets/duck'/f'{args.run_id}_assembled_{stage:02d}'
        while not (folder/'dataset.yaml').is_file():
            status_file=run/'python_status.json'
            if status_file.is_file() and json.loads(status_file.read_text()).get('stage') in {'failed','complete'}:
                return
            time.sleep(10)
        if not folder.resolve().is_relative_to((root/'datasets').resolve()):
            raise ValueError('Cache path must remain under generated datasets')
        total,created=cache_dataset(folder)
        print(f'Iteration {stage}: {created} caches created for {total} PNG images',flush=True)
        (run/f'image_cache_{stage:02d}.json').write_text(json.dumps({'images':total,'created':created,
            'format':'Ultralytics native raw BGR uint8 .npy','source_pixels_preserved':True},indent=2))

if __name__=='__main__': main()
