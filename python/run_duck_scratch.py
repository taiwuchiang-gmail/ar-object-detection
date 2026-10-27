"""Run both fresh YOLOv8 stages after duckScratchCandidates in MATLAB.

Only regenerated labels and new checkpoints are used. The existing Python
installation is left unchanged; compatibility settings are process-local.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parents[1]

def write_json(path,data):
    path.write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id',default='scratch_20261026_01')
    parser.add_argument('--epochs',type=int,default=50)
    args=parser.parse_args()
    if any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in args.run_id):
        parser.error('Invalid run ID')
    run=ROOT/'runs/duck'/args.run_id
    run.mkdir(parents=True,exist_ok=True)
    os.environ['YOLO_CONFIG_DIR']=str(run/'ultralytics-settings')
    os.environ['MPLCONFIGDIR']=str(run/'matplotlib-cache')
    os.environ['OMP_NUM_THREADS']='6'
    os.environ['WANDB_DISABLED']='true'
    # Reuse the installed Windows font rather than downloading one for plots.
    import shutil
    settings=run/'ultralytics-settings'; settings.mkdir(exist_ok=True)
    system_font=Path('C:/Windows/Fonts/arial.ttf')
    if system_font.is_file() and not (settings/'Arial.ttf').exists():
        shutil.copy2(system_font,settings/'Arial.ttf')
    # A generic supplied checkpoint is used only by Ultralytics' AMP parity check.
    # Training itself starts from architecture YAML with pretrained=False.
    generic=ROOT/'models/python/yolov8n.pt'
    if generic.is_file() and not (run/'yolov8n.pt').exists():
        shutil.copy2(generic,run/'yolov8n.pt')
    os.chdir(run)
    import cv2
    import gc
    import torch
    import ultralytics
    from ultralytics import YOLO
    from pipeline import split_dataset,load_config
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for this full training run')
    # Ultralytics 8.1 predates PyTorch 2.6's changed torch.load default.
    # This process loads its own checkpoints and the supplied generic AMP-check model.
    original_load=torch.load
    def load_local_checkpoint(*a,**kw):
        kw.setdefault('weights_only',False)
        return original_load(*a,**kw)
    torch.load=load_local_checkpoint
    torch.set_num_threads(6)
    cfg=load_config()
    started=time.time()
    status_path=run/'python_status.json'
    summary={'run_id':args.run_id,'manual_annotations_used':False,
             'historical_duck_models_used':False,'torch':torch.__version__,
             'ultralytics':ultralytics.__version__,'gpu':torch.cuda.get_device_name(0),
             'epochs_per_stage':args.epochs,'seed':17,'imgsz':640,'batch':8,
             'initialization':'YOLOv8n architecture YAML, random weights',
             'compatibility':'Process-local torch.load weights_only=False for own checkpoints'}
    write_json(run/'python_environment.json',summary)
    def status(stage,**extra):
        write_json(status_path,{'stage':stage,'elapsed_seconds':time.time()-started,**extra})
        print('STAGE:',stage,flush=True)
    def train(model,name,dataset):
        out=run/'yolov8'/name
        best=out/'weights/best.pt'
        results_file=out/'results.csv'
        completed_epochs=0
        if results_file.is_file():
            with results_file.open() as handle:
                completed_epochs=sum(1 for _ in csv.DictReader(handle))
        if best.is_file() and completed_epochs>=args.epochs:
            print('Reusing completed run checkpoint:',best,flush=True)
            return best
        if out.exists():
            raise FileExistsError(f'Incomplete training directory requires explicit resume: {out}')
        model.train(data=str(dataset/'dataset.yaml'),epochs=args.epochs,imgsz=640,
                    batch=8,device=0,workers=0,amp=True,seed=17,deterministic=True,
                    pretrained=False,project=str(out.parent),name=out.name,
                    exist_ok=False,patience=args.epochs,plots=True,verbose=False)
        if not best.is_file(): raise RuntimeError(f'No trained checkpoint: {best}')
        del model
        gc.collect(); torch.cuda.empty_cache()
        return best
    def candidates(model,source,out,confidence=.8,stride=6):
        if out.exists():
            done=out/'summary.json'
            if done.is_file(): return json.loads(done.read_text())
            raise FileExistsError(out)
        out.mkdir(parents=True); (out/'images').mkdir(); (out/'labels').mkdir()
        accepted=0; samples=0
        with (out/'provenance.jsonl').open('w',encoding='utf-8') as log:
            for index,r in enumerate(model.predict(source=str(source),stream=True,
                    conf=confidence,imgsz=640,max_det=1,device=0,vid_stride=stride,
                    verbose=False,save=False),1):
                samples+=1
                if r.boxes is None or len(r.boxes)==0: continue
                box=r.boxes[int(r.boxes.conf.argmax().item())]
                xywh=box.xywhn[0].cpu().tolist(); name=f'feedback_{index:08d}'
                if not cv2.imwrite(str(out/'images'/(name+'.png')),r.orig_img):
                    raise OSError('Could not write feedback image')
                (out/'labels'/(name+'.txt')).write_text('0 '+' '.join(f'{v:.8f}' for v in xywh)+'\n')
                log.write(json.dumps({'image':name+'.png','source':str(source),
                    'sample_index':index,'stride':stride,'confidence':float(box.conf.item()),
                    'threshold':confidence,'label_origin':'fresh_yolov8_iteration_01',
                    'model':str(summary['iteration_01_model'])})+'\n')
                accepted+=1
        info={'accepted':accepted,'sampled':samples,'threshold':confidence,'source':str(source)}
        write_json(out/'summary.json',info)
        return info
    def assess(checkpoint,source,name):
        model=YOLO(str(checkpoint))
        scores=[]; boxes=[]
        for r in model.predict(source=str(source),stream=True,conf=.01,imgsz=640,
                max_det=1,device=0,vid_stride=6,verbose=False,save=False):
            score=float(r.boxes.conf.max().item()) if r.boxes is not None and len(r.boxes) else 0.0
            scores.append(score)
        return {'model':str(checkpoint),'source':str(source),'sampled_frames':len(scores),
                'frames_with_score_ge_0_5':sum(s>=.5 for s in scores),
                'frames_with_score_ge_0_7':sum(s>=.7 for s in scores),
                'mean_best_score_including_no_detection':sum(scores)/max(1,len(scores)),
                'interpretation':'Prediction coverage/confidence, not independently measured accuracy'}
    try:
        status('assemble_bootstrap_candidates')
        assembled=ROOT/'datasets/duck'/(args.run_id+'_assembled_01')
        if not assembled.exists():
            split_dataset(run/'composite_split.json',assembled,'duck')
        status('train_yolov8_iteration_01')
        first=train(YOLO('yolov8n.yaml'),'iteration_01',assembled)
        summary['iteration_01_model']=str(first)
        write_json(run/'python_summary.json',summary)
        status('generate_yolov8_feedback')
        source=Path(cfg['original_matlab_root'])/'capture_movie_2024_04_28_18_41_48.mp4'
        feedback=ROOT/'datasets/duck'/(args.run_id+'_yolov8_feedback')
        info=candidates(YOLO(str(first)),source,feedback)
        summary['feedback']=info
        if info['accepted']<5:
            summary['feedback_new_recording']=info
            print('New recording has too few confident labels; expanding the original training recording instead',flush=True)
            source=Path(cfg['original_matlab_root'])/'duck1.mp4'
            feedback=ROOT/'datasets/duck'/(args.run_id+'_yolov8_feedback_training_recording')
            info=candidates(YOLO(str(first)),source,feedback,confidence=.8,stride=3)
            summary['feedback']=info
            if info['accepted']<5:
                raise RuntimeError('Too few confident feedback labels in both recordings')
        items=json.loads((run/'composite_split.json').read_text())
        items.append({'directory':str(feedback),'split':'train'})
        manifest=run/'feedback_split.json'; write_json(manifest,items)
        assembled2=ROOT/'datasets/duck'/(args.run_id+'_assembled_02')
        if not assembled2.exists(): split_dataset(manifest,assembled2,'duck')
        status('train_yolov8_iteration_02')
        second=train(YOLO(str(first)),'iteration_02',assembled2)
        summary['iteration_02_model']=str(second)
        status('evaluate_fixed_recording')
        validation_source=Path(cfg['original_matlab_root'])/'duck-test.mp4'
        summary['prediction_comparison']=[assess(first,validation_source,'iteration_01'),
                                          assess(second,validation_source,'iteration_02')]
        status('export_final_detection_video')
        final=YOLO(str(second)); video=cv2.VideoCapture(str(validation_source))
        fps=video.get(cv2.CAP_PROP_FPS); video.release()
        output=run/'yellow_duck_fresh_iteration_02.mp4'
        writer=cv2.VideoWriter(str(output),cv2.VideoWriter_fourcc(*'mp4v'),fps,(640,480))
        if not writer.isOpened(): raise RuntimeError('Video writer failed')
        n=0
        try:
            for r in final.predict(source=str(validation_source),stream=True,conf=.5,
                    imgsz=640,max_det=1,device=0,verbose=False,save=False):
                writer.write(cv2.resize(r.plot(),(640,480))); n+=1
                if n%300==0: print('Video frames',n,flush=True)
        finally: writer.release()
        summary['final_video']=str(output); summary['final_video_frames']=n
        summary['elapsed_seconds']=time.time()-started
        write_json(run/'python_summary.json',summary)
        status('complete',summary=str(run/'python_summary.json'))
    except Exception as error:
        import traceback
        status('failed',error=str(error),traceback=traceback.format_exc())
        raise

if __name__=='__main__': main()
