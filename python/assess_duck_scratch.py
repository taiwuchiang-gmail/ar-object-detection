"""Assess fresh models on complete recording samples and export a novel-view video."""
from pathlib import Path
import argparse,json,os

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id',default='scratch_20261026_01')
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    run=root/'runs/duck'/args.run_id
    assert json.loads((run/'python_status.json').read_text())['stage']=='complete'
    os.environ['YOLO_CONFIG_DIR']=str(run/'ultralytics-settings')
    os.environ['MPLCONFIGDIR']=str(run/'matplotlib-cache')
    import cv2,torch
    from ultralytics import YOLO
    from yolo_runtime import configure_local_checkpoint_loading
    configure_local_checkpoint_loading(root); torch.set_num_threads(6)
    cfg=json.loads((root/'configs/project.json').read_text())
    source_root=Path(cfg['original_matlab_root'])
    files=['duck-test.mp4','capture_movie_2024_04_28_18_41_48.mp4']
    result=[]; rows=[]
    for file in files:
        for iteration in [1,2]:
            checkpoint=run/f'yolov8/iteration_{iteration:02d}/weights/best.pt'
            model=YOLO(str(checkpoint)); source=source_root/file
            scores=[]
            for r in model.predict(source=str(source),stream=True,conf=.01,imgsz=640,
                    max_det=1,device=0,vid_stride=6,verbose=False,save=False):
                score=float(r.boxes.conf.max().item()) if len(r.boxes) else 0.0
                scores.append(score)
            record={'recording':file,'iteration':iteration,'sampled':len(scores),
                    'score_ge_0_5':sum(v>=.5 for v in scores),
                    'score_ge_0_7':sum(v>=.7 for v in scores),
                    'score_ge_0_8':sum(v>=.8 for v in scores),
                    'mean_best_returned_score':sum(scores)/len(scores)}
            result.append(record); print(record,flush=True)
            video=cv2.VideoCapture(str(source)); count=video.get(cv2.CAP_PROP_FRAME_COUNT)
            panels=[]
            for fraction in [.2,.5,.8]:
                video.set(cv2.CAP_PROP_POS_FRAMES,int(count*fraction)); ok,image=video.read(); assert ok
                r=model.predict(image,conf=.01,imgsz=640,max_det=1,device=0,verbose=False)[0]
                score=float(r.boxes.conf.max().item()) if len(r.boxes) else 0.0
                if len(r.boxes):
                    box=r.boxes.xyxy[0].cpu().tolist()
                    color=(0,255,0) if score>=.5 else (0,165,255)
                    cv2.rectangle(image,(int(box[0]),int(box[1])),(int(box[2]),int(box[3])),color,3)
                image=cv2.resize(image,(400,300))
                label=f'Iteration {iteration} | {score:.3f}'
                cv2.rectangle(image,(0,0),(400,28),(0,0,0),-1)
                cv2.putText(image,label,(7,20),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1)
                panels.append(image)
            video.release(); rows.append(cv2.hconcat(panels))
            del model; torch.cuda.empty_cache()
    cv2.imwrite(str(run/'final_model_comparison.png'),cv2.vconcat(rows))
    (run/'recording_assessment.json').write_text(json.dumps({'metrics':result,
        'interpretation':'Prediction coverage/confidence only; no independent bounding-box reference',
        'comparison_rows':['validation iteration 1','validation iteration 2','novel recording iteration 1','novel recording iteration 2']},indent=2))
    source=source_root/files[1]; video=cv2.VideoCapture(str(source))
    fps=video.get(cv2.CAP_PROP_FPS); video.release()
    model=YOLO(str(run/'yolov8/iteration_02/weights/best.pt'))
    output=run/'yellow_duck_novel_recording_iteration_02.mp4'
    assert not output.exists(),'Novel video already exists'
    writer=cv2.VideoWriter(str(output),cv2.VideoWriter_fourcc(*'mp4v'),fps,(720,540))
    assert writer.isOpened()
    n=0
    try:
        for r in model.predict(source=str(source),stream=True,conf=.5,imgsz=640,
                max_det=1,device=0,verbose=False,save=False):
            writer.write(cv2.resize(r.plot(),(720,540))); n+=1
    finally: writer.release()
    print('Novel detection video:',output,'frames:',n,flush=True)

if __name__=='__main__': main()
