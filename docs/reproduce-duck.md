# Reproduce the yellow-duck workflow from captured video

This run regenerates annotations and trains project-specific models from scratch. It does not use historical duck labels or duck model weights. The supplied MATLAB layer graph has empty convolution weights and is initialized for training; YOLOv8 starts from architecture YAML with pretrained=False. A supplied generic YOLOv8n checkpoint is used only for Ultralytics' AMP numerical check, not for training initialization.

## Inputs

Configure the original MATLAB source folder in configs/project.json. Required videos are duck.mp4 (captured synthetic render), background.mp4 (unlabeled room footage), duck1.mp4 (real training recording), duck-test.mp4 (separate real validation recording), and capture_movie_2024_04_28_18_41_48.mp4 (additional feedback footage).

Use your MATLAB installation's yolov2VehicleDetector.mat layer graph. Python needs a working CUDA PyTorch/Ultralytics environment. The reproduced environment was Python 3.12.2, PyTorch 2.6.0+cu124, and Ultralytics 8.1.33. The runner contains a process-local torch.load compatibility setting for trusted local checkpoints and does not modify installed packages.

## Run

Choose a new run identifier, then in MATLAB:

```matlab
addpath(fullfile(pwd,'matlab'));
runId = 'my_duck_run_001';
duckScratchSynthetic(runId);
duckScratchComposite(runId);
duckScratchBootstrapComposite(runId);
duckScratchAssessBootstrap(runId);
duckScratchCandidates(runId);
```

Then use the CUDA-enabled Python interpreter:

```powershell
python python/run_duck_scratch.py --run-id my_duck_run_001 --epochs 50
```

Each of the two YOLOv8 stages trains for the specified epoch count. The second initializes from this run's first model and mixes the original generated training data with new high-confidence feedback predictions. Validation stays in its original recording partition. If the additional recording yields fewer than five confident predictions, the runner records that result and attempts to expand the original training recording instead; it never adds validation images to training.

## Automatic annotations and the bootstrap gap

The synthetic preparation samples every third render frame, separates the yellow/orange duck from the black background/player UI, and places the isolated duck at varied sizes/positions. It saves boxes computed from that operation. Compositing preserves those boxes and replaces the background on 80 percent of examples with unlabeled room frames, with automatic brightness changes and horizontal background flips. The remaining examples stay black.

The black-background-only baseline is available separately with duckScratchBootstrap. In the first reproduction it trained successfully but produced only one real training candidate and zero validation candidates at confidence 0.70. It was excluded from YOLOv8 training. Compositing was therefore added as a documented adaptation, not a claim that the historical black-only result reproduced exactly.

The composite detector's real-scene predictions are selected at confidence 0.70, every sixth frame. The historical duck export used 0.50; the stricter cutoff and new synthetic preparation make this a documented reproduction/adaptation rather than a bit-for-bit replay. The high-capacity training set also retains the synthetic composites to supplement sparse real candidates. YOLOv8 feedback uses confidence 0.80. All labels remain automatically generated; no boxes are drawn or corrected manually.

## Results and limits

Every dataset/run is separate and records provenance. Outputs are under datasets/duck/<run-id>_* and runs/duck/<run-id>, with YOLOv2 checkpoints under runs/matlab/. Existing output directories are protected against replacement. A partially completed YOLOv8 training directory requires explicit recovery rather than being treated as a completed model merely because best.pt exists.

Validation AP measures agreement with the automatically generated boxes, not independent real-scene detection accuracy. Prediction coverage/confidence and rendered videos provide additional diagnostics. Inspect final localization and failure cases before claiming improved robustness.
