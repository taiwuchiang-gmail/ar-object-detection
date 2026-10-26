# Yellow duck reproduction

**No manual image annotation is required.** All project-specific training boxes are generated automatically: foreground extraction labels synthetic renders, YOLOv2 labels real-scene candidates, and YOLOv8 supplies labels for subsequent iterations. Scanning/rendering and choosing confidence thresholds do not involve drawing bounding boxes.

1. Scan the real duck in KIRI Engine on iPhone and export its 3D model. This is an external step; no scan or renderer project was supplied.
2. Render varied views on black backgrounds. Existing duck.mp4 and duck_cropped.mp4 permit starting here.
3. Inspect cropping and automatic labels. The historical duck_cropped.mp4 midpoint cuts off part of the duck. Its frame rate/duration differs from the original recording. Review these before regenerating training data.
4. Run `ar.prepareSynthetic('duck',renderedVideoPath,'render_001')` to produce full images/labels plus separate crops. Full-frame labels belong to full images; cropped images require their own coordinates.
5. Run `ar.trainBootstrap('duck',datasetDir,baseNetworkPath,'bootstrap_001')`. A gTruth MAT path containing automatically generated labels may be supplied instead of datasetDir; gTruth is MATLAB's label container, not a requirement for manual annotation.
6. Run `ar.exportCandidates('duck',realVideoPath,'recording_a_yolov2_001',modelPath,0.5)` to retain confident YOLOv2 predictions.
7. Assemble recording-level splits, train YOLOv8, and export a fresh candidate iteration. The partitions can use automatically generated labels. Optional human review is a separate quality-control step, not a training prerequisite. When evaluating against automatic labels, scores measure agreement with those labels rather than independently established detection accuracy.

## Verified baseline

MATLAB R2026a Update 3: all three geometry tests passed. The historical duck detector loaded with 128 x 128 input and one duck class. The old ground-truth video reference was relocated in a separate local copy; it has 2,633 label rows.

CPU inference on midpoint frames returned best scores 0.9047 (duck_cropped.mp4), 0.5798 (duck-test.mp4), and 0.3456 (duck1.mp4). A diagnostic threshold of 0.01 exposed the last prediction; it does not meet the original 0.5 candidate threshold. These sample scores are not full-video accuracy measurements.

Checkpoint scripts: duckStep01, duckBaselinePreview, duckDiagnosticPreview. Run them in order after configuring local assets and adding matlab/ to the path. Their outputs are local under runs/duck/step01_preflight. Existing output directories are protected against replacement.

Full retraining, Python model compatibility, and glasses connectivity remain to be reproduced. The small bootstrap model's generalization advantage is the original experimental observation to evaluate, not a universal claim.
