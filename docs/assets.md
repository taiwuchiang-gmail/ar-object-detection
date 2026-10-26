# Local assets and configuration

The repository distributes active source, tests, examples, and documentation. Your recordings, labels, model weights, generated datasets, and run output stay local. Existing installations retain them in place.

On a fresh clone, run `python python/setup_local.py --matlab-root "PATH_TO_ORIGINAL_MATLAB_FOLDER" --python-root "PATH_TO_ORIGINAL_PYTHON_FOLDER"`. The setup tool creates an ignored `configs/project.json`, copies selected duck/bike/laser ground truth and detector files into the ignored data/models directories, and reports missing assets. It does not modify the source directories or overwrite an existing destination.

Dataset generation creates directories under datasets/; training and prediction create directories under runs/. Iteration names must be new. Keep each recording session entirely within one train/validation/test partition, including all feedback iterations derived from it.

The saved gTruth/ground-truth files are annotation containers and can hold the automatically generated boxes; their name does not imply manual labeling. No manually annotated dataset is required. MATLAB ground truth can embed an old source path. `ar.relocateGroundTruth(inputPath,oldPath,newPath,newOutputPath)` saves a separate copy with corrected references. The original MAT file is preserved.

The YOLOv2 base network can be located with `which('yolov2VehicleDetector.mat')` in MATLAB. Pass that path to `ar.trainBootstrap`; its MAT file must contain lgraph. Its availability depends on the installed MATLAB products/support data.

Selected historical Python checkpoints were runs/detect/train2/weights/best.pt for duck and runs_bike/detect/train5/weights/best.pt for bike. These reflect prior script choices, not a ranking of all historical runs.

# Optional historical integrations

MagicStreamer is excluded: the supplied copy is marked Magic Leap Confidential and Proprietary and is not redistributed. The copied GStreamer experiment and its virtual environments are also kept outside this source repository. The active inference command accepts RTSP directly and does not import those packages. Any separate use of those integrations requires the appropriate software, permissions, and environment.

Historical scripts and migration manifests remain local under archive/, third_party/, and ignored documentation files. They are not active entry points. Their old paths and deletion commands have not been carried into the new workflow.
