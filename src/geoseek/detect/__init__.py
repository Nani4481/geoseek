"""Phase 8F-2: object-detection track (DOTA -> YOLO-OBB chips -> train -> evaluate -> ObjectDetectionModel).

  geoseek.detect.classes    - class subset, label-source decision, reporting groups, imbalance strategy
  geoseek.detect.chipping   - pure chip / box-clipping geometry (no I/O; unit-tested directly)
  geoseek.detect.sampling   - repeat-factor sampling for the train list (effect measured, not assumed)
  geoseek.detect.convert    - two-pass DOTA -> YOLO-OBB conversion into train / monitor / official-val splits

Nothing here imports ``ultralytics`` (AGPL-3.0) at module import time: the framework is only touched by the
scripts and by :mod:`geoseek.models.yolo_obb`, behind the ``ObjectDetectionModel`` interface, so the rest of
geoseek stays independent of it.
"""
