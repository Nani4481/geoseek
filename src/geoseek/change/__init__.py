"""Phase 3a: temporal pair preparation.

Make the 2019 / 2024 Sentinel-2 image pair genuinely comparable *before* any
change detection is attempted:

* ``coregister``  - confirm (and, if needed, correct) sub-pixel spatial
  alignment between the two dates via FFT phase correlation.
* ``normalize``   - relative radiometric normalization: fit a per-band linear
  gain/offset on pseudo-invariant pixels so unchanged ground reads the same
  value at both dates (kills the ~300-450 DN scene-wide inter-date bias).
* ``indices``     - NDVI / NDWI / NDBI spectral indices, computed per tile per
  date on the normalized reflectance.

There is deliberately NO change detection in this package yet - that is
Phase 3b. Everything here is offline (reads only from ``data/datasets/``).
"""
