"""Phase 5 Step C: Sentinel-1 SAR as corroborating evidence (never trained fusion).

We have no labelled optical+SAR change dataset, so there is no fusion model to
train - that would be inventing weights. Instead:

* ``backscatter``  - speckle-filtered VV / VH dB backscatter *change* between
  two Sentinel-1 GRD acquisitions. The un-applied absolute calibration cancels
  in a same-orbit dB difference, so change is ``10*log10(I_later / I_earlier)``.
* ``evidence``     - map an optical change candidate's footprint to the
  co-located SAR dB change and turn it into a corroboration factor for
  :mod:`geoseek.change.confidence`: agreement raises confidence, disagreement
  lowers it, SAR-unavailable is exactly neutral. Never an override.
"""

from geoseek.sar.backscatter import ENL_IW_GRDH, LEE_WINDOW, db_change, lee_filter

__all__ = ["lee_filter", "db_change", "ENL_IW_GRDH", "LEE_WINDOW"]
