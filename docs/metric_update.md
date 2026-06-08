# Metric Update: Standard AU-PRO

**Date:** 2026-05-14

This note documents the correction to the point-level AU-PRO evaluation used by this repository.

## What Changed

The original evaluation code reported a point-coverage-style AUC: points were sorted by anomaly score, and the metric integrated cumulative anomalous-point coverage as more points were selected. Although this ranking-based number is useful as a diagnostic, it is not the standard Area Under the Per-Region Overlap curve.

The corrected implementation in `test_standard_aupro.py` computes standard AU-PRO by:

1. Building connected ground-truth anomalous regions.
2. Sweeping anomaly-score thresholds.
3. Computing per-region overlap at each threshold.
4. Computing false positive rate from normal points.
5. Integrating the PRO-FPR curve up to the configured FPR limit, default `0.3`.

## Which Script To Use

Use `test_standard_aupro.py` for standard AU-PRO reporting.

`test.py` is retained only for compatibility with the previous point-coverage-style metric and should not be cited as standard AU-PRO.

## Impact

The corrected AU-PRO values are lower than the originally reported point-coverage-style values, as expected from the stricter thresholded region-overlap definition. The correction does not change the main methodological conclusions: BTP remains effective for point-level 3D anomaly localization, and the main experimental trends remain consistent.
