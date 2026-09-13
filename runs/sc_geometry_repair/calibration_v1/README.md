# Frozen geometry-repair data artifacts

The YAML and four CSV manifests in this directory are the immutable inputs for
the first `sc_geometry_repair` experiment. Their content hashes are recorded in
`calibration.yaml` and `manifest.json`.

`native-*.pt` files are a 308 MB regenerable cache of the 32 calibration
examples, so they are stored at
`/hai/scratch/yfsun/proteo_aa_runs/sc_geometry_repair/calibration_v1` rather
than in Git. The training and evaluation launchers do not need these tensors.
The GPU acceptance gate reads them through `SC_REPAIR_CALIBRATION_SAMPLE_DIR`.
They can be rebuilt with `prepare_sc_repair_calibration.py`, then audited with:

```bash
python scripts/utilities/calibrate_sc_geometry.py \
  --manifest runs/sc_geometry_repair/calibration_v1/manifest.json \
  --sample-root /path/to/native-cache \
  --output /tmp/recomputed-calibration.yaml
```
