# Official PXDesign / FAMPNN integration

Branch base: `17120fa` (stage4-fampnn). Merged Stage III: `d15de33`.
The newer commits in the supplied Stage III checkout are intentionally excluded.

Change set 1: semantic merge

- Kept FAMPNN state, AA adapter, shared packing and per-call q-feedback snapshots.
- Preserved native PXDesign sampling, AlphaProteo-10 tools, diagnostics, retry fixes.
- PXDesign is pinned to official `f788441313c84c3074fe9596ac2433f96b15c763`.
  Stage III's gitlink `2202ad0` cannot be fetched from the official remote.
  Apply `patches/pxdesign-embedders-protenix-2.0.patch` after submodule initialization.
  Protenix remains `c3bfc365b3e1341a11935eddfe7bfdc308092147`.
- Validation: Python compilation succeeded; all 610 tests collected; focused
  merge/state/feedback/data/native-sampling regressions: 74 passed, 2 skipped.
  Checkpoint-dependent upstream parity and GPU release gates remain outstanding.

Validation environment: `PYTHONPATH=PXDesign:Protenix LAYERNORM_TYPE=torch`, Python
3.11 from the existing `ml` environment. Runtime tests must also expose the pinned
FAMPNN source and the CCD data root.
