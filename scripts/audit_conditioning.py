"""Audit what target conditioning actually reaches the denoiser.

PXDesign conditions on a target through three channels, none of which is a
coordinate overwrite:

  * ``conditional_templ`` / ``conditional_templ_mask`` -- binned pair distances
    (64 bins over 2..22 A) over *resolved, non-design* tokens, embedded into
    ``z`` by ``ConditionTemplateEmbedder``. Masked pairs read bin 0.
  * ``restype`` -- 32 + 4 classes; design tokens carry the extra "to be
    designed" classes.
  * ``hotspot`` -- one scalar per token.

If the design region covers every token there is no condition token left, the
featurizer returns the all-zero template, and the model is generating a
monomer that happens to be scored against a target it never saw. This script
measures that rather than assuming it either way.
"""

import argparse
import json
import sys

import pandas as pd
import torch

from pxf.backbone.driver import featurize_structures, to_featurized


def summarize(name, feature_dict, design):
    out = {"target": name}
    n = int(design.reshape(-1).shape[0])
    d = design.reshape(-1).bool()
    out["n_token"] = n
    out["n_design"] = int(d.sum())
    out["n_condition"] = int((~d).sum())

    asym = feature_dict.get("asym_id")
    if asym is not None:
        a = asym.reshape(-1)[:n]
        out["chains"] = {int(c): int((a == c).sum()) for c in a.unique()}
        out["design_chains"] = sorted({int(c) for c in a[d].unique()})
        out["condition_chains"] = sorted({int(c) for c in a[~d].unique()})

    ct = feature_dict.get("conditional_templ")
    cm = feature_dict.get("conditional_templ_mask")
    if ct is None or cm is None:
        out["conditional_templ"] = "ABSENT FROM FEATURE DICT"
        return out

    ct = ct.reshape(ct.shape[-2], ct.shape[-1])[:n, :n]
    cm = cm.reshape(cm.shape[-2], cm.shape[-1])[:n, :n].bool()
    out["templ_mask_pairs"] = int(cm.sum())
    out["templ_mask_frac"] = round(float(cm.float().mean()), 6)
    out["templ_nonzero_bins"] = int((ct != 0).sum())
    if int(cm.sum()):
        out["templ_bin_min"] = int(ct[cm].min())
        out["templ_bin_max"] = int(ct[cm].max())

    # Where does the template actually live: target-target, or leaking onto design?
    tt = (~d)[:, None] & (~d)[None, :]
    dd = d[:, None] & d[None, :]
    cross = ~(tt | dd)
    out["templ_on_target_target"] = int((cm & tt).sum())
    out["templ_on_design_design"] = int((cm & dd).sum())
    out["templ_on_cross"] = int((cm & cross).sum())
    out["target_target_pairs_total"] = int(tt.sum())

    hot = feature_dict.get("hotspot")
    out["hotspot_sum"] = None if hot is None else float(hot.reshape(-1)[:n].sum())

    rt = feature_dict.get("restype")
    if rt is not None:
        rt = rt.reshape(-1, rt.shape[-1])[:n] if rt.dim() > 1 else rt.reshape(-1)[:n]
        if rt.dim() > 1:
            out["restype_dim"] = int(rt.shape[-1])
            cls = rt.argmax(-1)
            out["restype_design_classes"] = sorted({int(c) for c in cls[d].unique()})[:8]
            out["restype_cond_classes_n"] = len({int(c) for c in cls[~d].unique()})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepared", required=True)
    ap.add_argument("--max-targets", type=int, default=2)
    ap.add_argument("--crop-size", type=int, default=384)
    ap.add_argument("--proteoaa-root", default=None)
    args = ap.parse_args()

    frame = pd.read_parquet(args.prepared)
    rows = []
    for entry in list(frame.itertuples())[: args.max_targets]:
        sid, dataset = featurize_structures(
            [entry.cif_path],
            crop_size=args.crop_size,
            binder_chain_ids=[entry.converted_binder_chain],
            parser_dataset="Distillation",
            proteoaa_root=args.proteoaa_root,
        )[0]
        item = dataset[0]
        structure = to_featurized(sid, item)
        rows.append(
            summarize(
                str(entry.example_id),
                item["input_feature_dict"],
                structure.design_mask,
            )
        )
        print(json.dumps(rows[-1], indent=2), flush=True)

    print("\n=== VERDICT ===")
    for r in rows:
        if r.get("conditional_templ") == "ABSENT FROM FEATURE DICT":
            print(f"{r['target']}: NO conditional_templ feature at all")
        elif r.get("templ_mask_pairs", 0) == 0:
            print(f"{r['target']}: template present but EMPTY -> no target conditioning")
        else:
            print(
                f"{r['target']}: conditioned on {r['n_condition']} target tokens, "
                f"{r['templ_mask_pairs']} masked pairs "
                f"({r['templ_on_target_target']} target-target)"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
