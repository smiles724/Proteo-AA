#!/usr/bin/env python3
"""The Dunbrack template baseline, in the SAME metric the packer is scored in.

WHY THIS EXISTS. There are already two template-baseline numbers in this repo and
they disagree by almost a factor of two, because they are different quantities:

  * `scripts/sidechain/eval_template_quality.py` reports **1.277 A** --
    mean over residues of each residue's local-frame RMSD, on a hand-picked
    33-chain set, with no symmetry alignment.
  * `scripts/evaluation/eval_sidechain_template_baseline.py` reports
    **atom_weighted_rmse = 2.18 A** over 491 proteins -- sqrt of the global
    mean squared error, again with no symmetry alignment.

The packer is scored by `sidechain/metrics.diagnose_packing`, whose
`symmetry_rmsd` is sqrt(global MSE) over observed atoms **after** per-residue
symmetry alignment, on the 308 held-out monomers that pass the run's token
filter. That is a third quantity. Comparing the packer's 1.51 A against 1.277 A
is not a comparison at all: sqrt-of-mean is >= mean-of-sqrt (Jensen), the chain
sets differ, and only one of the two aligns symmetric side chains.

So this script computes the template baseline through the SAME mask and the SAME
estimator, on the SAME validation index the run used, and reports all three
conventions side by side so the older numbers stay interpretable.

    python scripts/evaluation/eval_template_baseline_matched.py \
        --filtered-index <run>/cache/recentPDB_monomer_validation_index.csv.gz \
        --output-dir <somewhere>
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", default="/hai/scratch/yfsun/protenix_data")
    p.add_argument("--filtered-index", required=True,
                   help="The run's cached recentPDB_monomer_validation_index.csv.gz")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--num-samples", type=int, default=491)
    p.add_argument("--min-n-token", type=int, default=16)
    p.add_argument("--max-n-token", type=int, default=384)
    p.add_argument("--crop-size", type=int, default=384)
    p.add_argument("--max-crop-retries", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--template-provider", default="dunbrack_mode",
                   choices=["dunbrack_mode", "dunbrack", "ccd", "gaussian"])
    return p.parse_args()


def _default_training_args(training_module):
    saved = sys.argv
    try:
        sys.argv = ["train_protenix_monomer.py"]
        return training_module.parse_args()
    finally:
        sys.argv = saved


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "training"))
    import train_protenix_monomer as training

    t = _default_training_args(training)
    t.training_stage = "sidechain_warmup"
    t.data_root = args.data_root
    t.output_dir = args.output_dir
    t.eval_interval = 1
    t.eval_samples = int(args.num_samples)
    t.eval_filtered_index = args.filtered_index
    t.min_n_token = int(args.min_n_token)
    t.max_n_token = int(args.max_n_token)
    t.crop_size = int(args.crop_size)
    t.max_crop_retries = int(args.max_crop_retries)
    t.eval_num_workers = int(args.num_workers)
    training.apply_training_stage_args(t)
    training._bootstrap_paths(t)

    os.environ.setdefault("PROTENIX_ROOT_DIR", str(Path(args.data_root).resolve()))
    os.environ.setdefault("LAYERNORM_TYPE", "torch")

    import torch

    from pxdesign_train.sidechain.frames import phi_psi_from_ncac
    from pxdesign_train.sidechain.losses import symmetry_align_prediction
    from pxdesign_train.sidechain.templates import PROVIDERS

    provider = PROVIDERS[args.template_provider]
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    loader, n_eval, index_path = training.build_eval_dataloader(t, out_dir)
    if loader is None:
        raise RuntimeError("Validation loader was not constructed")

    raw_sq = sym_sq = 0.0
    atoms = 0
    per_residue = []          # each residue's own RMSD, the third convention
    chi1_hits = chi1_total = 0
    rows = []

    for i, batch in enumerate(loader):
        feat = batch["input_feature_dict"]
        types = feat["aa_clean"].cpu().long()
        gt_local = feat["sc_gt_local"].cpu().float()
        bb = feat["sc_bb_coords"].cpu().float()
        bb_idx = feat["sc_bb_atom_idx"].cpu().long()

        # THE SAME MASK the training eval uses: observed side-chain atoms, with a
        # valid native frame, on design-owned canonical residues.
        mask = feat["sc_atom_mask"].cpu().bool()
        if "sc_frame_valid" in feat:
            mask = mask & feat["sc_frame_valid"].cpu().bool()[..., None]
        if "design_token_mask" in feat:
            mask = mask & feat["design_token_mask"].cpu().bool()[..., None]
        mask = mask & ((types >= 0) & (types < 20))[..., None]

        have = (bb_idx[..., :3] >= 0).all(dim=-1)
        phi, psi = phi_psi_from_ncac(
            bb[..., 0, :], bb[..., 1, :], bb[..., 2, :],
            feat["residue_index"].cpu(), feat["asym_id"].cpu(), have=have)

        template, t_mask = provider(types.clamp(0, 19), phi=phi, psi=psi)
        template = template.cpu().float()
        valid = mask & t_mask.cpu().bool()
        n = int(valid.sum())
        if n == 0:
            continue

        aligned = symmetry_align_prediction(
            template[None], gt_local[None], valid[None], types[None])[0]
        raw = float((((template - gt_local) ** 2).sum(-1) * valid).sum())
        sym = float((((aligned - gt_local) ** 2).sum(-1) * valid).sum())
        raw_sq += raw
        sym_sq += sym
        atoms += n

        # Per-residue RMSD, the convention eval_template_quality.py reports.
        d2 = (((aligned - gt_local) ** 2).sum(-1) * valid).sum(-1)
        cnt = valid.sum(-1)
        keep = cnt > 0
        per_residue.extend(torch.sqrt(d2[keep] / cnt[keep]).tolist())

        # chi1 recovery, same 40 degree criterion the metrics module uses.
        from pxdesign_train.sidechain.buildsc import chi_from_local
        from pxdesign_train.sidechain.chi_constants import CHI_MASK
        pred_chi1 = chi_from_local(types.clamp(0, 19), aligned)[:, 0]
        gt_chi1 = chi_from_local(types.clamp(0, 19), gt_local)[:, 0]
        ok = CHI_MASK[types.clamp(0, 19)][:, 0] & keep & torch.isfinite(pred_chi1) & torch.isfinite(gt_chi1)
        delta = torch.atan2((pred_chi1 - gt_chi1).sin(), (pred_chi1 - gt_chi1).cos()).abs()
        chi1_hits += int(((delta < math.radians(40)) & ok).sum())
        chi1_total += int(ok.sum())

        rows.append(dict(index=i, atoms=n, raw_rmsd=math.sqrt(raw / n),
                         sym_rmsd=math.sqrt(sym / n)))

    pr = torch.tensor(per_residue)
    record = dict(
        provider=args.template_provider,
        chains=len(rows), validation_index=str(index_path), requested=n_eval,
        observed_atoms_total=atoms,
        observed_atoms_per_chain=atoms / max(len(rows), 1),
        # THE comparable number: same estimator as metrics.summarize_metrics.
        symmetry_rmsd=math.sqrt(sym_sq / atoms),
        atom_weighted_rmsd_no_symmetry=math.sqrt(raw_sq / atoms),
        mean_per_residue_rmsd=float(pr.mean()),
        median_per_residue_rmsd=float(pr.median()),
        chi1_recovery_40deg=chi1_hits / max(chi1_total, 1),
        chi1_counted=chi1_total,
    )
    (out_dir / "template_baseline_matched.json").write_text(
        json.dumps(dict(record, per_chain=rows), indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
