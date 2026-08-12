#!/usr/bin/env python3
"""DIAGNOSTIC ONLY: reproduce template MSE with unresolved atoms included.

Normal training and evaluation must use ``eval_sidechain_template_baseline.py``.
This script installs a process-local wrapper around side-chain target extraction
that restores the legacy behavior for one retrospective comparison: an atom row
with ``is_resolved=False`` (normally a global-coordinate ``(0, 0, 0)``
placeholder) is treated as supervised.  No production source file or checkpoint
is modified, and the wrapper disappears when this process exits.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

import eval_sidechain_template_baseline as resolved_baseline


_RESTORED_ATOMS = 0
_RESTORED_UNRESOLVED_ANNOTATION = 0
_RESTORED_ZERO_COORDINATE = 0


def _install_legacy_target_wrapper() -> None:
    """Patch only this process; never change the production featurizer on disk."""
    from pxdesign_train.data.featurizer import DesignFeaturizer
    from pxdesign_train.sidechain.frames import to_local
    from pxdesign_train.sidechain.instantiate import MAX_SC, sidechain_atoms

    safe_target_fn = DesignFeaturizer._compute_sidechain_targets

    def include_unresolved(self, atom_array, feature_dict, binder_atom_mask):
        global _RESTORED_ATOMS
        global _RESTORED_UNRESOLVED_ANNOTATION
        global _RESTORED_ZERO_COORDINATE

        out = safe_target_fn(self, atom_array, feature_dict, binder_atom_mask)
        rep_mask = (
            feature_dict["distogram_rep_atom_mask"].bool().detach().cpu().numpy()
        )
        rep_idx = np.nonzero(rep_mask)[0]
        coord = np.asarray(atom_array.coord, dtype=np.float32)
        atom_name = np.asarray(atom_array.atom_name)
        res_name = np.asarray(atom_array.res_name)
        chain_id = np.asarray(atom_array.chain_id)
        res_id = np.asarray(atom_array.res_id)
        binder = np.asarray(binder_atom_mask, dtype=bool)
        resolved_annotation = (
            np.asarray(atom_array.is_resolved, dtype=bool)
            if "is_resolved" in atom_array.get_annotation_categories()
            else None
        )

        res_atoms: dict[tuple, dict[str, int]] = {}
        for idx in range(len(atom_array)):
            key = (chain_id[idx], res_id[idx])
            res_atoms.setdefault(key, {})[str(atom_name[idx])] = idx

        mask = out["sc_atom_mask"]
        gt_local = out["sc_gt_local"]
        frame_R = out["sc_frame_R"]
        frame_t = out["sc_frame_t"]
        bb_idx = out["sc_bb_atom_idx"]

        for token_idx, rep_atom_idx in enumerate(rep_idx):
            if not binder[rep_atom_idx]:
                continue
            if not bool((bb_idx[token_idx, :3] >= 0).all()):
                continue
            atoms = res_atoms[(chain_id[rep_atom_idx], res_id[rep_atom_idx])]
            for slot, name in enumerate(sidechain_atoms(str(res_name[rep_atom_idx]))):
                if slot >= MAX_SC:
                    break
                if name not in atoms or bool(mask[token_idx, slot]):
                    continue

                # The safe target function masks this row because it is unresolved
                # or has a zero placeholder. Restore precisely the legacy target:
                # transform that stored global coordinate (usually the origin) into
                # the residue frame and supervise it.
                atom_idx = atoms[name]
                global_coord = torch.from_numpy(coord[atom_idx])[None, None]
                local = to_local(
                    global_coord,
                    frame_R[token_idx][None],
                    frame_t[token_idx][None],
                )[0, 0]
                gt_local[token_idx, slot] = local
                mask[token_idx, slot] = True
                _RESTORED_ATOMS += 1
                if resolved_annotation is not None and not resolved_annotation[atom_idx]:
                    _RESTORED_UNRESOLVED_ANNOTATION += 1
                if float(np.abs(coord[atom_idx]).max()) <= 1e-3:
                    _RESTORED_ZERO_COORDINATE += 1

        return out

    DesignFeaturizer._compute_sidechain_targets = include_unresolved


def main() -> None:
    args = resolved_baseline.parse_args()
    training_dir = Path(__file__).resolve().parent.parent / "training"
    sys.path.insert(0, str(training_dir))

    import train_protenix_monomer as training

    train_args = resolved_baseline._default_training_args(training)
    train_args.protenix_code_dir = args.protenix_code_dir
    train_args.pxdesign_code_dir = args.pxdesign_code_dir
    training._bootstrap_paths(train_args)

    _install_legacy_target_wrapper()
    print(
        "WARNING: diagnostic legacy baseline; unresolved placeholder atoms are "
        "intentionally included. Do not use this value for training or model "
        "selection.",
        flush=True,
    )
    resolved_baseline.main()

    metrics_path = Path(args.output_dir).resolve() / "template_baseline.json"
    metrics = json.loads(metrics_path.read_text())
    metrics.update(
        {
            "diagnostic_only": True,
            "includes_unresolved_atoms": True,
            "restored_atoms_total": int(_RESTORED_ATOMS),
            "restored_is_resolved_false_atoms": int(
                _RESTORED_UNRESOLVED_ANNOTATION
            ),
            "restored_zero_coordinate_atoms": int(_RESTORED_ZERO_COORDINATE),
        }
    )
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    print(f"restored_atoms_total={_RESTORED_ATOMS}")
    print(
        "restored_is_resolved_false_atoms="
        f"{_RESTORED_UNRESOLVED_ANNOTATION}"
    )
    print(f"restored_zero_coordinate_atoms={_RESTORED_ZERO_COORDINATE}")
    print(f"rewrote_diagnostic_metadata={metrics_path}")


if __name__ == "__main__":
    main()
