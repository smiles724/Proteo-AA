"""Command-line entrypoint for the paired rotation ablation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .bundle import load_bundle
from .experiment import RotationAblation
from .factory import modules_from_checkpoint, random_modules, synthetic_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="", help="Training checkpoint containing S_phi weights")
    parser.add_argument("--bundle", default="", help="Captured SidechainInputBundle .pt")
    parser.add_argument("--output", default="rotation_ablation/results.json")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n-rotations", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--translation-scale", type=float, default=10.0)
    parser.add_argument("--n-heads", type=int, default=16, help="Not inferable from checkpoint tensors")
    parser.add_argument("--synthetic-length", type=int, default=6)
    parser.add_argument("--invariant-tolerance", type=float, default=1e-4)
    parser.add_argument("--coordinate-tolerance", type=float, default=1e-3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")

    if args.checkpoint:
        module, feedback, model_meta = modules_from_checkpoint(
            args.checkpoint, n_heads=args.n_heads
        )
    else:
        module, feedback, model_meta = random_modules(seed=args.seed)
    module = module.to(device)
    feedback = feedback.to(device)

    if args.bundle:
        bundle = load_bundle(args.bundle, map_location=device)
    else:
        bundle = synthetic_bundle(
            c_res=module.w_res.in_features,
            n_type=module.w_aa.in_features,
            length=args.synthetic_length,
            seed=args.seed,
            device=device,
        )
    bundle.metadata = {**(bundle.metadata or {}), **model_meta}

    runner = RotationAblation(
        module,
        feedback,
        bundle,
        invariant_tolerance=args.invariant_tolerance,
        coordinate_tolerance=args.coordinate_tolerance,
    )
    result = runner.run(
        n_rotations=args.n_rotations,
        seed=args.seed,
        translation_scale=args.translation_scale,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=True) + "\n")

    print(f"wrote={output.resolve()}")
    for arm, verdict in result["verdicts"].items():
        status = "SENSITIVE" if verdict["rotation_sensitive"] else "PASS"
        print(
            f"{status:9s} {arm} "
            f"hidden_max={verdict['max_atom_feature_relative_error']:.3e} "
            f"coord_eq_max={verdict['max_coordinate_equivariance_rmsd']:.3e}"
        )
    if not result["has_recomputed_h_res_pair"]:
        print("note: B/D require h_res_q in a captured paired bundle; C is exact and complete")


if __name__ == "__main__":
    main()
