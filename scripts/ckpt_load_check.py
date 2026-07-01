#!/usr/bin/env python3
"""Check the official PXDesign-d checkpoint loads into ProtenixDesignTrain with
strict=False (our residue-type head is new, so missing keys are expected)."""
import argparse
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    from protenix.config.config import parse_configs
    from pxdesign_train.configs.configs_train import training_configs
    from pxdesign_train.model import ProtenixDesignTrain

    configs = parse_configs(training_configs, arg_str="")
    configs.load_strict = False
    print("building model...")
    model = ProtenixDesignTrain(configs).to(args.device)
    n = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"model built: {n:.1f}M params")

    obj = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    if isinstance(obj, dict):
        # unwrap common containers
        for k in ("model", "state_dict", "ema", "module"):
            if k in obj and isinstance(obj[k], dict):
                print(f"checkpoint top-level dict; using ['{k}']")
                state = obj[k]
                break
        else:
            state = obj
    else:
        state = obj
    print(f"checkpoint tensors: {len(state)}")

    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"\nMISSING keys (in model, not in ckpt): {len(missing)}")
    for k in missing[:12]:
        print("  -", k)
    print(f"\nUNEXPECTED keys (in ckpt, not in model): {len(unexpected)}")
    for k in unexpected[:12]:
        print("  -", k)

    # Sanity: are the missing keys exactly our new heads?
    new_head_missing = [k for k in missing if "design_residue_type_head" in k
                        or "design_distogram_head" in k or "design_diffusion_distogram" in k]
    print(f"\nmissing that ARE our new heads: {len(new_head_missing)}")
    other_missing = [k for k in missing if k not in new_head_missing]
    print(f"missing that are NOT our heads: {len(other_missing)}")
    for k in other_missing[:12]:
        print("   !", k)
    print("\nLOAD CHECK DONE")


if __name__ == "__main__":
    main()
