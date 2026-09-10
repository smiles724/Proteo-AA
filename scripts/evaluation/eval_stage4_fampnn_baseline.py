#!/usr/bin/env python3
"""Frozen FaMPNN comparisons on identical saved predicted-backbone states.

The states_manifest.csv written by eval_stage4_codesign.py can be used directly.
Each trusted .pt has separate 'state' and 'targets' objects. This also accepts
externally prepared states for the exact backbone set used by another AA model.
No backbone generation, packing, or model fitting occurs in this entry point.
"""
import argparse
import csv
from dataclasses import fields, replace
import hashlib
import json
from pathlib import Path
import sys
import torch


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--training-clusters", required=True)
    parser.add_argument("--fampnn-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--query-fraction", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=0.)
    parser.add_argument("--decode-blocks", type=int, default=4)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(root))
    from pxdesign_train.aa.fampnn_head import FaMPNNHead
    from pxdesign_train.aa.atom_mapping import AA_ORDER
    from pxdesign_train.codesign import CycleConfig, decode, random_query
    from pxdesign_train.stage4 import masked_aa_objective
    head = FaMPNNHead(args.fampnn_checkpoint).to(args.device).eval()
    head.requires_grad_(False)
    config = CycleConfig(rounds=1, decode_blocks=args.decode_blocks,
                         query_fraction=args.query_fraction, whole_mask_probability=0., temperature=args.temperature)
    manifest = Path(args.manifest).resolve()
    training = {k: set(map(str,v)) for k,v in json.loads(Path(args.training_clusters).read_text()).items()}
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    metrics = []
    for index, row in enumerate(csv.DictReader(manifest.open())):
        if row["split"] not in ("val", "test") or row["cluster_id"] in training[row["source"]]:
            raise ValueError(f"State is not held out: {row}")
        path = manifest.parent / row["path"]
        saved = torch.load(path, map_location=args.device, weights_only=False)
        state, targets = saved["state"], saved["targets"]
        state = replace(state, **{f.name:getattr(state,f.name).to(args.device) for f in fields(state) if torch.is_tensor(getattr(state,f.name))})
        native = targets["native_aa"].to(args.device)
        with path.open("rb") as stream:
            checksum = hashlib.file_digest(stream, "sha256").hexdigest()
        torch.manual_seed(args.seed + index)
        query = random_query(state.design_mask, config)
        base = replace(state, query_mask=query)
        for mode in ("binder_sc_hidden", "generated_binder_sc"):
            torch.manual_seed(args.seed + index)
            logits,_ = head(**base.aa_input(sc_feedback=mode == "generated_binder_sc"))
            nll, recovery = masked_aa_objective([(logits, query)], native)
            metrics.append(dict(sample_id=row["sample_id"], source=row["source"], cluster_id=row["cluster_id"],
                state_sha256=checksum, mode=mode, nll=float(nll), recovery=float(recovery),
                queried=int(query.sum()), seed=args.seed + index, temperature=args.temperature))
        # Free sequence decoding starts with all design identities and SC hidden.
        # Fixed receptor context is identical to the conditional comparisons.
        torch.manual_seed(args.seed + index)
        free = replace(state, query_mask=state.design_mask)
        free, records = decode(free, head, replace(config, sc_to_aa=False))
        valid = state.design_mask & (native >= 0) & (native < 20)
        recovery = (free.assigned_aa[valid] == native[valid]).float().mean()
        nll,_ = masked_aa_objective(records, native)
        metrics.append(dict(sample_id=row["sample_id"], source=row["source"], cluster_id=row["cluster_id"],
            state_sha256=checksum, mode="free_sequence", nll=float(nll), recovery=float(recovery),
            queried=int(valid.sum()), seed=args.seed + index, temperature=args.temperature))
        with (output / f"sample{index:05d}.fasta").open("w") as stream:
            for b in range(free.assigned_aa.shape[0]):
                for s in range(free.assigned_aa.shape[1]):
                    sequence = "".join(AA_ORDER[int(v)] for v in free.assigned_aa[b,s][free.design_mask[b,s]])
                    stream.write(f">{row['sample_id']} batch={b} sample={s}\n{sequence}\n")
    if metrics:
        with (output / "metrics.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(metrics[0])); writer.writeheader(); writer.writerows(metrics)
    (output / "protocol.json").write_text(json.dumps(dict(head=head.identity, arguments=vars(args),
        backbone_policy="Identical supplied predicted states in every comparison; no packing or backbone updates",
        teacher_context=False, confidence_metric=False), indent=2))


if __name__ == "__main__":
    main()
