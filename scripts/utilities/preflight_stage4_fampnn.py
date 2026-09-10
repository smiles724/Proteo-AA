#!/usr/bin/env python3
"""Resolve Stage IV provenance before submitting training or changing optimizers."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import torch


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--donor",required=True)
    parser.add_argument("--fampnn-checkpoint",required=True)
    parser.add_argument("--data-root",required=True)
    parser.add_argument("--output",required=True)
    parser.add_argument("--phase",choices=["IV-A","IV-B","IV-C"],default="IV-A")
    parser.add_argument("--train-rounds",type=int,default=1)
    parser.add_argument("--inference-rounds",type=int,default=3)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root))
    from pxdesign_train.aa.fampnn_head import FaMPNNHead
    donor=Path(args.donor).resolve()
    checkpoint=torch.load(donor,map_location="cpu",weights_only=False,mmap=True)
    state={k.removeprefix("module."):v for k,v in checkpoint["model"].items()}
    for prefix in ("diffusion_module.","sidechain_module."):
        if not any(k.startswith(prefix) for k in state):
            raise ValueError(f"Donor has no {prefix} parameters: {donor}")
    arch=checkpoint.get("sidechain_arch")
    if not arch or "edm" not in arch:
        raise ValueError("Donor lacks the recorded one-step side-chain layout; identify its original configuration first")
    if arch["edm"] or "local_coord_input" in arch:
        raise ValueError("Donor does not use the supported one-step global-coordinate packer")
    if args.train_rounds < 1 or args.inference_rounds < 2:
        raise ValueError("Training requires >=1 round; production inference requires >=2")
    head=FaMPNNHead(args.fampnn_checkpoint)
    data=Path(args.data_root).resolve()
    required=[data/"protenix_data/common/components.cif",data/"pinder/2024-02/indices/pinder_ppi_complex.parquet",data/"pinder/2024-02/raw/pdbs.zip"]
    for path in required:
        if not path.is_file(): raise FileNotFoundError(path)
    revision=subprocess.check_output(["git","-C",str(root),"rev-parse","HEAD"],text=True).strip()
    source_paths=sorted([root/"scripts/training/train_protenix_monomer.py",*root.glob("pxdesign_train/**/*.py"),*root.glob("scripts/training/*stage4*"),*root.glob("scripts/evaluation/*stage4*"),*root.glob("scripts/utilities/*stage4*")])
    source_hash=hashlib.sha256()
    for path in source_paths:
        source_hash.update(str(path.relative_to(root)).encode());source_hash.update(path.read_bytes())
    with donor.open("rb") as stream: donor_sha=hashlib.file_digest(stream,"sha256").hexdigest()
    record=dict(proteoaa_revision=revision,implementation_sha256=source_hash.hexdigest(),donor=str(donor),donor_sha256=donor_sha,
        donor_step=checkpoint.get("step"),donor_sidechain_arch=arch,aa_backend=head.identity,
        arguments=vars(args),sidechain_edm=False,optimizer_policy="fresh AA/SC/BB groups for donor warm start",
        mask_policy="query-X-hide-SC-before-encoding-v1",resolved_configuration="resolved_config.json and arguments.json record effective settings after CLI overrides",early_stopping=False,
        validation="PINDER val clusters excluded from PINDER train; separate recent-PDB monomer retention; cross-source homology not established")
    path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(record,indent=2))
    print("STAGE4_PREFLIGHT_OK",path,flush=True)


if __name__ == "__main__": main()
